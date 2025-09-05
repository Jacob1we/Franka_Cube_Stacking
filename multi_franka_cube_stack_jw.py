# Datei 1: `stacking_runner.py`

```python
from __future__ import annotations

# --- Isaac Sim boot ---
from isaacsim import SimulationApp

# Stdlib
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np

# Isaac/Omniverse
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller import StackingController
from isaacsim.robot.manipulators.examples.franka.tasks import Stacking
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import isaacsim.core.utils.prims as prim_utils
import omni.usd
from pxr import UsdGeom, UsdShade, Gf, Sdf


class IsaacStackingRunner:
    """
    Kapselt die komplette Simulation (App-Init, World/Task/Controller-Aufbau, Loop, Logging)
    in eine Klasse, damit man sie per separatem Launcher-Skript mehrfach parallel starten kann
    (je Prozess genau *eine* SimulationApp!).
    """

    # --- Default-Parameter (können im __init__ überschrieben werden) ---
    SCENE_WIDTH: float = 0.50
    SCENE_LENGTH: float = 0.75
    FRANKA_BASE_CLEARANCE: float = 0.25
    FORWARD_AXIS: str = "x"  # oder "y"
    CUBE_SIDE: float = 0.0515
    MIN_DIST: float | None = 1.2 * CUBE_SIDE
    PLANE_LIFT: float = 0.00
    MAX_TRIES: int = 200

    # Material-Pool
    ALLOWED_AREA_MATS = [
        ("AllowedArea_Steel_Brushed",   (0.62, 0.62, 0.62, 1.00)),
        ("AllowedArea_Aluminum_Mill",   (0.77, 0.77, 0.78, 1.00)),
        ("AllowedArea_Wood_Oak",        (0.65, 0.53, 0.36, 1.00)),
        ("AllowedArea_Wood_BirchPly",   (0.85, 0.74, 0.54, 1.00)),
        ("AllowedArea_Plastic_HDPE_Black",(0.08, 0.08, 0.08, 1.00)),
        ("AllowedArea_Rubber_Mat",      (0.12, 0.12, 0.12, 1.00)),
        ("AllowedArea_Acrylic_Frosted", (1.00, 1.00, 1.00, 1.00)),
    ]

    def __init__(
        self,
        headless: bool = True,
        logdir: Path | str = "./00_my_envs/Franka_Cube_Stacking_JW/logs",
        cam_freq: int = 20,
        cam_res: tuple[int, int] = (256, 256),
        seed: int = 312,
        forward_axis: str | None = None,
        rand_cube_rotation: bool = True,
        rotation_mode: str = "yaw",                 # "yaw" | "xyz"
        yaw_range: tuple[float, float] = (-5.0, 5.0),
        keep_cubes_rotated: bool = False,
        material_seed: int | None = None,
    ) -> None:
        self.headless = bool(headless)
        self.logdir = Path(logdir)
        self.cam_freq = int(cam_freq)
        self.cam_res = tuple(cam_res)
        self.seed = int(seed)
        self.forward_axis = (forward_axis or self.FORWARD_AXIS)
        self.rand_cube_rotation = bool(rand_cube_rotation)
        self.rotation_mode = rotation_mode
        self.yaw_range = tuple(yaw_range)
        self.keep_cubes_rotated = bool(keep_cubes_rotated)
        self.material_seed = material_seed

        # Platzhalter, gefüllt nach build()
        self.simulation_app: SimulationApp | None = None
        self.world: World | None = None
        self.task: Stacking | None = None
        self.robot = None
        self.controller: StackingController | None = None
        self.logger = None
        self._log_cb_holder: dict = {}

    # ------------------ Mathe/Utils ------------------
    @staticmethod
    def quat_to_rot(q):
        w, x, y, z = q
        xx, yy, zz = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        return np.array([
            [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
            [    2*(xy + wz),  1 - 2*(xx + zz),        2*(yz - wx)],
            [    2*(xz - wy),      2*(yz + wx),     1 - 2*(xx + yy)],
        ], dtype=float)

    @classmethod
    def _forward_side_axes_world(cls, base_quat_world, forward_axis='x'):
        R = cls.quat_to_rot(base_quat_world)
        if forward_axis == 'x':
            fwd = R[:, 0]
            side = R[:, 1]
        else:
            fwd = R[:, 1]
            side = R[:, 0]
        fwd  = fwd  / (np.linalg.norm(fwd)  + 1e-12)
        side = side / (np.linalg.norm(side) + 1e-12)
        return fwd, side

    @staticmethod
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=float)

    # ------------------ USD/Material ------------------
    def _get_or_create_color_material(self, stage, name, rgba=(1.0, 0.0, 0.0, 1.0), mat_root="/World/Looks"):
        mat_path = f"{mat_root}/{name}"
        mat = UsdShade.Material.Get(stage, mat_path)
        if not mat:
            mat = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBR")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgba[:3])
            shader.CreateInput("roughness",   Sdf.ValueTypeNames.Float).Set(0.6)
            shader.CreateInput("metallic",    Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("opacity",     Sdf.ValueTypeNames.Float).Set(rgba[3])
            surf_out   = mat.CreateSurfaceOutput()
            shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            surf_out.ConnectToSource(shader_out)
        return mat

    def _ensure_material_pool(self, stage, named_rgba_list, mat_root="/World/Looks"):
        materials = []
        for name, rgba in named_rgba_list:
            materials.append(self._get_or_create_color_material(stage, name=name, rgba=rgba, mat_root=mat_root))
        return materials

    @staticmethod
    def _bind_random_material_to_prim(stage, prim_path, material_list, seed=None):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim {prim_path} nicht gefunden oder ungültig.")
        rng = np.random.default_rng(seed)
        mat = material_list[int(rng.integers(0, len(material_list)))]
        UsdShade.MaterialBindingAPI(prim).Bind(mat)
        return mat

    # ------------------ Geometrie/DR ------------------
    def _add_or_update_allowed_area_plane(
        self,
        stage,
        robot_obj,
        width: float,
        length: float,
        forward_axis: str,
        prim_path: str = "/World/AllowedAreaPlane",
        lift: float = 0.0,
        material_pool_named_rgba = None,
        material_seed = None,
    ):
        base_pos_w, base_quat_w = robot_obj.get_world_pose()
        fwd, side = self._forward_side_axes_world(base_quat_w, forward_axis=forward_axis)

        half_w = width * 0.5
        z_lift = float(base_pos_w[2] + lift)

        def make_point(u_lateral, v_forward):
            p = base_pos_w + u_lateral * side + v_forward * fwd
            return Gf.Vec3f(float(p[0]), float(p[1]), z_lift)

        p0 = make_point(-half_w, 0.0)
        p1 = make_point( half_w, 0.0)
        p2 = make_point( half_w, length)
        p3 = make_point(-half_w, length)

        mesh = UsdGeom.Mesh.Get(stage, prim_path)
        if not mesh:
            mesh = UsdGeom.Mesh.Define(stage, prim_path)

        mesh.CreatePointsAttr([p0, p1, p2, p3])
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2,   0, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateDoubleSidedAttr(True)

        if material_pool_named_rgba:
            mats = self._ensure_material_pool(stage, material_pool_named_rgba)
            self._bind_random_material_to_prim(stage, prim_path, mats, seed=material_seed)

        return mesh

    @staticmethod
    def _get_base_axes_from_local_quat(base_quat_local, forward_axis='x'):
        R = IsaacStackingRunner.quat_to_rot(base_quat_local)
        if forward_axis == 'x':
            x_fwd = R[:, 0] / np.linalg.norm(R[:, 0])
            y_side = R[:, 1] / np.linalg.norm(R[:, 1])
        else:
            x_fwd = R[:, 1] / np.linalg.norm(R[:, 1])
            y_side = R[:, 0] / np.linalg.norm(R[:, 0])
        return x_fwd, y_side

    def _sample_points_in_front_rectangle_local(
        self,
        n: int,
        base_pos_local: np.ndarray,
        base_quat_local: np.ndarray,
        width: float,
        length: float,
        z_levels = None,
        min_dist: float | None = None,
        max_tries: int = 200,
        base_clearance: float = 0.25,
        seed = None,
        forward_axis: str = 'x',
    ):
        assert n >= 1
        assert width > 0.0 and length > 0.0
        assert base_clearance >= 0.0
        if z_levels is not None:
            assert len(z_levels) == n

        rng = np.random.default_rng(seed)
        x_fwd, y_side = self._get_base_axes_from_local_quat(base_quat_local, forward_axis=forward_axis)

        pts = []

        def ok(p_xy):
            if np.linalg.norm(p_xy - base_pos_local[:2]) < base_clearance:
                return False
            if min_dist is not None:
                for q in pts:
                    if np.linalg.norm(p_xy - q[:2]) < min_dist:
                        return False
            return True

        for i in range(n):
            z = z_levels[i] if z_levels is not None else float(base_pos_local[2])
            for _ in range(max_tries):
                u = rng.uniform(-width * 0.5, width * 0.5)
                v = rng.uniform(0.0, length)
                p = base_pos_local + u * y_side + v * x_fwd
                p_xy = p[:2].astype(float)
                if ok(p_xy):
                    pts.append(np.array([p_xy[0], p_xy[1], z], dtype=float))
                    break
            else:
                u = rng.uniform(-width * 0.5, width * 0.5)
                v = rng.uniform(0.0, length)
                p = base_pos_local + u * y_side + v * x_fwd
                p_xy = p[:2].astype(float)
                vec_xy = p_xy - base_pos_local[:2]
                norm = float(np.linalg.norm(vec_xy))
                if norm < base_clearance:
                    if norm < 1e-9:
                        vec_xy = x_fwd[:2]
                        norm = float(np.linalg.norm(vec_xy)) + 1e-12
                    p_xy = base_pos_local[:2] + (vec_xy / norm) * base_clearance
                pts.append(np.array([p_xy[0], p_xy[1], z], dtype=float))
        return np.vstack(pts)

    def _randomize_stacking_in_rectangle_existing_task(
        self,
        task,
        robot_obj,
        width,
        length,
        keep_cubes_z,
        min_dist,
        base_clearance,
        seed,
        forward_axis,
        randomize_rotation: bool = True,
        rotation_mode: str = "yaw",
        yaw_range_deg: tuple = (-180.0, 180.0),
        keep_cubes_rot: bool = False,
        max_tries: int = 200,
    ):
        base_pos_local, base_quat_local = robot_obj.get_local_pose()
        cube_names = task.get_cube_names()
        n_cubes = len(cube_names)

        z_levels = None
        if keep_cubes_z:
            z_levels = []
            for name in cube_names:
                pos_l, _ = task.scene.get_object(name).get_local_pose()
                z_levels.append(float(pos_l[2]))

        starts_local = self._sample_points_in_front_rectangle_local(
            n=n_cubes,
            base_pos_local=base_pos_local,
            base_quat_local=base_quat_local,
            width=width,
            length=length,
            z_levels=z_levels,
            min_dist=min_dist,
            max_tries=max_tries,
            base_clearance=base_clearance,
            seed=seed,
            forward_axis=forward_axis,
        )

        for i, name in enumerate(cube_names):
            cube = task.scene.get_object(name)
            _, current_quat = cube.get_local_pose()
            new_quat = current_quat
            if randomize_rotation and not keep_cubes_rot:
                local_rng = np.random.default_rng(None if seed is None else seed + 100 + i)
                if rotation_mode == "yaw":
                    yaw_deg = float(local_rng.uniform(*yaw_range_deg))
                    q_delta = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, yaw_deg]), degrees=True)
                elif rotation_mode == "xyz":
                    roll_deg  = float(local_rng.uniform(-5.0, 5.0))
                    pitch_deg = float(local_rng.uniform(-5.0, 5.0))
                    yaw_deg   = float(local_rng.uniform(*yaw_range_deg))
                    q_delta = rot_utils.euler_angles_to_quats(np.array([roll_deg, pitch_deg, yaw_deg]), degrees=True)
                else:
                    raise ValueError(f"Unbekannter rotation_mode: {rotation_mode}")
                new_quat = self.quat_mul(current_quat, q_delta)
            cube.set_local_pose(starts_local[i], new_quat)

        target_local = self._sample_points_in_front_rectangle_local(
            n=1,
            base_pos_local=base_pos_local,
            base_quat_local=base_quat_local,
            width=width,
            length=length,
            z_levels=[0.0],
            min_dist=None,
            base_clearance=base_clearance,
            seed=None if seed is None else seed + 1,
            forward_axis=forward_axis,
        )[0]

        task.set_params(
            stack_target_position=np.array([target_local[0], target_local[1], 0.0], dtype=float)
        )

    # ------------------ Logging ------------------
    @staticmethod
    def _make_frame_logging_func(robot_obj):
        def _log(tasks, scene):
            jp = robot_obj.get_joint_positions()
            aa = robot_obj.get_applied_action()
            return {
                "joint_positions": jp.tolist(),
                "applied_joint_positions": (aa.joint_positions.tolist() if aa and aa.joint_positions is not None else []),
            }
        return _log

    @staticmethod
    def _unique_log_path(base_dir: Path, prefix="isaac_sim_data"):
        base_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base_dir / f"{prefix}_{ts}.json"

    def _ensure_logger_running(self, world: World, robot, logger, log_func_ref_container: dict):
        if "func" not in log_func_ref_container:
            log_func_ref_container["func"] = self._make_frame_logging_func(robot)
        logger.add_data_frame_logging_func(log_func_ref_container["func"])
        if not logger.is_started():
            logger.start()

    # ------------------ Aufbau & Loop ------------------
    def _build_world(self):
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()

        task = Stacking()
        world.add_task(task)
        world.reset()

        robot_name = task.get_params()["robot_name"]["value"]
        robot = world.scene.get_object(robot_name)

        stage = omni.usd.get_context().get_stage()
        self._add_or_update_allowed_area_plane(
            stage=stage,
            robot_obj=robot,
            width=self.SCENE_WIDTH,
            length=self.SCENE_LENGTH,
            forward_axis=self.forward_axis,
            lift=self.PLANE_LIFT,
            prim_path="/World/AllowedAreaPlane",
            material_pool_named_rgba=self.ALLOWED_AREA_MATS,
            material_seed=self.material_seed,
        )

        self._randomize_stacking_in_rectangle_existing_task(
            task=task,
            robot_obj=robot,
            width=self.SCENE_WIDTH,
            length=self.SCENE_LENGTH,
            keep_cubes_z=True,
            min_dist=self.MIN_DIST,
            base_clearance=self.FRANKA_BASE_CLEARANCE,
            seed=self.seed,
            forward_axis=self.forward_axis,
            randomize_rotation=self.rand_cube_rotation,
            rotation_mode=self.rotation_mode,
            yaw_range_deg=self.yaw_range,
            keep_cubes_rot=self.keep_cubes_rotated,
            max_tries=self.MAX_TRIES,
        )

        Camera(
            prim_path="/World/camera",
            position=np.array([0.48, -3.6, 1.8]),
            frequency=self.cam_freq,
            resolution=self.cam_res,
            orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 22.5, 90.0]), degrees=True),
        )

        prim_utils.create_prim(
            "/World/Distant_Light",
            "DistantLight",
            position=np.array([1.0, 1.0, 1.0]),
            attributes={"inputs:intensity": 500, "inputs:color": (1.0, 1.0, 1.0)},
        )

        controller = StackingController(
            name="stacking_controller",
            gripper=robot.gripper,
            robot_articulation=robot,
            picking_order_cube_names=task.get_cube_names(),
            robot_observation_name=robot_name,
        )

        return world, task, robot, controller

    def run(self):
        """Startet die Simulation bis zum Ende (separate *Prozess*-Instanz)."""
        # WICHTIG: SimulationApp pro Prozess genau einmal initialisieren
        self.simulation_app = SimulationApp({"headless": self.headless})
        try:
            self.world, self.task, self.robot, self.controller = self._build_world()
            art = self.robot.get_articulation_controller()
            self.logger = self.world.get_data_logger()
            self._log_cb_holder = {}
            self._ensure_logger_running(self.world, self.robot, self.logger, self._log_cb_holder)

            reset_needed = False
            last_saved_path = None

            while self.simulation_app.is_running():
                self.world.step(render=not self.headless)

                if self.world.is_stopped() and not reset_needed:
                    reset_needed = True

                if self.world.is_playing():
                    if reset_needed:
                        save_path = self._unique_log_path(self.logdir)
                        self.logger.pause()
                        self.logger.save(log_path=str(save_path))
                        n = self.logger.get_num_of_data_frames()
                        if n >= 1:
                            idx = min(2, n - 1)
                            try:
                                print(f"[DataLogger] frames={n} peek@{idx}:", self.logger.get_data_frame(data_frame_index=idx))
                            except Exception as e:
                                print("[DataLogger] Could not peek frame:", e)
                        self.logger.reset()
                        self._ensure_logger_running(self.world, self.robot, self.logger, self._log_cb_holder)
                        last_saved_path = save_path

                        self.world.reset()
                        self.controller.reset()
                        reset_needed = False

                    obs = self.world.get_observations()
                    act = self.controller.forward(observations=obs)
                    art.apply_action(act)
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            try:
                if self.logger and self.logger.is_started() and self.logger.get_num_of_data_frames() > 0:
                    final_path = self._unique_log_path(self.logdir, prefix="isaac_sim_data_final")
                    self.logger.pause()
                    self.logger.save(log_path=str(final_path))
                    print(f"[DataLogger] Final log saved to: {final_path}")
            except Exception as e:
                print("[DataLogger] Failed to save final log:", e)
            if self.simulation_app is not None:
                self.simulation_app.close()


# ------------------ CLI-Entry für Einzelstart ------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="Run without UI")
    p.add_argument("--logdir", type=Path, default=Path("./00_my_envs/Franka_Cube_Stacking_JW/logs"), help="Directory for JSON logs")
    p.add_argument("--cam_freq", type=int, default=20, help="Camera frequency (Hz)")
    p.add_argument("--cam_res", type=str, default="256x256", help="Camera resolution WxH, e.g. 640x480")
    p.add_argument("--seed", type=int, default=312)
    p.add_argument("--forward_axis", type=str, default="x", choices=["x", "y"])
    p.add_argument("--rand_cube_rotation", action="store_true")
    p.add_argument("--rotation_mode", type=str, default="yaw", choices=["yaw", "xyz"])
    p.add_argument("--yaw_min", type=float, default=-5.0)
    p.add_argument("--yaw_max", type=float, default=5.0)
    p.add_argument("--keep_cubes_rotated", action="store_true")
    p.add_argument("--material_seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    W, H = map(int, args.cam_res.lower().split("x"))
    runner = IsaacStackingRunner(
        headless=args.headless,
        logdir=args.logdir,
        cam_freq=args.cam_freq,
        cam_res=(W, H),
        seed=args.seed,
        forward_axis=args.forward_axis,
        rand_cube_rotation=args.rand_cube_rotation,
        rotation_mode=args.rotation_mode,
        yaw_range=(args.yaw_min, args.yaw_max),
        keep_cubes_rotated=args.keep_cubes_rotated,
        material_seed=args.material_seed,
    )
    runner.run()


if __name__ == "__main__":
    main()

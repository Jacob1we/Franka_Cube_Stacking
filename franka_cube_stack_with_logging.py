"""
Franka Cube Stacking mit Data Logging für dino_wm Training.
Basiert auf franka_cube_stack_reworked.py mit integriertem DataLogger.
"""

import isaacsim
from isaacsim import SimulationApp

from pathlib import Path
from datetime import datetime
import logging, os, sys
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

launch_config = {"headless": False}  # True für schnellere Datensammlung
simulation_app = SimulationApp(launch_config)

import omni
from Franka_Env_JW import Stacking_JW
from Franka_Env_JW import StackingController_JW

from omni.isaac.core import World
import isaacsim.core.utils.nucleus as nucleus_utils
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prims_utils
import isaacsim.core.utils.rotations as rotations_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.sensors.camera import Camera
from isaacsim.core.api.robots import Robot
from pxr import UsdGeom, UsdShade, Gf, Sdf, Usd, UsdLux
import carb
from isaacsim.core.cloner import GridCloner

# Data Logger Import
from data_logger import FrankaDataLogger, get_franka_state, get_franka_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_collection.log", encoding="utf-8")
    ],
    force=True,
)

log = logging.getLogger("FrankaCubeStacking")

#region: Constants
SEED = 111
WORLD_ROOT = "/World"

# Datensammlung Konfiguration
NUM_EPISODES = 100           # Anzahl zu sammelnder Episoden
DATASET_PATH = "./dataset"
DATASET_NAME = "franka_cube_stack_ds"
SAVE_PNG = True              # Speichere alle Bilder auch als PNG

# Kamera
SIDE_CAM_BASE_POS = np.array ([1.6, -2.0, 1.27])       # ([1.41, -1.67, 1.27]) # m  
SIDE_CAM_EULER = (66.0, 0.0, 32.05) #(58.0, 0.0, 32.05) # deg
CAM_FREQUENCY = 20
CAM_RESOLUTION = (256, 256)

# Szene
SCENE_WIDTH = 0.60
SCENE_LENGTH = 0.75
FRANKA_BASE_CLEARANCE = 0.3
PLANE_LIFT = 0.001

# Würfel
N_CUBES = 2
CUBE_SIDE = 0.05
MIN_DIST = 1.5 * CUBE_SIDE
MAX_TRIES = 200
YAW_RANGE = (-180.0, 180.0)

# Materialien
ALLOWED_AREA_MATS = [
    ("AllowedArea_Steel_Brushed",   (0.62, 0.62, 0.62, 1.00)),
    ("AllowedArea_Aluminum_Mill",   (0.77, 0.77, 0.78, 1.00)),
    ("AllowedArea_Wood_Oak",        (0.65, 0.53, 0.36, 1.00)),
    ("AllowedArea_Wood_BirchPly",   (0.85, 0.74, 0.54, 1.00)),
    ("AllowedArea_Plastic_HDPE_Black", (0.08, 0.08, 0.08, 1.00)),
    ("AllowedArea_Rubber_Mat",      (0.12, 0.12, 0.12, 1.00)),
    ("AllowedArea_Acrylic_Frosted", (1.00, 1.00, 1.00, 1.00)),
]
#endregion


class Franka_Cube_Stack():
    
    def __init__(self, robot_name: str = "Franka") -> None:
        self.robot_name = robot_name
        self.world = None
        self.stage = None
        self.task = None
        self.rng = SEED
        self.world_prim_path = WORLD_ROOT
        self.robot_prim_path = f"{self.world_prim_path}/{robot_name}"   
        self.materials = None 
        self.logdir = "./logs"

    def setup_world(self):
        world = World(stage_units_in_meters=1.0)
        self.world = world
        world.scene.add_default_ground_plane()
        
        stage = stage_utils.get_current_stage()
        self.stage = stage
        
        task_name = "stacking_task"
        self.task_root = f"{self.world_prim_path}/Task"
        
        self.task = Stacking_JW(
            name=task_name,
            cube_size=[CUBE_SIDE] * 3,
            offset=[0.0, 0.0, 0.0],
            parent_prim_path=self.task_root,
            cube_amount=N_CUBES,
        )
        world.add_task(self.task)  
        world.reset()
        log.info("World Setup Complete")
        return

    def setup_post_load(self):
        log.info("Setup Post Load")
        robot_name = self.task.get_params()["robot_name"]["value"]
        self.franka = self.world.scene.get_object(robot_name)

        base_pos, base_quat = self.franka.get_local_pose()
        self.base_pos = base_pos
        self.base_quat = base_quat

        controller = StackingController_JW(
            name="stacking_controller",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
            picking_order_cube_names=self.task.get_cube_names(),
            robot_observation_name=robot_name,
        )
        return controller
    
    def set_scene_light(self, light_seed, light_prim_path=None):
        half_w = SCENE_WIDTH / 2.0
        half_l = SCENE_LENGTH / 2.0

        rng = np.random.default_rng(light_seed)
        px = rng.uniform(-0.8 * half_w, 0.8 * half_w)
        py = rng.uniform(-0.8 * half_l, 0.8 * half_l)
        pz = rng.uniform(0.8, 3.0)  
        
        if light_prim_path is not None:
            self.world.scene.remove_object(light_prim_path)

        light_xform_path = f"{self.task_root}/light_xform"
        light_prim_path = f"{light_xform_path}/light"
        
        UsdGeom.Xform.Define(self.stage, light_xform_path)
        light_xform_api = UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(light_xform_path))
        light_xform_api.SetTranslate(Gf.Vec3d(*np.array([px, py, pz])))     

        light = UsdLux.SphereLight.Define(self.stage, light_prim_path)
        light.GetIntensityAttr().Set(float(rng.uniform(5500.0, 7000.0)))
        light.GetRadiusAttr().Set(float(rng.uniform(0.4, 0.6)))
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    
    def add_scene_cam(self):
        cam_xform_path = f"{self.task_root}/camera_xform"
        cam_prim_path = f"{cam_xform_path}/camera"
        
        UsdGeom.Xform.Define(self.stage, cam_xform_path)
        UsdGeom.Camera.Define(self.stage, cam_prim_path)

        cam_xform_api = UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(cam_xform_path))
        cam_xform_api.SetTranslate(Gf.Vec3d(*SIDE_CAM_BASE_POS))     
        cam_xform_api.SetRotate(Gf.Vec3f(*SIDE_CAM_EULER))

        cam = Camera(
            prim_path=cam_prim_path,
            frequency=CAM_FREQUENCY,
            resolution=CAM_RESOLUTION,
        )
        return cam
    
    def get_camera_matrices(self, camera):
        """Extrahiert Kamera Intrinsic und Extrinsic Matrizen."""
        # Intrinsic Matrix (aus Kamera-Parametern)
        # Für eine typische Kamera mit FOV
        fov = 60.0  # Default FOV in Grad
        h, w = CAM_RESOLUTION
        fx = w / (2.0 * np.tan(np.radians(fov / 2.0)))
        fy = fx  # Quadratische Pixel
        cx, cy = w / 2.0, h / 2.0
        
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Extrinsic Matrix (World to Camera)
        cam_pos = np.array(SIDE_CAM_BASE_POS)
        cam_euler = np.array(SIDE_CAM_EULER)
        
        # Rotation Matrix aus Euler-Winkeln
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('xyz', cam_euler, degrees=True)
        rot_matrix = rot.as_matrix()
        
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = rot_matrix
        extrinsic[:3, 3] = cam_pos
        
        return intrinsic, extrinsic

    def get_materials(self, plane_root):
        self.materials = []
        for name, rgba in ALLOWED_AREA_MATS:
            mat_path = f"{plane_root}/{name}"
            mat = UsdShade.Material.Get(self.stage, mat_path)
            if not mat:
                mat = UsdShade.Material.Define(self.stage, mat_path)
                shader = UsdShade.Shader.Define(self.stage, f"{plane_root}/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgba[:3])
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(rgba[3])

                surf_out = mat.CreateSurfaceOutput()
                shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                surf_out.ConnectToSource(shader_out)
            self.materials.append(mat)
    
    def add_or_update_plane(self, seed):
        plane_root = f"{self.task_root}/Plane"
        mesh = UsdGeom.Mesh.Get(self.stage, plane_root)
        if not mesh:
            mesh = UsdGeom.Mesh.Define(self.stage, plane_root)
        plane_prim = mesh.GetPrim()

        def make_point(x, y):
            p = self.base_pos + np.array([x, y, PLANE_LIFT])
            return Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
        
        p0_w = make_point(0.0, -SCENE_WIDTH/2)
        p1_w = make_point(0.0, SCENE_WIDTH/2)
        p2_w = make_point(SCENE_LENGTH, SCENE_WIDTH/2)
        p3_w = make_point(SCENE_LENGTH, -SCENE_WIDTH/2)

        points_local = [p0_w, p1_w, p2_w, p3_w]

        mesh.CreatePointsAttr(points_local)
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        up = Gf.Vec3f(0.0, 0.0, 1.0)
        mesh.CreateNormalsAttr([up, up, up, up])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateDoubleSidedAttr(True)

        if self.materials is None:
            self.get_materials(plane_root)

        rng = np.random.default_rng(seed)
        random_material_index = int(rng.integers(0, len(self.materials)))
        material = self.materials[random_material_index]
        UsdShade.MaterialBindingAPI(plane_prim).Bind(material)
        return 

    def domain_randomization(self, seed):
        log.info("Start Randomization")
        self.rng = np.random.default_rng(seed)

        def sample_cube_positions():
            points = []
            w = 0.6 * CUBE_SIDE

            def point_valid(point_xy):
                if np.linalg.norm(point_xy - self.base_pos[:2]) < FRANKA_BASE_CLEARANCE:
                    return False
                for p in points:
                    if np.linalg.norm(point_xy - p[:2]) < MIN_DIST:
                        return False
                return True
            
            for _ in range(N_CUBES + 1):
                fallback_point = np.array([0.5, 0.0, 1.1 * CUBE_SIDE])
                found_valid = False
                for _ in range(MAX_TRIES):
                    u = self.rng.uniform(0, SCENE_LENGTH)
                    v = self.rng.uniform(-SCENE_WIDTH/2, SCENE_WIDTH/2)
                    point = self.base_pos + np.array([u, v, w])
                    point_xy = point[:2].astype(float)

                    if point_valid(point_xy):
                        points.append(point)
                        found_valid = True
                        break
                    
                if not found_valid:
                    points.append(fallback_point)
            return points

        def sample_cube_orientation(seed):
            orientations = []
            for n in range(N_CUBES):
                cube_rng = np.random.default_rng(seed + n)
                yaw_deg = float(cube_rng.uniform(*YAW_RANGE))
                orientation = rotations_utils.euler_angles_to_quat(
                    np.array([0.0, 0.0, yaw_deg]), degrees=True
                )
                orientations.append(orientation)
            return orientations
        
        def cube_randomization_in_existing_task():
            cube_names = self.task.get_cube_names()
            sample_points = sample_cube_positions()
            cube_sample_points = sample_points[:N_CUBES]
            stack_target_sample_point = sample_points[N_CUBES]
            sample_orientations = sample_cube_orientation(seed)

            for n, name in enumerate(cube_names):
                cube_name = name
                cube_pos = cube_sample_points[n]
                cube_ori = sample_orientations[n]
                cube_target = stack_target_sample_point
                self.task.set_params(cube_name, cube_pos, cube_ori, cube_target)
            
            return cube_sample_points, stack_target_sample_point

        cube_positions, target_position = cube_randomization_in_existing_task()
        self.add_or_update_plane(seed)
        self.set_scene_light(seed)

        return cube_positions, target_position


def main():
    seed = SEED
    env = Franka_Cube_Stack()
    env.setup_world()
    controller = env.setup_post_load()
    
    simulation_context = SimulationContext()
    articulation = env.franka.get_articulation_controller()
    camera = env.add_scene_cam()
    camera.initialize()
    
    # ================================================================
    # DATA LOGGER SETUP
    # ================================================================
    current_folder = os.path.basename(os.getcwd())

    if current_folder == "ISAACSIM" or current_folder == "isaacsim":
        local_save_path = "./00_my_envs/Franka_Cube_Stacking_JW/dataset"
    else: 
        local_save_path = DATASET_PATH
        
    logger = FrankaDataLogger(
        save_path=local_save_path,
        object_name=DATASET_NAME,
        image_size=CAM_RESOLUTION,
        max_timesteps=None,  # Unbegrenzt - läuft bis Controller fertig ist
        save_png=SAVE_PNG,   # Speichere alle Bilder auch als PNG
    )
    
    # Kamera-Kalibrierung speichern
    try:
        intrinsic, extrinsic = env.get_camera_matrices(camera)
        logger.set_camera_calibration(intrinsic, extrinsic)
    except ImportError:
        log.warning("scipy nicht verfügbar, überspringe Kamera-Kalibrierung")
    
    episode_count = 0
    
    # ================================================================
    # HAUPTSCHLEIFE - Datensammlung
    # ================================================================
    while simulation_app.is_running() and episode_count < NUM_EPISODES:
        # Domain Randomization
        cube_positions, target_position = env.domain_randomization(seed)
        
        # Neue Episode starten
        logger.start_episode(episode_count)
        logger.set_episode_params({
            "seed": seed,
            "cube_positions": [p.tolist() for p in cube_positions],
            "target_position": target_position.tolist(),
        })
        
        step_count = 0
        
        # Episode durchführen - läuft bis Controller fertig ist
        while not controller.is_done():
            simulation_app.update()
            env.world.step()
            
            # Beobachtungen sammeln
            obs = env.task.get_observations()
            action = controller.forward(observations=obs)
            
            # ============================================================
            # DATEN LOGGEN
            # ============================================================
            try:
                # RGB Bild
                rgba = camera.get_rgba()
                if rgba is not None:
                    rgb = rgba[:, :, :3]  # Entferne Alpha-Kanal
                    
                    # State extrahieren
                    state = get_franka_state(env.franka, env.task)
                    
                    # Action extrahieren
                    action_vec = get_franka_action(action)
                    
                    # Optional: Depth
                    # depth = camera.get_depth()
                    
                    # Loggen
                    logger.log_step(
                        rgb_image=rgb,
                        state=state,
                        action=action_vec,
                        additional_data={
                            "timestep": step_count,
                            "cube_0_pos": obs.get(env.task.get_cube_names()[0], {}).get("position", [0,0,0]),
                        }
                    )
            except Exception as e:
                log.warning(f"Fehler beim Loggen: {e}")
            
            # Aktion ausführen
            articulation.apply_action(action)
            step_count += 1
        
        # Episode beenden
        logger.end_episode()
        
        # Screenshot speichern (optional)
        try:
            rgba = camera.get_rgba()
            if rgba is not None:
                image_path = f"{env.logdir}/00_Screenshots/Episode_{episode_count:03d}.png"
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                plt.imsave(image_path, rgba)
        except Exception as e:
            log.warning(f"Screenshot fehlgeschlagen: {e}")
        
        log.info(f"Episode {episode_count} abgeschlossen: {step_count} Schritte")
        
        # Reset
        env.world.reset()
        controller.reset()
        env.world.step()
        
        seed += 1
        episode_count += 1
        
        # Zwischenspeichern alle 10 Episoden
        if episode_count % 10 == 0:
            logger.save_dataset()
            log.info(f"Zwischenstand gespeichert: {episode_count} Episoden")
    
    # ================================================================
    # FINALES SPEICHERN
    # ================================================================
    logger.save_dataset()
    log.info(f"Datensammlung abgeschlossen: {episode_count} Episoden")
    
    simulation_app.close()


if __name__ == "__main__":
    main()


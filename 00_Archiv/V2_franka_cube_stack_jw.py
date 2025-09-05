# --- Isaac Sim boot ---
from isaacsim import SimulationApp

# CLI-Parameter (optional)
import argparse
from pathlib import Path
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="Run without UI")
    p.add_argument("--logdir", type=Path, default=Path("./00_my_envs/Franka_Cube_Stacking_JW/logs"), help="Directory for JSON logs")
    p.add_argument("--cam_freq", type=int, default=20, help="Camera frequency (Hz)")
    p.add_argument("--cam_res", type=str, default="256x256", help="Camera resolution WxH, e.g. 640x480")
    return p.parse_args()

ARGS = parse_args()
W, H = map(int, ARGS.cam_res.lower().split("x"))
simulation_app = SimulationApp({"headless": ARGS.headless})

# --- import Isaac modules that require the app context ---
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller import StackingController
from isaacsim.robot.manipulators.examples.franka.tasks import Stacking
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import isaacsim.core.utils.prims as prim_utils
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdShade, Gf, Sdf


SCENE_WIDTH  = 0.50  # 50 cm
SCENE_LENGTH = 0.80  # 80 cm
FRAKNA_BASE_CLEARANCE = 0.20 # 20 cm
FORWARD_AXIS = 'x'  # bei Bedarf 'y' wählen, falls Franka in deiner Szene nach +Y „schaut“
SEED = 000
CUBE_SIDE = 0.0515  # nur für Mindestabstand
MIN_DIST = 1.2 * CUBE_SIDE  # Optional: verhindert Überschneidungen der Startwürfel
FRANKA_BASE_CLEARANCE = 0.25


## Helper Functions
def quat_to_rot(q):
    """[w,x,y,z] -> 3x3 Rotationsmatrix."""
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),        2*(yz - wx)],
        [    2*(xz - wy),      2*(yz + wx),     1 - 2*(xx + yy)],
    ], dtype=float)

def get_base_axes_from_local_quat(base_quat_local, forward_axis='x'):
    """Vorwärts/Seit-Achse im lokalen Task-Frame aus der Basis-Orientation.
       Standard: +X = vor, +Y = seitlich. Falls dein Franka nach +Y „schaut“:
       forward_axis='y' setzen.
    """
    R = quat_to_rot(base_quat_local)
    if forward_axis == 'x':
        x_fwd = R[:, 0] / np.linalg.norm(R[:, 0])
        y_side = R[:, 1] / np.linalg.norm(R[:, 1])
    else:  # 'y'
        x_fwd = R[:, 1] / np.linalg.norm(R[:, 1])
        y_side = R[:, 0] / np.linalg.norm(R[:, 0])
    return x_fwd, y_side

def _forward_side_axes_world(base_quat_world, forward_axis='x'):
    """
    Vorwärts- (+X oder +Y) und Seitenachse der Basis **in Weltkoordinaten**.
    forward_axis='x' bedeutet: +X der Basis zeigt „nach vorne“ (typisch Franka).
    """
    R = quat_to_rot(base_quat_world)
    if forward_axis == 'x':
        fwd = R[:, 0]
        side = R[:, 1]
    else:
        fwd = R[:, 1]
        side = R[:, 0]
    # normieren (numerische Stabilität)
    fwd  = fwd  / (np.linalg.norm(fwd)  + 1e-12)
    side = side / (np.linalg.norm(side) + 1e-12)
    return fwd, side


## Visualisieren der Erlaubten Zone, Domain-Randomization: Plattenfarbe
def _get_or_create_red_material(stage, mat_path="/World/Looks/AllowedAreaMat", rgba=(1.0,0.0,0.0,1.0)):
    """
    Erstellt (oder holt) ein einfaches UsdPreviewSurface-Material in Rot.
    Verbindet das Material-Surface-Output korrekt mit dem Shader-Surface-Output.
    """
    mat = UsdShade.Material.Get(stage, mat_path)
    if not mat:
        mat = UsdShade.Material.Define(stage, mat_path)

        shader = UsdShade.Shader.Define(stage, mat_path + "/PBR")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Inputs setzen
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgba[:3])
        shader.CreateInput("roughness",   Sdf.ValueTypeNames.Float).Set(0.6)
        shader.CreateInput("metallic",    Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("opacity",     Sdf.ValueTypeNames.Float).Set(rgba[3])

        # >>> Korrekte Verbindung: Output-auf-Output
        surf_out   = mat.CreateSurfaceOutput()  # Material-Output
        shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)  # Shader-Output
        surf_out.ConnectToSource(shader_out)

    return mat

def add_or_update_allowed_area_plane(
    stage,
    robot_obj,
    width=0.50,            # Breite des erlaubten Bereichs (quer, ±width/2 um die Basis)
    length=0.80,           # Länge des Bereichs (von der Basis nach vorne)
    forward_axis='x',      # 'x' (typisch) oder 'y' je nach Basisausrichtung in deiner Szene
    lift=0.001,            # 1 mm anheben, um Z-Fighting mit dem Boden zu vermeiden
    prim_path="/World/AllowedAreaPlane",
    bind_red_material=True # True: rotes Material automatisch binden
):
    """
    Legt eine **einfache Mesh-Plane** (2 Dreiecke) in den **erlaubten Bereich**:
    - Hintere Kante verläuft durch die Franka-Basis.
    - Breite symmetrisch (±width/2 entlang Seitenachse).
    - Länge 0..length entlang Vorwärtsachse.
    - Orientierung richtet sich automatisch nach der Basis-Quaternion.
    - Position/Vertices werden **in Weltkoordinaten** gesetzt.

    Kann mehrfach aufgerufen werden: existiert das Prim bereits, werden nur die
    Geometrie-Attribute (Punkte/Indizes) aktualisiert.
    """
    # 1) Basispose **in Weltkoordinaten** holen
    base_pos_w, base_quat_w = robot_obj.get_world_pose()   # base_pos_w: (3,), base_quat_w: [w,x,y,z]

    # 2) Vorwärts- und Seitenachsen in Welt berechnen (aus Quaternion)
    fwd, side = _forward_side_axes_world(base_quat_w, forward_axis=forward_axis)

    # 3) Vier Ecken des Rechtecks (in Weltkoordinaten) konstruieren
    #    Hintere Kante (v=0) geht durch die Basis, v wächst nach vorne bis 'length'.
    #    Reihenfolge: p0 -> p1 -> p2 -> p3 ergibt zwei Dreiecke (0,1,2) und (0,2,3)
    half_w = width * 0.5
    z_lift = float(base_pos_w[2] + lift)  # kleine Anhebung

    # Hilfsfunktion, um 3D-Punkt aus Basis + Offsets zu bauen
    def make_point(u_lateral, v_forward):
        p = base_pos_w + u_lateral * side + v_forward * fwd
        return Gf.Vec3f(float(p[0]), float(p[1]), z_lift)

    p0 = make_point(-half_w, 0.0)     # hinten links
    p1 = make_point( half_w, 0.0)     # hinten rechts
    p2 = make_point( half_w, length)  # vorne  rechts
    p3 = make_point(-half_w, length)  # vorne  links

    # 4) Mesh-Prim anlegen oder holen
    mesh = UsdGeom.Mesh.Get(stage, prim_path)
    if not mesh:
        mesh = UsdGeom.Mesh.Define(stage, prim_path)

    # 5) Punkte/Topologie setzen (2 Dreiecke)
    mesh.CreatePointsAttr([p0, p1, p2, p3])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2,   0, 2, 3])

    # 6) Normale (nach oben) und Double-Sided (sichtbar von oben/unten)
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0),
                            Gf.Vec3f(0.0, 0.0, 1.0),
                            Gf.Vec3f(0.0, 0.0, 1.0),
                            Gf.Vec3f(0.0, 0.0, 1.0)])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)

    # 7) Optional: rotes Material binden (nur einmal anlegen, dann wiederverwenden)
    if bind_red_material:
        mat = _get_or_create_red_material(stage)
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

    return mesh

## Domain Randomization
def sample_points_in_front_rectangle_local(
    n,                          # int: Anzahl zu samplender Punkte (z.B. Anzahl Startwürfel)
    base_pos_local,             # np.ndarray (3,): Basisposition der Franka im *lokalen Task-Frame*
    base_quat_local,            # np.ndarray (4,): Basisorientation [w,x,y,z] im *lokalen Task-Frame*
    width=0.50,                 # float: Rechteck-Breite (quer zur Vorwärtsrichtung), in m
    length=0.80,                # float: Rechteck-Länge (entlang der Vorwärtsrichtung), in m
    z_levels=None,              # Optional[List[float]]: Z-Höhe je Punkt; None → Z der Basis nehmen
    min_dist=None,              # Optional[float]: Mindest-XY-Abstand zwischen *Punkten* (Kollisionsvermeidung)
    max_tries=200,              # int: Anzahl Zufallsversuche pro Punkt, bevor Fallback greift
    base_clearance=0.20,        # float: **NEU** Mindest-XY-Abstand zur *Basis* (Sicherheitszone), in m
    seed=None,                  # Optional[int]: RNG-Seed für Reproduzierbarkeit
    forward_axis='x'            # 'x' oder 'y': Vorwärtsachse der Basis in deinem Setup
):
    """
    Gibt ein (n,3)-Array zurück mit Punkten im *lokalen Task-Frame*, die:
      - im Rechteck vor der Basis liegen (Breite 'width', Länge 'length')
      - mind. 'base_clearance' Meter von der Basis (in XY) entfernt sind
      - optional mind. 'min_dist' XY-Abstand zueinander haben
    """

    # --- Plausibilitätschecks (fangen triviale Fehler früh ab) ---
    assert n >= 1, "n muss >= 1 sein"
    assert width  > 0.0 and length > 0.0, "width und length müssen > 0 sein"
    assert base_clearance >= 0.0, "base_clearance darf nicht negativ sein"
    if z_levels is not None:
        assert len(z_levels) == n, "z_levels muss Länge n haben (oder None sein)"

    # RNG initialisieren (deterministisch bei gesetztem seed)
    rng = np.random.default_rng(seed)

    # Vorwärts-/Seitenachsen der Basis (im lokalen Task-Frame)
    x_fwd, y_side = get_base_axes_from_local_quat(base_quat_local, forward_axis=forward_axis)

    # Hier sammeln wir die Punkte
    pts = []

    def ok(p_xy):
        """
        Prüft, ob ein Kandidat ok ist:
          1) Abstand zur Basis >= base_clearance (XY)
          2) (optional) Abstand zu allen bereits platzierten Punkten >= min_dist (XY)
        p_xy ist ein Vektor der Form [x, y] (Z wird separat gehandhabt).
        """
        # 1) Basis-Sicherheitszone
        if np.linalg.norm(p_xy - base_pos_local[:2]) < base_clearance:
            return False
        # 2) Paarweiser Mindestabstand (nur wenn gefordert)
        if min_dist is not None:
            for q in pts:
                if np.linalg.norm(p_xy - q[:2]) < min_dist:
                    return False
        return True

    # Für jeden gewünschten Punkt samplen
    for i in range(n):
        # Z-Höhe bestimmen: explizit vorgegeben oder Z der Basis
        z = z_levels[i] if z_levels is not None else float(base_pos_local[2])

        # Versuche bis zu 'max_tries' eine gültige Position zu finden
        for _ in range(max_tries):
            # Seitlicher Versatz u ~ U(-width/2, +width/2)
            u = rng.uniform(-width * 0.5, width * 0.5)
            # Vorwärtsversatz v ~ U(0, length)
            v = rng.uniform(0.0, length)

            # Kandidat im lokalen Frame: Basis + u*side + v*fwd
            p = base_pos_local + u * y_side + v * x_fwd

            # XY-Teil für Prüfungen, Z später anfügen
            p_xy = p[:2].astype(float)

            if ok(p_xy):
                pts.append(np.array([p_xy[0], p_xy[1], z], dtype=float))
                break
        else:
            # --------- Fallback (nur wenn alle Versuche scheitern) ----------
            # Wir akzeptieren den Punkt, aber "klemmen" ihn ggf. auf den Rand der Basis-Sicherheitszone.
            u = rng.uniform(-width * 0.5, width * 0.5)
            v = rng.uniform(0.0, length)
            p = base_pos_local + u * y_side + v * x_fwd
            p_xy = p[:2].astype(float)

            # Abstand zur Basis prüfen & ggf. auf die Kreisgrenze (base_clearance) schieben
            vec_xy = p_xy - base_pos_local[:2]
            norm   = float(np.linalg.norm(vec_xy))
            if norm < base_clearance:
                # Richtung ermitteln (Sonderfall: exakt an der Basis → nimm fwd-Richtung)
                if norm < 1e-9:
                    vec_xy = x_fwd[:2]
                    norm   = float(np.linalg.norm(vec_xy)) + 1e-12
                p_xy = base_pos_local[:2] + (vec_xy / norm) * base_clearance

            pts.append(np.array([p_xy[0], p_xy[1], z], dtype=float))
            # Hinweis: Im Fallback ignorieren wir min_dist zwischen *Punkten*,
            #          um immer n Punkte liefern zu können (Deadlock-Prävention).

    # In ein (n,3)-Array stapeln und zurückgeben
    return np.vstack(pts)

def randomize_stacking_in_rectangle_existing_task(
    task,                                   # Stacking-Task (bestehende IsaacSim-Klasse)
    robot_obj,                              # Franka-Objekt (Articulation) aus der Szene, z.B. my_franka
    width=0.50,                             # Breite des Sampling-Rechtecks (m)
    length=0.80,                            # Länge des Sampling-Rechtecks (m)
    keep_cubes_z=True,                      # True: Z-Höhen der vorhandenen Würfel beibehalten
    min_dist=None,                          # Mindest-XY-Abstand zwischen Startwürfeln (z.B. 1.2 * Kantenlänge)
    seed=None,                              # RNG-Seed (int) für Reproduzierbarkeit
    forward_axis='x',                       # 'x' (typisch Franka) oder 'y' (falls deine Szene so definiert ist)
    base_clearance=FRANKA_BASE_CLEARANCE    # Mindestabstand zur Franka-Basis (XY) in m
):
    """
    Randomisiert:
      - Startpositionen aller Würfel (nur XY, Z optional beibehalten)
      - XY-Position der Turmbasis (Z=0 im Task-Frame; der Task stapelt dann in +Z)
    unter folgenden Nebenbedingungen:
      - Alle Punkte liegen im definierten Rechteck vor der Basis.
      - Alle Punkte sind mind. base_clearance Meter von der Basis (XY) entfernt.
      - Startwürfel haben untereinander optional mind. min_dist XY-Abstand.
    """

    # --- 1) Basispose IM LOKALEN TASK-FRAME ---
    # Warum lokal? Weil set_local_pose und set_params im selben Frame arbeiten → kein Frame-Mismatch.
    base_pos_local, base_quat_local = robot_obj.get_local_pose()

    # --- 2) Welche Würfel existieren im Task? ---
    cube_names = task.get_cube_names()
    n_cubes = len(cube_names)

    # --- 3) Z-Höhen für die Startwürfel bestimmen ---
    #    a) keep_cubes_z=True → wir lesen die aktuelle Z je Würfel aus der Szene (robust bzgl. Bodenhöhe)
    #    b) keep_cubes_z=False → später wird Z = Z der Basis verwendet
    z_levels = None
    if keep_cubes_z:
        z_levels = []
        for name in cube_names:
            pos_l, _ = task.scene.get_object(name).get_local_pose()
            z_levels.append(float(pos_l[2]))

    # --- 4) Startpositionen samplen (mit Basis-Sicherheitszone & min_dist) ---
    starts_local = sample_points_in_front_rectangle_local(
        n=n_cubes,
        base_pos_local=base_pos_local,
        base_quat_local=base_quat_local,
        width=width,
        length=length,
        z_levels=z_levels,
        min_dist=min_dist,
        base_clearance=base_clearance,  # <- NEU: Sicherheitszone aktiv
        seed=seed,
        forward_axis=forward_axis,
    )

    # --- 5) Gesampelte Startpositionen in der Szene anwenden ---
    # Wir setzen *direkt* am Würfel-Prim (positional args!), Orientierung bleibt unverändert.
    for i, name in enumerate(cube_names):
        cube = task.scene.get_object(name)
        _, current_quat = cube.get_local_pose()          # Quaternion [w,x,y,z]
        cube.set_local_pose(starts_local[i], current_quat)

    # --- 6) Turmbasis-XY im Rechteck samplen (Z=0 im Task-Frame) ---
    # Separater Seed, damit der Zielpunkt unabhängig von den Startpunkten ist (wenn gewünscht).
    target_local = sample_points_in_front_rectangle_local(
        n=1,
        base_pos_local=base_pos_local,
        base_quat_local=base_quat_local,
        width=width,
        length=length,
        z_levels=[0.0],                   # Turm "auf dem Boden" des Task-Frames
        min_dist=None,                    # nur ein Zielpunkt → keine Paarprüfung
        base_clearance=base_clearance,    # ebenfalls Sicherheitszone zur Basis
        seed=None if seed is None else seed + 1,
        forward_axis=forward_axis,
    )[0]

    # --- 7) Turmbasis in Task-Parameter schreiben ---
    # Achtung: Deine Task.set_params schreibt hier nur ein Feld, kein Prim-Call → API-sicher.
    task.set_params(
        stack_target_position=np.array([target_local[0], target_local[1], 0.0], dtype=float)
    )

## Data Logging
def make_frame_logging_func(robot_obj):
    """Return a per-frame logging function bound to the actual robot object."""
    def _log(tasks, scene):
        # safer: don't re-lookup the object by name each frame
        jp = robot_obj.get_joint_positions()
        aa = robot_obj.get_applied_action()
        return {
            "joint_positions": jp.tolist(),
            "applied_joint_positions": (aa.joint_positions.tolist() if aa and aa.joint_positions is not None else []),
        }
    return _log

def unique_log_path(base_dir: Path, prefix="isaac_sim_data"):
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{prefix}_{ts}.json"

def ensure_logger_running(world: World, robot, logger, log_func_ref_container: dict):
    """(Re)register logging func if needed and start the logger."""
    # If we previously registered a function, re-add it after a logger reset.
    if "func" not in log_func_ref_container:
        log_func_ref_container["func"] = make_frame_logging_func(robot)
    
    # The logger may drop callbacks on reset in some versions—so:
    logger.add_data_frame_logging_func(log_func_ref_container["func"])
    if not logger.is_started():
        logger.start()

def build_world(cam_freq: int, cam_res: tuple[int, int]):
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # world.scene.add_ground_plane()
    
    # stage = omni.usd.get_context().get_stage()

    # plane = UsdGeom.Mesh.Define(stage, "/World/Plane")
    # UsdGeom.XformCommonAPI(plane).SetTranslate(Gf.Vec3f(SCENE_LENGTH/2, 0.0, 0.001))
    # UsdGeom.XformCommonAPI(plane).SetScale(Gf.Vec3f(SCENE_LENGTH, SCENE_WIDTH, 1.0))

    task = Stacking()
    world.add_task(task)
    world.reset()  # instantiate prims / initialize timeline

    # Robot + Controller
    robot_name = task.get_params()["robot_name"]["value"]
    robot = world.scene.get_object(robot_name)

    stage = omni.usd.get_context().get_stage()
    # Lege/aktualisiere die erlaubte Fläche (50 cm breit, 80 cm lang)
    # forward_axis='x' falls Franka in deiner Szene nach +X schaut; sonst 'y'
    add_or_update_allowed_area_plane(
        stage=stage,
        robot_obj=robot,
        width=0.50,
        length=0.80,
        forward_axis='x',
        prim_path="/World/AllowedAreaPlane"
    )


    randomize_stacking_in_rectangle_existing_task(
        task=task,
        robot_obj=robot,
        width=SCENE_WIDTH,
        length=SCENE_LENGTH,
        keep_cubes_z=True,
        min_dist=MIN_DIST,
        base_clearance=FRAKNA_BASE_CLEARANCE,   # 20 cm Sicherheitszone
        seed=SEED,
        forward_axis=FORWARD_AXIS,
    )

    # Camera
    Camera(
        prim_path="/World/camera",
        position=np.array([0.48, -3.6, 1.8]),
        frequency=cam_freq,
        resolution=cam_res,
        orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 22.5, 90.0]), degrees=True),
    )

    # Light
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

## Main Simulation Loop
def main():
    world, task, robot, ctrl = build_world(ARGS.cam_freq, (W, H))
    art = robot.get_articulation_controller()
    logger = world.get_data_logger()
    log_cb_holder = {}

    ensure_logger_running(world, robot, logger, log_cb_holder)

    reset_needed = False
    last_saved_path = None

    try:
        while simulation_app.is_running():
            world.step(render=not ARGS.headless)

            # Track stop/play transitions via the world's timeline state
            if world.is_stopped() and not reset_needed:
                reset_needed = True

            if world.is_playing():
                if reset_needed:
                    # Save previous episode (if we actually recorded anything)
                    save_path = unique_log_path(ARGS.logdir)
                    logger.pause()  # safe write
                    logger.save(log_path=str(save_path))

                    # Optional: peek at a frame if available
                    n = logger.get_num_of_data_frames()
                    if n >= 1:
                        # Use the last frame to avoid IndexError
                        idx = min(2, n - 1)
                        try:
                            print(f"[DataLogger] frames={n} peek@{idx}:", logger.get_data_frame(data_frame_index=idx))
                        except Exception as e:
                            print("[DataLogger] Could not peek frame:", e)

                    # Reset logger for next episode and re-register callback
                    logger.reset()
                    ensure_logger_running(world, robot, logger, log_cb_holder)
                    last_saved_path = save_path

                    # Reset sim episode
                    world.reset()
                    ctrl.reset()
                    reset_needed = False

                # Controller step
                obs = world.get_observations()
                act = ctrl.forward(observations=obs)
                art.apply_action(act)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    
    finally:
        # On exit: save whatever we have (without overwriting)
        try:
            if logger.is_started() and logger.get_num_of_data_frames() > 0:
                final_path = unique_log_path(ARGS.logdir, prefix="isaac_sim_data_final")
                logger.pause()
                logger.save(log_path=str(final_path))
                print(f"[DataLogger] Final log saved to: {final_path}")
            elif last_saved_path:
                print(f"[DataLogger] Last episode was saved at: {last_saved_path}")
        except Exception as e:
            print("[DataLogger] Failed to save final log:", e)
        simulation_app.close()

if __name__ == "__main__":
    main()

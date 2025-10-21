# --- Isaac Sim boot ---
from isaacsim import SimulationApp

# CLI-Parameter (optional)
import argparse
from pathlib import Path
from datetime import datetime
import logging, os, sys
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="Run without UI")
    p.add_argument("--logdir", type=Path, default=Path("./logs"), help="Directory for JSON logs")
    p.add_argument("--cam_freq", type=int, default=20, help="Camera frequency (Hz)")
    p.add_argument("--cam_res", type=str, default="256x256", help="Camera resolution WxH, e.g. 640x480")
    p.add_argument("--scenes", type=int, default=4, help="Number of Scenes")
    return p.parse_args()

ARGS = parse_args()
W, H = map(int, ARGS.cam_res.lower().split("x"))
simulation_app = SimulationApp({"headless": ARGS.headless})


os.environ.setdefault("PYTHONUNBUFFERED", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),                 # live in Konsole
        logging.FileHandler("live.log", encoding="utf-8")  # persistente Datei
    ],
    force=True,  # überschreibt evtl. Vorkonfiguration
)

log = logging.getLogger("FrankaCubeStacking")

# --- import Isaac modules that require the app context ---
from isaacsim.core.api import World
# from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller_jw import StackingController_JW
# from isaacsim.robot.manipulators.examples.franka.tasks import Stacking_JW

# from Franka_Env_JW import StackingController_JW
from Franka_Env_JW import StackingController as StackingController_JW
from Franka_Env_JW import Stacking_JW

from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import isaacsim.core.utils.prims as prim_utils
from omni.isaac.core.articulations import Articulation
import numpy as np
import omni.usd
# import pxr
from pxr import UsdGeom, UsdShade, Gf, Sdf, Usd, UsdLux

#region: Constants
SEED = 111
Problemseeds =  [10, 11, 12, 14]

NUM_SCENES = ARGS.scenes

SIDE_CAM_BASE_POS = np.array([2.4, -3.2, 2.2]) # m  [0.48, -3.6, 1.8]
SIDE_CAM_EULER = (63.2, 0.0, 33.0) # np.array([60, 29, 15])  # deg  [0.0, 22.5, 90.0]
SCENE_SPACING = 5.0  # m
ROBOTS_PER_LANE = np.round(np.sqrt(NUM_SCENES)).astype(int) # Für Lichtplatzierung und Positionierung der Szenen

SCENE_WIDTH  = 0.60  # cm
SCENE_LENGTH = 0.75  # cm
FRANKA_BASE_CLEARANCE = 0.3 # cm
FORWARD_AXIS = 'x'  # bei Bedarf 'y' wählen, falls Franka in deiner Szene nach +Y „schaut“
PLANE_LIFT = 0.001

N_CUBES = 2
CUBE_SIDE = 0.05  # für Mindestabstand und Würfelgröße
MIN_DIST = 1.5 * CUBE_SIDE  # Optional: verhindert Überschneidungen der Startwürfel
MAX_TRIES = 200
RAND_CUBE_ROTATION = True # Master-Schalter
ROTATION_MODE = "yaw"               # "yaw" | "xyz"
YAW_RANGE: tuple = (-180.0, 180.0) #(-5.0, 5.0) #(-180.0, 180.0)   # für mode="yaw"
KEEP_CUBES_ROTATED: bool = False             # vorhandene Ori behalten?

# Material-Pool (Beispiele: Farben + Transparenzen)
ALLOWED_AREA_MATS = [
    # Metalle
    ("AllowedArea_Steel_Brushed",   (0.62, 0.62, 0.62, 1.00 )),# 0.35)),  # gebürsteter Stahl, leicht dunkelgrau
    ("AllowedArea_Aluminum_Mill",   (0.77, 0.77, 0.78, 1.00 )),# 0.35)),  # Aluminium, etwas heller & kühler

    # Hölzer
    ("AllowedArea_Wood_Oak",        (0.65, 0.53, 0.36, 1.00 )),# 0.55)),  # Eiche, warmes Mittelbraun
    ("AllowedArea_Wood_BirchPly",   (0.85, 0.74, 0.54, 1.00 )),# .55)),  # Birke/Sperrholz, hell & gelblich

    # Kunststoffe / Elastomere
    ("AllowedArea_Plastic_HDPE_Black",(0.08, 0.08, 0.08, 1.00 )),# 0.40)), # schwarzes HDPE (Schneid-/Montageplatte)
    ("AllowedArea_Rubber_Mat",      (0.12, 0.12, 0.12, 1.00 )),# 0.45)),  # Gummimatte, sehr matt & dunkel

    # „Glas“ / Acryl (vereinfacht über Opazität)
    ("AllowedArea_Acrylic_Frosted", (1.00, 1.00, 1.00, 1.00 )),# 0.20)),  # satiniertes Acryl/Glas, milchig
]

log.info("Konstanten erfolgreich geladen. Set default Seed to %03d.", SEED)
#endregion

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
    # R = base_quat_local.get
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

def quat_mul(q1, q2):
    """Quaternion-Produkt (w,x,y,z). Wirkt wie: erst q2, dann q1."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)

def get_allowed_area_center_world(robot_obj,
                                  width: float = SCENE_WIDTH,
                                  length: float = SCENE_LENGTH,
                                  forward_axis: str = FORWARD_AXIS,
                                  lift: float = PLANE_LIFT) -> np.ndarray:
    """
    Mittelpunkt des Rechtecks, das du in add_or_update_allowed_area_plane erzeugst:
    - hintere Kante geht durch die Basis
    - Vorwärtsrichtung = fwd (aus Basisquat)
    - Center liegt bei Basis + 0.5*length * fwd (seitlich genau mittig)
    - z = Basis.z + lift
    """
    base_pos_w, base_quat_w = robot_obj.get_world_pose()  # quat: [w,x,y,z]
    R = quat_to_rot(base_quat_w)
    if forward_axis == 'x':
        fwd = R[:, 0] / (np.linalg.norm(R[:, 0]) + 1e-12)
    else:
        fwd = R[:, 1] / (np.linalg.norm(R[:, 1]) + 1e-12)

    center = base_pos_w + 0.5 * length * fwd
    center = np.array([center[0], center[1], base_pos_w[2] + lift], dtype=float)
    return center

def convert_ops_from_transform(prim) -> tuple:

    # Get the Xformable from prim
    xform = UsdGeom.Xformable(prim)

    # Gets local transform matrix - used to convert the Xform Ops.
    pose = omni.usd.get_local_transform_matrix(prim)

    # Compute Scale
    x_scale = Gf.Vec3d(pose[0][0], pose[0][1], pose[0][2]).GetLength()
    y_scale = Gf.Vec3d(pose[1][0], pose[1][1], pose[1][2]).GetLength()
    z_scale = Gf.Vec3d(pose[2][0], pose[2][1], pose[2][2]).GetLength()

    # Clear Transforms from xform.
    xform.ClearXformOpOrder()

    # Add the Transform, orient, scale set
    xform_op_t = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_r = xform.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_s = xform.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")

    xform_op_t.Set(pose.ExtractTranslation())
    xform_op_r.Set(pose.ExtractRotationQuat().GetNormalized())
    xform_op_s.Set(Gf.Vec3d(x_scale, y_scale, z_scale))

    return xform_op_t, xform_op_r, xform_op_s

log.info("Helper functions erfolgreich definiert.")

## Visualisieren der Erlaubten Zone, Domain-Randomization: Plattenfarbe
def _get_or_create_color_material(stage, name, rgba=(1.0, 0.0, 0.0, 1.0), mat_root="/World/Looks"):
    """
    Legt ein einfaches UsdPreviewSurface-Material in gewünschter Farbe an (oder holt es).
    name: z.B. "AllowedArea_Red"
    """
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
        # Output korrekt verbinden
        surf_out   = mat.CreateSurfaceOutput()
        shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        surf_out.ConnectToSource(shader_out)
    return mat

def _ensure_material_pool(stage, named_rgba_list, mat_root="/World/Looks"):
    """
    named_rgba_list: Liste wie [("AllowedArea_Red",(1,0,0,0.8)), ("AllowedArea_Green",(0,1,0,0.35)), ...]
    Gibt die zugehörigen UsdShade.Material-Objekte zurück (in gleicher Reihenfolge).
    """
    materials = []
    for name, rgba in named_rgba_list:
        materials.append(_get_or_create_color_material(stage, name=name, rgba=rgba, mat_root=mat_root))
    return materials

def bind_random_material_to_prim(stage, prim_path, material_list, seed=None):
    """
    Wählt deterministisch (bei seed) oder zufällig (ohne seed) ein Material aus und bindet es an prim_path.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim {prim_path} nicht gefunden oder ungültig.")
    rng = np.random.default_rng(seed)
    mat = material_list[int(rng.integers(0, len(material_list)))]
    UsdShade.MaterialBindingAPI(prim).Bind(mat)
    return mat

def add_or_update_allowed_area_plane(
    stage,
    robot_obj,
    width: float = SCENE_WIDTH,
    length: float = SCENE_LENGTH,
    forward_axis: str = FORWARD_AXIS,
    prim_path: str = "/World/AllowedAreaPlane",
    lift: float = PLANE_LIFT,
    material_pool_named_rgba = ALLOWED_AREA_MATS,
    material_seed = None,
    bind_red_material: bool = False,
):
    """
    Legt oder aktualisiert eine Allowed-Area-Plane relativ zum Szenen-Xform, der
    aus dem prim_path automatisch erkannt wird (z. B. /World/Scenes/Scene_000/...).

    Wenn die Szene verschoben oder rotiert wird, folgt die Plane automatisch.
    """

    # 1) Anker bestimmen (Parent-Xform, also die Szene)
    anchor_path = str(prim_path.rsplit("/", 1)[0])
    anchor_prim = stage.GetPrimAtPath(anchor_path)
    if not anchor_prim:
        raise RuntimeError(f"Anchor prim not found: {anchor_path}")

    anchor_xform = UsdGeom.Xformable(anchor_prim)
    M_world_to_anchor = Gf.Matrix4d(anchor_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).GetInverse()

    # 2) Basispose des Roboters (in Weltkoordinaten)
    base_pos_w, base_quat_w = robot_obj.get_world_pose()

    # 3) Vorwärts- und Seitenachsen in Welt berechnen
    fwd_w, side_w = _forward_side_axes_world(base_quat_w, forward_axis=forward_axis)

    # 4) Eckpunkte in Welt berechnen
    half_w = width * 0.5
    z_lift = float(base_pos_w[2] + lift)

    def make_point_world(u, v):
        p = base_pos_w + u * side_w + v * fwd_w
        return Gf.Vec3d(float(p[0]), float(p[1]), z_lift)

    p0_w = make_point_world(-half_w, 0.0)
    p1_w = make_point_world( half_w, 0.0)
    p2_w = make_point_world( half_w, length)
    p3_w = make_point_world(-half_w, length)

    # 5) Punkte in Anchor-Lokalraum transformieren
    def to_anchor_local(p_w: Gf.Vec3d) -> Gf.Vec3f:
        p_a = M_world_to_anchor.Transform(p_w)
        return Gf.Vec3f(p_a[0], p_a[1], p_a[2])

    points_local = [to_anchor_local(p) for p in [p0_w, p1_w, p2_w, p3_w]]

    # 6) Mesh anlegen oder aktualisieren
    mesh = UsdGeom.Mesh.Get(stage, prim_path)
    if not mesh:
        mesh = UsdGeom.Mesh.Define(stage, prim_path)

    mesh.CreatePointsAttr(points_local)
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])

    up = Gf.Vec3f(0.0, 0.0, 1.0)
    mesh.CreateNormalsAttr([up, up, up, up])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(True)

    # 7) Material (optional)
    if bind_red_material:
        mats = _ensure_material_pool(stage, {"red": (1.0, 0.0, 0.0, 1.0)})
        bind_random_material_to_prim(stage, prim_path, mats, seed=material_seed)
    elif material_pool_named_rgba:
        mats = _ensure_material_pool(stage, material_pool_named_rgba)
        bind_random_material_to_prim(stage, prim_path, mats, seed=material_seed)

    return mesh

log.info("Visualisierungs-Funktionen erfolgreich definiert.")

## Domain Randomization
def sample_points_in_front_rectangle_local(
    n: int,
    base_pos_local: np.ndarray,
    base_quat_local: np.ndarray,
    width: float,
    length: float,
    min_dist: float = None,
    max_tries: int = MAX_TRIES,
    base_clearance: float = FRANKA_BASE_CLEARANCE,
    seed = None,
    forward_axis: str = FORWARD_AXIS,
):
    """
    Gibt ein (n,3)-Array zurück mit Punkten im *lokalen Task-Frame*, die:
      - im Rechteck vor der Basis liegen (Breite 'width', Länge 'length')
      - mind. 'base_clearance' Meter von der Basis (in XY) entfernt sind
      - optional mind. 'min_dist' XY-Abstand zueinander haben
    """

    # --- Plausibilitätschecks ---
    assert n >= 1, "n muss >= 1 sein"
    assert width  > 0.0 and length > 0.0, "width und length müssen > 0 sein"
    assert base_clearance >= 0.0, "base_clearance darf nicht negativ sein"
    z = float(base_pos_local[2])  # Z-Höhe (im lokalen Frame) für alle Punkte

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
    stage,
    task,
    robot_obj,
    seed,
    scene_prim_path: str = "/World",
    width=SCENE_WIDTH,
    length=SCENE_LENGTH,
    keep_cubes_z=True,
    min_dist=MIN_DIST,
    base_clearance=FRANKA_BASE_CLEARANCE,
    forward_axis=FORWARD_AXIS,
    randomize_rotation=RAND_CUBE_ROTATION,
    rotation_mode=ROTATION_MODE,
    yaw_range_deg=YAW_RANGE,
    keep_cubes_rot=KEEP_CUBES_ROTATED,
    max_tries=MAX_TRIES,           
):
    """
    Randomisiert:
      - Startpositionen aller Würfel (nur XY, Z optional beibehalten)
      - optional: Yaw-Rotation um lokale Z-Achse pro Würfel
      - XY-Position der Turmbasis (Z=0 im Task-Frame; der Task stapelt dann in +Z)
    Nebenbedingungen:
      - Alle Punkte liegen im Rechteck vor der Basis.
      - Alle Punkte sind mind. base_clearance Meter von der Basis (XY) entfernt.
      - Startwürfel haben untereinander optional mind. min_dist XY-Abstand.
    """

    # 1) Basispose im lokalen Task-Frame
    base_pos_local, base_quat_local = robot_obj.get_local_pose()
    log.info(f"Basis im lokalen Task-Frame: Pos {np.round(base_pos_local,3)}, Quat {np.round(base_quat_local,3)}")
    
    scene_prim = stage.GetPrimAtPath(scene_prim_path)
    log.info(f"Szenen-Prim für Transform: {scene_prim}")
    scene_xform = UsdGeom.Xformable(scene_prim)
    log.info(f"Szenen-Xform für Transform: {scene_xform}")

    scene_ops = convert_ops_from_transform(scene_prim)
    log.info(f"Converted Szene Ops: {scene_ops}")

    # scene_xform_ops = scene_xform.GetOrderedXformOps()
    # log.info(f"Szenen-XformOps für Transform: {scene_xform_ops}")
    # time = Sdf.TimeCode.GetValue()
    # scene_transform = scene_xform_ops.GetOpTransfrom(time)
    # log.info(f"Szenen-Transform für Transform: {scene_transform}")

    # # base_transform = scene_xform.GetLocalTransformation()
    # # log.info(f"Basis im Szenen-Frame: Transform {base_transform} in Pfad {scene_prim_path}")
    # # base_pos_xform = base_transform[0]
    # # base_quat_xform = base_transform[1]
    # # log.info(f"current_pos_xform: {base_pos_xform}")
    # # log.info(f"current_quat_xform: {base_quat_xform}")

    # # (A) Lokale Matrix + Flag (kein Quaternion!)
    # local_mat = scene_xform.GetLocalTransformation()
    # # Dekomposition (lokal)
    # base_local_4D_transfrom = Gf.Transform(local_mat).GetMatrix()
    # log.info(f"UNUSED Local Transform: {base_local_4D_transfrom}")

    # # (B) Welt-Transform berechnen
    # world_mat = scene_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # base_global_4D_transfrom = Gf.Transform(world_mat).GetMatrix()
    # log.info(f"UNUSED World Transform: {base_global_4D_transfrom}")

    # 2) Würfel-Liste
    cube_names = task.get_cube_names()
    n_cubes = len(cube_names)

    # 3) Positions-Sampling (mit Sicherheitszone & min_dist)
    starts_local = sample_points_in_front_rectangle_local(
        n=n_cubes+1,  # +1 für Turmbasis
        base_pos_local=base_pos_local,
        base_quat_local=base_quat_local,
        width=width,
        length=length,
        min_dist=min_dist,
        max_tries=max_tries,
        base_clearance=base_clearance,  
        seed=seed,
        forward_axis=forward_axis,
    )
    
    # 4) Startposen + optionale Rotation anwenden (UNABHÄNGIG je Würfel)
    X_positions = []
    Y_positions = []
    cube_orientations  = []
    for i, name in enumerate(cube_names):
        cube = task.scene.get_object(name)

        # Get the Xformable from prim
        xform = UsdGeom.Xformable(cube.prim)
        log.info(f"Cube {i} '{name}' Xformable: {xform}")
        # Gets local transform matrix - used to convert the Xform Ops.
        pose = omni.usd.get_local_transform_matrix(cube.prim)
        # current_quat_xform = pose.ExtractRotationQuat()

        cube_transform = xform.GetLocalTransformation()
        log.info(f"Cube {i} '{name}' Transform: {cube_transform}")

        current_pos_xform = cube_transform[0]
        current_quat_xform = cube_transform[1]
        log.info(f"Cube {i} '{name}' current_pos_xform: {current_pos_xform}")
        log.info(f"Cube {i} '{name}' current_quat_xform: {current_quat_xform}")

        current_pos, current_quat = cube.get_local_pose()
        log.info(f"Cube {i} '{name}' current_quat: {current_quat}")

        X_positions.append(starts_local[i][0])
        Y_positions.append(starts_local[i][1])
        new_pos = np.array(starts_local[i], dtype=float)
        if keep_cubes_z:
            new_pos[2] = float(current_pos[2])

        # new_quat = current_quat
        new_quat = current_quat

        if randomize_rotation and not keep_cubes_rot:
            local_rng = np.random.default_rng(None if seed is None else seed + 100 + i)
            if rotation_mode == "yaw":
                yaw_deg = float(local_rng.uniform(*yaw_range_deg))
                q_delta = rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, yaw_deg]), degrees=True)
            else:
                log.error(f"Unbekannter rotation_mode: {rotation_mode}")
                raise ValueError(f"Unbekannter rotation_mode: {rotation_mode}")
            new_quat = quat_mul(current_quat, q_delta)

        cube_orientations.append(np.asarray(new_quat, dtype=float).tolist())
        cube.set_local_pose(new_pos, new_quat)
        log.info(f"Würfel {i} '{name}': Startpos {np.round(new_pos,3)}, Ori {np.round(new_quat,3)}")
    
    # 5) Ziel in Task-Params schreiben
    stack_target_xy = np.asarray(starts_local[n_cubes], dtype=float)
    task.set_params(
        stack_target_position=[float(stack_target_xy[0]), float(stack_target_xy[1]), 0.0],
        cube_position = new_pos,
        cube_orientation=new_quat
    )

    #6) Log-Ausgabe und Min-Abstandsprüfung
    cube_min_dist = np.linalg.norm(starts_local[0] - starts_local[1])
    log.info(f"Min. Distanz zwischen den Würfeln {cube_min_dist:.3f} m")

    for j in range(n_cubes):
        dist = np.linalg.norm(starts_local[j][:2] - starts_local[n_cubes][:2])
        log.info(f"Distanz Würfel {j} zum Turm-Ziel: {dist:.3f} m")
        if dist < min_dist:
            log.warning(f"Würfel {j} zu nah am Turm-Ziel: {dist:.3f} m < {min_dist:.3f} m")

def add_scene_light(i: int,
                    light_seed: int,
                    width: float = SCENE_WIDTH,
                    length: float = SCENE_LENGTH,
                    scene_root: str = "/World",
                    stage = None,
                    ):

    half_w = width / 2.0
    half_l = length / 2.0

    rng = np.random.default_rng(light_seed)

    # zufällige Position innerhalb der Szene
    px = rng.uniform(-0.8 * half_w, 0.8 * half_w)
    py = rng.uniform(-0.8 * half_l, 0.8 * half_l)
    pz = rng.uniform(0.8, 3.0)  

    # Licht und Licht-xForm Pfade definieren
    light_xform_path = f"{scene_root}/light_{i}_xform"
    light_prim_path = f"{light_xform_path}/light_{i}"
    
    # Licht-xForm anlegen
    UsdGeom.Xform.Define(stage, light_xform_path)

    # xForm-API von des Licht-xForms definieren und das das Licht xForm dann Verschieben
    light_xform_api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(light_xform_path))
    light_xform_api.SetTranslate(Gf.Vec3d(*np.array([px, py, pz])))     

    # Licht im Licht-xForm plazieren und Parameter setzen
    light = UsdLux.SphereLight.Define(stage, light_prim_path)
    light.GetIntensityAttr().Set(float(rng.uniform(5500.0, 7000.0)))
    light.GetRadiusAttr().Set(float(rng.uniform(0.4, 0.6)))
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

def add_scene_cam(i, scene_root, stage , cam_freq, cam_res):
    
    # Kamerapfad (eindeutiger Pfad je Szene)
    cam_xform_path = f"{scene_root}/camera_{i}_xform"
    cam_prim_path = f"{cam_xform_path}/camera_{i}"
    
    UsdGeom.Xform.Define(stage, cam_xform_path)
    UsdGeom.Camera.Define(stage, cam_prim_path)

    # Kamera relativ zur Szene platzieren
    cam_xform_api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(cam_xform_path))
    cam_xform_api.SetTranslate(Gf.Vec3d(*SIDE_CAM_BASE_POS))     
    cam_xform_api.SetRotate(Gf.Vec3f(*SIDE_CAM_EULER))

    cam = Camera(
        prim_path=cam_prim_path,
        # position=SIDE_CAM_BASE_POS + scene_offset,
        frequency=cam_freq,
        resolution=cam_res,
    )
    return cam

log.info("Domain-Randomization-Funktionen erfolgreich definiert.")  

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

log.info("Data Logging-Funktionen erfolgreich definiert.")

## Build World
def build_worlds(cam_freq: int, cam_res: tuple[int, int], num_scenes : int = NUM_SCENES, seed: int = SEED):
    
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()

    tasks = []
    robots = []
    ctrls = []
    cameras = []
    n_cubes = N_CUBES

    x_offset = 0
    y_offset = 0
    
    # --- alle Szenen anlegen ---
    for i in range(num_scenes):
        
        # Definiere den Szenen-Offset im Raster
        scene_offset = np.array([x_offset, y_offset, 0.0]) * SCENE_SPACING
        log.info(f"--- Baue Szene {i} bei Offset {scene_offset} ---")

        if (i+1)%ROBOTS_PER_LANE == 0 and i > 0:
                x_offset += 1
                y_offset = 0 #y_offset - ROBOTS_PER_LANE # also y_offset = 0  
        else: 
            y_offset += 1

        # Definiere den Szenen-Root-Pfad und verschiebe die Szene nch Offset
        scene_root = f"/World/Scenes/Scene_{i:03d}"
        xform_scene_root = UsdGeom.Xform.Define(stage, scene_root)
        UsdGeom.XformCommonAPI(xform_scene_root).SetTranslate(Gf.Vec3d(*scene_offset))

        # --- Task Vorbereitung ---
        task_name = f"stacking_task_{i}"
        # Ordne Task dem verschobenen Szenen-Root zu 
        task_root = f"{scene_root}/Task"
        # xform_task_root = UsdGeom.Xform.Define(stage, task_root)

        cube_size = [CUBE_SIDE] * 3

        task = Stacking_JW(
            name=task_name,
            cube_size=cube_size,
            offset=[0.0, 0.0, 0.0],  # kein weiterer Offset – Szene-Root verschiebt schon
            parent_prim_path=task_root,
            cube_amount=n_cubes,
        )
        world.add_task(task)
        tasks.append(task)

    world.reset() # instantiate prims & timeline
    log.info(f"--- Alle {num_scenes} Szenen angelegt. ---")
    
    x_offset = 0
    y_offset = 0

    # --- Setzen und Randomisieren des Szenen- und Task-Environments --- 
    for i, task in enumerate(tasks):
        
        scene_root = f"/World/Scenes/Scene_{i:03d}"
        task_root = f"{scene_root}/Task"
        

        # # --- Extrahieren der Roboter aus den Tasks ---

        # robot_name = task.get_params()["robot_name"]["value"]
        

        # robot_root = f"{task_root}/{robot_name}"
        # robot = task.scene.get_object(f"{task_root}/{robot_name}")

        # # robot_prim = stage.GetPrimAtPath(robot_root)
        # # robot_prim_path = robot_prim.GetPath().pathString
        # # robot = world.scene.get_object(robot_prim_path)  

        # log.info(f"Szene {task_name}: Robot name: {robot_name}")
        # log.info(f"Path Robot: {robot_root}")
        # # log.info(f"Robot Prim Path: {robot_prim_path}")
        # log.info(f"Robot object: {robot}")
        # log.info(f"[Szene {task_name}: Cube names: {task.get_cube_names()}")
        # Roboter-Pfad direkt aus dem Task lesen
        
        robot_name = task.get_params()["robot_name"]["value"]
        robot = world.scene.get_object(robot_name)
        robots.append(robot)

        # robot_path = f"{task_root}/{robot_name}"
        # # Roboter-Prim vom Stage holen
        # robot_prim = stage.GetPrimAtPath(robot_path)
        # # Als Articulation-Objekt registrieren (nur falls nicht schon vorhanden)
        # robot = world.scene.get_object(robot_path)
        # if robot is None:
        #     robot = Articulation(prim_path=robot_path, name=f"robot_{i}")
        #     world.scene.add(robot)

        # log.info(f"[Scene {i}] Robot prim: {robot_path}")
        log.info(f"[Scene {i}] Task: {task_name}")
        log.info(f"[Scene {i}] Robot object: {robot}")
        log.info(f"[Scene {i}] Cube names: {task.get_cube_names()}")
                
        # Controller
        ctrl = StackingController_JW(
            name=f"stacking_controller_{i}",
            gripper=robot.gripper,
            robot_articulation=robot,
            picking_order_cube_names=task.get_cube_names(),
            robot_observation_name=robot_name,
        )
        ctrls.append(ctrl)

        # Seed pro Szene
        sample_seed = seed + i*100

        # Allowed Area Plane (eindeutiger Pfad je Szene)
        add_or_update_allowed_area_plane(
            stage=stage,
            robot_obj=robot,
            prim_path=f"{scene_root}/AllowedAreaPlane_{i}",
            material_seed=None
        )
        
        # add_scene_cam(i,scene_root, stage=stage, cam_freq=cam_freq, cam_res=cam_res)
        cam = add_scene_cam(i,scene_root, stage=stage, cam_freq=cam_freq, cam_res=cam_res)
        cameras.append(cam)

        # --- Zufälliges Licht pro Szene ---
        add_scene_light(i, light_seed = sample_seed, scene_root=scene_root, stage=stage,)

        # --- Zufällige Würfelpositionen pro Szene ---
        randomize_stacking_in_rectangle_existing_task(
                            stage = stage,
                            task=task,
                            robot_obj=robot,
                            seed=sample_seed,
                            scene_prim_path = scene_root
                        )
        
    return world, tasks, robots, ctrls, cameras


## Main Simulation Loop
def main():
    # Seeds pro Szene
    current_seed = SEED
    scene_seeds = [current_seed + i*100 for i in range(NUM_SCENES)]
    stage = omni.usd.get_context().get_stage()

    world, tasks, robots, ctrls, cams = build_worlds(ARGS.cam_freq, (W, H), NUM_SCENES,current_seed)

    for cam in cams:
        cam.initialize()

    arts = [r.get_articulation_controller() for r in robots]

    logger = world.get_data_logger()
    log_cb_holder = {}
    ensure_logger_running(world, robots[0], logger, log_cb_holder)  # Callback an einen Robot binden reicht

    reset_needed = False
    last_saved_path = None

    log.info("Starting main simulation loop. Press Ctrl+C to exit.")

    try:
        while simulation_app.is_running():
            world.step(render=not ARGS.headless)

            # Track stop/play transitions via the world's timeline state
            if world.is_stopped() and not reset_needed:
                reset_needed = True

            if world.is_playing():

                #### RESET EPISODE #### RESET EPISODE #### RESET EPISODE #### RESET EPISODE ####
                if reset_needed:
                    # Save previous episode (if we actually recorded anything)
                    save_path = unique_log_path(ARGS.logdir)
                    logger.pause()  # safe write
                    logger.save(log_path=str(save_path))
                    
                    # Reset logger for next episode and re-register callback
                    logger.reset()
                    ensure_logger_running(world, robots[0], logger, log_cb_holder)
                    last_saved_path = save_path

                    for i, cam in enumerate(cams):
                        try:
                            rgba = cam.get_rgba()  # (H, W, 4)
                            image_path = f"{ARGS.logdir}/00_Screenshots/Scene{i:02d}_Episode_{scene_seeds[i]:03d}.png"
                            plt.imsave(image_path, rgba)
                        except Exception as e:
                            log.warning(f"[Scene {i}] Screenshot fehlgeschlagen: {e}")

                    # Reset sim episode
                    world.reset()
                    for ctrl in ctrls:
                        ctrl.reset()
                    arts = [r.get_articulation_controller() for r in robots]
                    world.step(render=False)

                    # Seed erhöhen (lokal) und randomisieren
                    current_seed += 1
                    log.info("---------------------------------------------------")
                    log.info("Resetting episode. Next Seed: %03d", current_seed)

                    # Seeds erhöhen & Szenen neu randomisieren
                    for i, (task, robot) in enumerate(zip(tasks, robots)):
                        scene_seeds[i] += 1
                        scene_prim_path = f"/World/Scenes/Scene_{i:03d}"
                        log.info("---------------------------------------------------")
                        log.info(f"[Scene {i}] Resetting episode. Next Seed: {scene_seeds[i]:03d}")

                        randomize_stacking_in_rectangle_existing_task(
                            stage = stage,
                            task = task,
                            robot_obj = robot,
                            seed = current_seed,
                            scene_prim_path = scene_prim_path
                        )
                        
                        log.info(f"[Scene {i}] New stacking target position: {task.get_params()['stack_target_position']['value']}")

                        add_or_update_allowed_area_plane(
                            stage=stage,
                            robot_obj=robot,
                            prim_path=f"{scene_prim_path}/AllowedAreaPlane_{i}",
                            material_seed=None
                        )

                    reset_needed = False

                # Controller-Schritt für alle Szenen
                obs = world.get_observations()              # PROBLEM: lokale Obervationen werden als global gesetzt
                
                for i, (art, ctrl) in enumerate(zip(arts, ctrls)):
                    act = ctrl.forward(observations=obs)
                    art.apply_action(act)
                    log.debug(f"Actions {act} applied.")

                if all(ctrl.is_done() for ctrl in ctrls):
                    reset_needed = True

                # for robot in robots:
                #     robot.get_joints_state()
                #     log.debug(f"Robot '{robot.name}' Joints State: {robot.joints_state}")
                    
                for i, task in enumerate(tasks):
                    if i == 0 or i%NUM_SCENES == 0:
                        log.info(f"------------- Observations -------------")
                        
                    task_obs = task.get_observations()          # lokale Observations des Tasks
                    # task_obs.joint

                    robot_name = task.get_params()["robot_name"]["value"]
                    joint_positions = task_obs[robot_name]["joint_positions"]

                    # joint_velocities = task_obs[robot_name]["joint_velocities"]
                    ee_local = task_obs[robot_name]["end_effector_position"]
                    # log.info(f"[Scene {i}] Robot '{robot_name}' Joints: {np.round(joint_positions,3)}, Velocities: {np.round(joint_velocities,3)}, EE Local Pos: {np.round(ee_local,3)}")

                    cube_names = task.get_cube_names()
                    cube0_name = cube_names[0]
                    cube0_pos = task_obs[cube0_name]["position"]
                    cube0_ori = task_obs[cube0_name]["orientation"]
                    cube0_target = task_obs[cube0_name]["target_position"]
                    log.info(f"Cube0 '{cube0_name}' Pos: {np.round(cube0_pos,3)}, Ori: {np.round(cube0_ori,3)}, Target: {np.round(cube0_target,3)}")

                # ## DEBUG START
                # log.info(f"OBS keys: {list(obs.keys())[:20]}")
                # log.info(f"Obs_0 example: {list(obs.values())[0]}")

                # for i, (ctrl, task) in enumerate(zip(ctrls, tasks)):
                #     robot_key = getattr(ctrl, "_robot_observation_name", None)
                #     if robot_key is None:
                #         robot_key = task.get_params()["robot_name"]["value"]

                #     cube_keys = getattr(ctrl, "_picking_order_cube_names", None)
                #     if cube_keys is None:
                #         cube_keys = task.get_cube_names()

                #     log.info(f"[Scene {i}] expects robot key: {robot_key} and cubes: {cube_keys}")
                # ## DEBUG END


            

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
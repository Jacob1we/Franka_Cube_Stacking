"""
Franka Cube Stacking mit Data Logging für dino_wm Training.
PARALLEL VERSION - Unterstützt mehrere Umgebungen gleichzeitig.

Basiert auf franka_cube_stack_reworked.py mit integriertem DataLogger.
"""

from datetime import datetime
import logging, os, sys
import numpy as np
import yaml
import argparse
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

def load_config(config_path: str = None) -> dict:
    """Lädt Konfiguration aus YAML-Datei."""
    if config_path is None:
        # Standard: config.yaml im gleichen Verzeichnis
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

# Argument Parser für Config-Pfad
parser = argparse.ArgumentParser(description="Franka Cube Stacking Data Collection")
parser.add_argument("--config", type=str, default=None, help="Pfad zur YAML-Konfigurationsdatei")
args, _ = parser.parse_known_args()

# Config laden
CFG = load_config(args.config)

# SimulationApp mit headless aus Config starten
import isaacsim
from isaacsim import SimulationApp

launch_config = {"headless": CFG["simulation"]["headless"]}
simulation_app = SimulationApp(launch_config)

import omni
from Franka_Env_JW import Stacking_JW
from Franka_Env_JW import StackingController_JW

from omni.isaac.core import World
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.rotations as rotations_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom, UsdShade, Gf, Sdf, UsdLux
from scipy.spatial.transform import Rotation as R


# Data Logger Import
from min_data_logger import MinDataLogger as FrankaDataLogger

# CSV Episode Logger Import (simpel, keine Abhängigkeiten)
from csv_episode_logger import CSVEpisodeLogger

from Franka_Env_JW.rmpflow_controller_jw import (
    RMPFlowController_JW,
    PRESET_LOCK_WRIST_ROTATION,
    PRESET_LOCK_UPPER_ARM,
    PRESET_MINIMAL_MOTION,
    PRESET_LOCK_FOREARM,
    PRESET_ESSENTIAL_ONLY,
    PRESET_LOCK_TWO,
    PRESET_LOCK_THREE,
)

# Erweiterte Logging-Konfiguration mit mehr Details
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_collection.log", encoding="utf-8")
    ],
    force=True,
)

log = logging.getLogger("FrankaCubeStacking")

# Detailliertes Exception-Logging aktivieren
def log_exception(exc_type, exc_value, exc_traceback):
    """Detailliertes Exception-Logging mit vollem Stack Trace."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    log.error(
        "=" * 80,
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    log.error(f"Exception Type: {exc_type.__name__}")
    log.error(f"Exception Value: {exc_value}")
    log.error("Full Traceback:")
    import traceback
    for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
        log.error(line.rstrip())

# Setze custom exception handler
sys.excepthook = log_exception

#region: Konstanten aus Config
SEED = CFG["simulation"]["seed"]
WORLD_ROOT = CFG["simulation"]["world_root"]

# Parallelisierung
NUM_ENVS = CFG["parallel"]["num_envs"]
ENV_SPACING = CFG["parallel"]["env_spacing"]

# Datensammlung
NUM_EPISODES = CFG["dataset"]["num_episodes"]
DATASET_PATH = CFG["dataset"]["path"]
DATASET_NAME = CFG["dataset"]["name"]
SAVE_PNG = CFG["dataset"]["save_png"]

# Kamera
SIDE_CAM_BASE_POS = np.array(CFG["camera"]["position"])
SIDE_CAM_EULER = tuple(CFG["camera"]["euler"])
CAM_FREQUENCY = CFG["camera"]["frequency"]
CAM_RESOLUTION = tuple(CFG["camera"]["resolution"])

# Szene
SCENE_WIDTH = CFG["scene"]["width"]
SCENE_LENGTH = CFG["scene"]["length"]
PLANE_LIFT = CFG["scene"]["plane_lift"]

# Würfel
N_CUBES = CFG["cubes"]["count"]
CUBE_SIDE = CFG["cubes"]["side"]
MIN_DIST = CFG["cubes"]["min_dist_factor"] * CUBE_SIDE
MAX_TRIES = CFG["cubes"]["max_placement_tries"]
YAW_RANGE = tuple(CFG["cubes"]["yaw_range"])

TRAJECTORY_RESOLUTION = CFG["controller"]["trajectory_resolution"]
AIR_SPEED_MULTIPLIER = CFG["controller"]["air_speed_multiplier"]
HEIGHT_ADAPTIVE_SPEED = CFG["controller"]["height_adaptive_speed"]
CRITICAL_HEIGHT_THRESHOLD = CFG["controller"]["critical_height_threshold"]
CRITICAL_SPEED_FACTOR = CFG["controller"]["critical_speed_factor"]
AIR_DT = CFG["controller"]["air_dt"]
CRIT_DT = CFG["controller"]["critical_dt"]
WAIT_DT = CFG["controller"]["wait_dt"]
GRIP_DT = CFG["controller"]["grip_dt"]
RELEASE_DT = CFG["controller"]["release_dt"]
GUARANTEE_FINAL_POSITION = CFG["controller"].get("guarantee_final_position", True)
GUARANTEE_PHASES = CFG["controller"].get("guarantee_phases", [1, 6])
BASE_DT = [AIR_DT, CRIT_DT, WAIT_DT, GRIP_DT, AIR_DT, AIR_DT, CRIT_DT, RELEASE_DT, AIR_DT, AIR_DT]
BASE_EVENT_DT = [AIR_DT, CRIT_DT, WAIT_DT, GRIP_DT, RELEASE_DT]

# [AIR, CRIT]
# Phase Overview (for understanding speed parameters):
  #   ---------------------------------------------------
  #   Phase 0: Move EE above cube at initial height     [AIR - can be fast]
  #   Phase 1: Lower EE down to cube                    [CRITICAL - must be precise]
  #   Phase 2: Wait for inertia to settle               [WAIT]
  #   Phase 3: Close gripper                            [GRIP]
  #   Phase 4: Lift EE up with cube                     [AIR - can be fast]
  #   Phase 5: Move EE toward target XY                 [AIR - can be fast]
  #   Phase 6: Lower EE to place cube                   [CRITICAL - must be precise]
  #   Phase 7: Open gripper                             [RELEASE]
  #   Phase 8: Lift EE up                               [AIR - can be fast]
  #   Phase 9: Return EE to starting position           [AIR - can be fast]

# Roboter Workspace
FRANKA_BASE_CLEARANCE = CFG["robot"]["base_clearance"]
FRANKA_MAX_REACH = CFG["robot"]["max_reach"]
FRANKA_MIN_REACH = CFG["robot"]["min_reach"]

# Materialien aus Config laden
ALLOWED_AREA_MATS = [
    (mat["name"], tuple(mat["rgba"])) for mat in CFG["materials"]
]

# Validierung
XY_TOLERANCE = CFG["validation"]["xy_tolerance"]
Z_MIN_HEIGHT = CFG["validation"]["z_min_height"]
Z_STACK_TOLERANCE = CFG["validation"]["z_stack_tolerance"]
#endregion

DATASET_NAME = f"{datetime.now().strftime('%Y_%m_%d_%H%M')}_fcs_dset"

class Franka_Cube_Stack():
    """
    Franka Cube Stacking Environment.
    
    Unterstützt sowohl Single-Env als auch Multi-Env (parallel) Modus.
    Für Parallelisierung: Erstelle mehrere Instanzen mit verschiedenen env_idx.
    """
    
    def __init__(
        self, 
        robot_name: str = "Franka",
        env_idx: int = 0,           # Index der Umgebung (0 für Single-Mode)
        offset: np.ndarray = None,  # Räumlicher Offset für parallele Envs
    ) -> None:
        self.robot_name = robot_name
        self.env_idx = env_idx
        self.offset = offset if offset is not None else np.array([0.0, 0.0, 0.0])
        
        self.world = None
        self.stage = None
        self.task = None
        self.rng = SEED + env_idx * 1000  # Unterschiedliche Seeds pro Env
        self.world_prim_path = WORLD_ROOT
        self.robot_prim_path = f"{self.world_prim_path}/{robot_name}"   
        self.materials = None 
        self.logdir = "./logs"
        
        # Für parallele Envs: Eindeutige Pfade
        if env_idx > 0 or NUM_ENVS > 1:
            self.task_root = f"{self.world_prim_path}/Env_{env_idx}/Task"
        else:
            self.task_root = f"{self.world_prim_path}/Task"

    def setup_world(self, world: World = None, stage = None):
        """
        Setup der Welt.
        
        Args:
            world: Existierende World (für parallelen Modus) oder None (erstellt neue)
            stage: Existierende Stage oder None
        """
        # Im parallelen Modus wird die World von außen übergeben
        if world is None:
            world = World(stage_units_in_meters=1.0)
            world.scene.add_default_ground_plane()
            self.world = world
        else:
            self.world = world
        
        if stage is None:
            stage = stage_utils.get_current_stage()
        self.stage = stage
        
        task_name = f"stacking_task_{self.env_idx}"
        
        self.task = Stacking_JW(
            name=task_name,
            cube_size=[CUBE_SIDE] * 3,
            offset=self.offset.tolist(),  # Offset für parallele Platzierung
            parent_prim_path=self.task_root,
            cube_amount=N_CUBES,
        )
        self.world.add_task(self.task)
        
        # Nur im Single-Mode hier resetten (parallel macht es nach allen Tasks)
        if NUM_ENVS == 1:
            world.reset()
            log.info("World Setup Complete")
        
        return self.world

    def setup_post_load(self, event_dt: list = None):
        log.info("Setup Post Load")
        robot_name = self.task.get_params()["robot_name"]["value"]
        self.franka = self.world.scene.get_object(robot_name)

        base_pos, base_quat = self.franka.get_local_pose()
        self.base_pos = base_pos
        self.base_quat = base_quat

        BASE_EVENT_DT = [AIR_DT, CRIT_DT, WAIT_DT, GRIP_DT, RELEASE_DT]

        if event_dt is None:
            event_dt = BASE_EVENT_DT
        else:
            event_dt = event_dt
        
        dt_list = [event_dt[0], event_dt[1], event_dt[2], event_dt[3], event_dt[0], event_dt[0], event_dt[1], event_dt[4], event_dt[0], event_dt[0]]

        controller = StackingController_JW(
            name="stacking_controller",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
            picking_order_cube_names=self.task.get_cube_names(),
            robot_observation_name=robot_name,
            preferred_joints=PRESET_LOCK_THREE,  # Soft constraint: prefer neutral pose   PRESET_MINIMAL_MOTION, PRESET_ESSENTIAL_ONLY
            trajectory_resolution=TRAJECTORY_RESOLUTION,               # Base resolution (affects ALL phases)
            air_speed_multiplier=AIR_SPEED_MULTIPLIER,                # Speed up AIR phases only (0,4,5,8,9)
            height_adaptive_speed=HEIGHT_ADAPTIVE_SPEED,              # DYNAMIC: Slow down near ground!
            critical_height_threshold=CRITICAL_HEIGHT_THRESHOLD,           # Below xx cm = critical zone
            critical_speed_factor=CRITICAL_SPEED_FACTOR,               # slower in critical zone
            guarantee_final_position=GUARANTEE_FINAL_POSITION,        # Snap to exact pick/place height
            guarantee_phases=GUARANTEE_PHASES,                        # Phases to guarantee (default: [1, 6])
            events_dt=dt_list,
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
        
        # Licht-Position mit Offset für parallele Umgebungen
        light_pos = np.array([px, py, pz]) + self.offset
        
        UsdGeom.Xform.Define(self.stage, light_xform_path)
        light_xform_api = UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(light_xform_path))
        light_xform_api.SetTranslate(Gf.Vec3d(*light_pos))     

        light = UsdLux.SphereLight.Define(self.stage, light_prim_path)
        light.GetIntensityAttr().Set(float(rng.uniform(5500.0, 7000.0)))
        light.GetRadiusAttr().Set(float(rng.uniform(0.4, 0.6)))
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    
    def add_scene_cam(self):
        cam_xform_path = f"{self.task_root}/camera_xform"
        cam_prim_path = f"{cam_xform_path}/camera"
        
        UsdGeom.Xform.Define(self.stage, cam_xform_path)
        UsdGeom.Camera.Define(self.stage, cam_prim_path)

        # Kamera-Position mit Offset für parallele Umgebungen
        cam_pos = SIDE_CAM_BASE_POS + self.offset
        
        cam_xform_api = UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(cam_xform_path))
        cam_xform_api.SetTranslate(Gf.Vec3d(*cam_pos))     
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
                dist_from_base = np.linalg.norm(point_xy - self.base_pos[:2])
                
                # Zu nah am Roboter-Sockel
                if dist_from_base < FRANKA_MIN_REACH:
                    return False
                
                # Außerhalb der Reichweite (Workspace-Limit)
                if dist_from_base > FRANKA_MAX_REACH:
                    return False
                
                # Zu nah an anderen Würfeln
                for p in points:
                    if np.linalg.norm(point_xy - p[:2]) < MIN_DIST:
                        return False
                return True
            
            for _ in range(N_CUBES + 1):
                # Fallback: Sicherer Punkt im Arbeitsbereich
                fallback_point = self.base_pos + np.array([0.4, 0.0, 1.1 * CUBE_SIDE])
                found_valid = False
                for _ in range(MAX_TRIES):
                    # Sample innerhalb der konfigurierten Szene (SCENE_LENGTH x SCENE_WIDTH)
                    u = self.rng.uniform(0, SCENE_LENGTH)
                    v = self.rng.uniform(-SCENE_WIDTH/2, SCENE_WIDTH/2)
                    point = self.base_pos + np.array([u, v, w])
                    point_xy = point[:2].astype(float)

                    # Validierung: Szene UND Workspace
                    if point_valid(point_xy):
                        points.append(point)
                        found_valid = True
                        break
                    
                if not found_valid:
                    log.warning(f"Kein gültiger Punkt gefunden, verwende Fallback")
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


def validate_positions_reachable(cube_positions: list, target_position: np.ndarray, base_pos: np.ndarray) -> tuple:
    """
    Validiert VOR der Episode, ob alle Positionen im Arbeitsbereich liegen.
    
    Args:
        cube_positions: Liste der Würfelpositionen
        target_position: Zielposition für das Stapeln
        base_pos: Position der Roboter-Basis
    
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    all_positions = list(cube_positions) + [target_position]
    
    for i, pos in enumerate(all_positions):
        pos = np.array(pos)
        dist = np.linalg.norm(pos[:2] - base_pos[:2])
        
        # Zu nah
        if dist < FRANKA_MIN_REACH:
            name = f"Würfel {i}" if i < len(cube_positions) else "Zielposition"
            return False, f"{name} zu nah am Roboter (d={dist:.3f}m < {FRANKA_MIN_REACH}m)"
        
        # Zu weit
        if dist > FRANKA_MAX_REACH:
            name = f"Würfel {i}" if i < len(cube_positions) else "Zielposition"
            return False, f"{name} außerhalb Reichweite (d={dist:.3f}m > {FRANKA_MAX_REACH}m)"
    
    return True, "Alle Positionen erreichbar"


def validate_stacking(task, target_position: np.ndarray) -> tuple:
    """
    Validiert, ob die Würfel korrekt gestapelt wurden.
    
    Prüft:
    1. Alle Würfel sind in X/Y nahe der Zielposition
    2. Alle Würfel sind über dem Boden (nicht durchgefallen)
    3. Würfel sind übereinander gestapelt (Z-Koordinaten aufsteigend)
    
    Args:
        task: Die Stacking Task mit den Würfeln
        target_position: Zielposition [x, y, z]
    
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    cube_names = task.get_cube_names()
    cube_positions = []
    
    # Sammle aktuelle Würfelpositionen
    for name in cube_names:
        try:
            cube = task.scene.get_object(name)
            pos, _ = cube.get_world_pose()
            cube_positions.append(pos)
        except Exception as e:
            return False, f"Konnte Würfel {name} nicht finden: {e}"
    
    # Prüfung 1: X/Y Toleranz zur Zielposition
    for i, pos in enumerate(cube_positions):
        dx = abs(pos[0] - target_position[0])
        dy = abs(pos[1] - target_position[1])
        
        if dx > XY_TOLERANCE or dy > XY_TOLERANCE:
            return False, f"Würfel {i} nicht an Zielposition (dx={dx:.3f}, dy={dy:.3f})"
    
    # Prüfung 2: Alle Würfel über dem Boden
    for i, pos in enumerate(cube_positions):
        if pos[2] < Z_MIN_HEIGHT:
            return False, f"Würfel {i} unter Boden (z={pos[2]:.3f})"
    
    # Prüfung 3: Würfel sind gestapelt (Z-Koordinaten)
    z_values = sorted([pos[2] for pos in cube_positions])
    for i in range(1, len(z_values)):
        expected_z_diff = CUBE_SIDE  # Würfel sollten ~CUBE_SIDE übereinander sein
        actual_z_diff = z_values[i] - z_values[i-1]
        
        # Toleranz: Würfel sollten mindestens halb übereinander sein
        if actual_z_diff < CUBE_SIDE * 0.5 - Z_STACK_TOLERANCE:
            return False, f"Würfel nicht korrekt gestapelt (z_diff={actual_z_diff:.3f})"
    
    return True, "Stacking erfolgreich"


def compute_grid_offsets(num_envs: int, spacing: float) -> list:
    """Berechnet Grid-Offsets für parallele Umgebungen."""
    grid_size = int(np.ceil(np.sqrt(num_envs)))
    offsets = []
    for i in range(num_envs):
        row = i // grid_size
        col = i % grid_size
        offset = np.array([col * spacing, row * spacing, 0.0])
        offsets.append(offset)
    return offsets


def get_rgb(camera, env_idx: int = 0):
    """
    Extrahiert RGB-Bild aus Kamera-Feed mit automatischer Format-Konvertierung.
    
    Args:
        camera: Camera-Objekt von Isaac Sim
        env_idx: Index der Umgebung (für Logging)
    
    Returns:
        np.ndarray: RGB-Bild (H, W, 3) als uint8, oder None bei Fehler
    """
    # Prüfe ob Kamera bereit ist
    if not hasattr(camera, 'get_rgba'):
        log.warning(f"Env {env_idx}: Kamera hat keine get_rgba() Methode")
        return None
    
    rgba = camera.get_rgba()
    
    # Prüfe Format von rgba und konvertiere falls nötig
    if rgba is None:
        log.debug(f"Env {env_idx}: rgba ist None (Kamera noch nicht bereit?)")
        return None
    
    # Konvertiere zu numpy array falls nötig
    if not isinstance(rgba, np.ndarray):
        rgba = np.array(rgba)
    
    # Prüfe ob Array leer ist (Kamera noch nicht bereit)
    if rgba.size == 0:
        log.debug(f"Env {env_idx}: rgba ist leer (Kamera noch nicht bereit?)")
        return None
    
    # Prüfe Shape und konvertiere falls nötig
    if rgba.ndim == 1:
        # 1D Array - möglicherweise flach, versuche zu reshapen
        expected_size = CAM_RESOLUTION[0] * CAM_RESOLUTION[1] * 4  # RGBA
        if rgba.size == expected_size:
            rgba = rgba.reshape((CAM_RESOLUTION[0], CAM_RESOLUTION[1], 4))
        else:
            log.debug(f"Env {env_idx}: rgba Shape {rgba.shape} != erwartet {expected_size}")
            return None
    elif rgba.ndim == 2:
        # 2D Array - möglicherweise Graustufen, konvertiere zu RGBA
        if rgba.size == 0:
            return None
        log.debug(f"Env {env_idx}: rgba ist 2D, konvertiere zu RGBA")
        rgba = np.stack([rgba, rgba, rgba, np.ones_like(rgba) * 255], axis=-1)
    elif rgba.ndim == 3:
        # 3D Array - sollte (H, W, C) sein
        if rgba.shape[2] == 1:
            # Graustufen, konvertiere zu RGBA
            rgba = np.repeat(rgba, 4, axis=2)
            rgba[:, :, 3] = 255  # Alpha = 255
        elif rgba.shape[2] == 3:
            # RGB, füge Alpha-Kanal hinzu
            alpha = np.ones((rgba.shape[0], rgba.shape[1], 1), dtype=rgba.dtype) * 255
            rgba = np.concatenate([rgba, alpha], axis=2)
        # rgba.shape[2] == 4 ist OK
    else:
        # Unerwartete Dimensionen
        log.debug(f"Env {env_idx}: Unerwartete rgba Dimensionen: {rgba.ndim}, Shape: {rgba.shape}")
        return None
    
    # Extrahiere RGB (erste 3 Kanäle)
    rgb = rgba[:, :, :3].copy()
    
    # Stelle sicher, dass es uint8 ist
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
    
    return rgb


def main():
    """
    Hauptfunktion - unterstützt Single und Parallel Mode.
    
    Single Mode (NUM_ENVS=1): Sequentielle Datensammlung
    Parallel Mode (NUM_ENVS>1): Parallele Datensammlung mit Grid-Layout
    """
    
    log.info("=" * 60)
    log.info(f"Franka Cube Stacking - {'PARALLEL' if NUM_ENVS > 1 else 'SINGLE'} Mode")
    log.info(f"  Config: {args.config or 'config.yaml (default)'}")
    log.info(f"  Headless: {CFG['simulation']['headless']}")
    log.info(f"  Anzahl Umgebungen: {NUM_ENVS}")
    log.info(f"  Episoden gesamt: {NUM_EPISODES}")
    if NUM_ENVS > 1:
        log.info(f"  Episoden pro Env: {NUM_EPISODES // NUM_ENVS}")
    log.info("=" * 60)
    
    # ================================================================
    # SETUP - Single oder Parallel
    # ================================================================
    offsets = compute_grid_offsets(NUM_ENVS, ENV_SPACING)
    
    # Listen für parallele Objekte
    envs = []
    controllers = []
    articulations = []
    cameras = []
    seeds = [SEED + i * 1000 for i in range(NUM_ENVS)]
    episode_counts = [0] * NUM_ENVS
    env_done = [False] * NUM_ENVS
    
    # Shared World für alle Envs
    shared_world = None
    shared_stage = None
    
    # Erstelle alle Umgebungen
    for i in range(NUM_ENVS):
        log.info(f"Setup Environment {i}...")
        
        env = Franka_Cube_Stack(
            robot_name="Franka",
            env_idx=i,
            offset=offsets[i],
        )
        
        # Erste Env erstellt die World, weitere nutzen sie mit
        if i == 0:
            shared_world = env.setup_world()
            shared_stage = env.stage
        else:
            env.setup_world(world=shared_world, stage=shared_stage)
        
        envs.append(env)
    
    # Nach allen Tasks: World reset
    shared_world.reset()
    log.info("All environments created, world reset complete")
    event_dt = None  # Kann angepasst werden für Sweeps

    # Controller und Kameras für alle Envs
    for i, env in enumerate(envs):

        # # airtime sweep
        # event_dt = [AIR_DT + i*0.1, CRIT_DT, WAIT_DT, GRIP_DT, RELEASE_DT]

        # # Grip/ Release Time sweep
        # event_dt = [AIR_DT, CRIT_DT, WAIT_DT, GRIP_DT + i*0.1, RELEASE_DT + i*0.1]

        # # Critical Time sweep
        # event_dt = [AIR_DT, CRIT_DT + i*0.005, WAIT_DT, GRIP_DT, RELEASE_DT]

        if event_dt == None:
            event_dt = BASE_EVENT_DT

        controller = env.setup_post_load(event_dt=event_dt)
        controllers.append(controller)
        
        articulation = env.franka.get_articulation_controller()
        articulations.append(articulation)
        
        camera = env.add_scene_cam()
        camera.initialize()
        cameras.append(camera)
    
    simulation_context = SimulationContext()
    
    # Warte ein paar Frames, damit Kameras initialisiert werden können
    log.info("Warte auf Kamera-Initialisierung...")
    for _ in range(10):
        simulation_app.update()
        shared_world.step(render=False)  # render=False für schnellere Initialisierung
    log.info("Kamera-Initialisierung abgeschlossen")
    
    # ================================================================
    # DATA LOGGER SETUP
    # ================================================================
    # Logger mit Config initialisieren
    # Action wird automatisch aus EE-Bewegung berechnet
    logger = FrankaDataLogger(
        config=CFG,
        dataset_path=Path(DATASET_PATH) / DATASET_NAME
    )
    
    # Überschreibe Dataset-Name mit Timestamp (wird in Config verwendet, aber wir überschreiben)
    logger.object_name = DATASET_NAME
    logger.dataset_path = Path(DATASET_PATH) / DATASET_NAME
    logger.dataset_path.mkdir(parents=True, exist_ok=True)
    (logger.dataset_path / "cameras").mkdir(exist_ok=True)
    
    # CSV Episode Logger initialisieren (simpel, keine Abhängigkeiten)
    csv_logger = CSVEpisodeLogger(
        output_dir=str(logger.dataset_path),
        filename="episode_tracking.csv"
    )
    log.info(f"CSV Episode Logger initialisiert: {csv_logger.filepath}")
    
    # Kamera-Kalibrierung setzen (für erste Kamera)
    if len(cameras) > 0:
        intrinsic, extrinsic = envs[0].get_camera_matrices(cameras[0])
        logger.set_camera_calibration(intrinsic, extrinsic)
        log.info("Kamera-Kalibrierung gesetzt")
    
    # Tracking für parallele Episodes (eine pro Env gleichzeitig aktiv)
    active_episode_ids = [None] * NUM_ENVS  # Globale Episode-ID pro Env
    episode_data = [{} for _ in range(NUM_ENVS)]  # Temporäre Daten pro Env
    
    # Tracking für Validierung
    failed_seeds = []  # Alle fehlgeschlagenen Seeds
    successful_counts = [0] * NUM_ENVS  # Erfolgreiche Episoden pro Env
    target_positions = [None] * NUM_ENVS  # Aktuelle Zielpositionen
    cube_positions_list = [None] * NUM_ENVS  # Würfelpositionen pro Env
    
    # Globale Episode-Zählung
    global_episode_id = 0  # Nächste freie Episode-ID
    total_successful = 0
    
    # Episoden pro Env berechnen
    episodes_per_env = NUM_EPISODES // NUM_ENVS
    
    # ================================================================
    # HAUPTSCHLEIFE - Datensammlung
    # ================================================================
    total_episodes = 0
    step_counts = [0] * NUM_ENVS
    
    # Initial: Domain Randomization und Episode starten für alle Envs
    for i in range(NUM_ENVS):
        cube_pos, target_pos = envs[i].domain_randomization(seeds[i])
        target_positions[i] = target_pos
        cube_positions_list[i] = cube_pos
        
        # Episode-Daten initialisieren (noch nicht im Logger starten)
        # Action wird im Logger automatisch aus EE-Bewegung berechnet!
        active_episode_ids[i] = global_episode_id
        episode_data[i] = {
            "observations": [],
            "depths": [],
            "ee_positions": [],
            "ee_quaternions": [],
            "cube_positions": [],
            "phase_timesteps": {p: 0 for p in range(10)},  # Timesteps pro Phase (0-9)
            "current_phase": 0,  # Aktuelle Phase für Tracking
            "params": {
                "seed": seeds[i],
                "env_idx": i,
                "cube_positions": [p.tolist() for p in cube_pos],
                "target_position": target_pos.tolist(),
            }
        }
        global_episode_id += 1
    
    # Warte ein paar Frames, damit Kameras initialisiert werden können
    log.info("Warte auf Kamera-Initialisierung...")
    for _ in range(10):
        simulation_app.update()
        shared_world.step(render=False)  # render=False für schnellere Initialisierung
    log.info("Kamera-Initialisierung abgeschlossen")
    
    while simulation_app.is_running() and total_successful < NUM_EPISODES:
        simulation_app.update()
        
        # Beobachtungen für alle Envs sammeln
        all_obs = shared_world.get_observations()
        
        # ============================================================
        # STEP für jede Umgebung
        # ============================================================
        for i in range(NUM_ENVS):
            if env_done[i]:
                continue
            
            env = envs[i]
            controller = controllers[i]
            camera = cameras[i]
            
            # Action berechnen
            action = controller.forward(observations=all_obs)
            
            # Daten in temporärem Buffer sammeln
            try:
                # Extrahiere RGB-Bild aus Kamera
                rgb = get_rgb(camera, env_idx=i)

                if rgb is None:
                    # Kamera nicht bereit - überspringe diesen Timestep
                    articulations[i].apply_action(action)
                    step_counts[i] += 1
                    continue
                
                if rgb is not None and rgb.size > 0:
                    
                    # Endeffektor-Pose extrahieren
                    ee_pos, ee_quat = env.franka.end_effector.get_world_pose()
                    ee_pos = np.atleast_1d(ee_pos).flatten()[:3].astype(np.float32)
                    ee_quat = np.atleast_1d(ee_quat).flatten()[:4].astype(np.float32)
                    
                    # Würfel-Positionen extrahieren (x, y, z, yaw)
                    cube_positions = []
                    cube_names = env.task.get_cube_names()
                    for cube_name in cube_names:
                        try:
                            cube = env.task.scene.get_object(cube_name)
                            cube_pos, cube_quat = cube.get_world_pose()
                            cube_pos = np.atleast_1d(cube_pos).flatten()[:3]
                            cube_quat = np.atleast_1d(cube_quat).flatten()[:4]
                            
                            # Yaw aus Quaternion extrahieren
                            rot = R.from_quat([cube_quat[1], cube_quat[2], cube_quat[3], cube_quat[0]])  # [x, y, z, w]
                            euler = rot.as_euler('xyz', degrees=False)
                            yaw = float(euler[2])  # Z-Rotation = Yaw
                            
                            cube_positions.append((float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2]), yaw))
                        except Exception as e:
                            log.warning(f"Env {i}: Konnte Würfel {cube_name} nicht finden: {e}")
                            cube_positions.append((0.0, 0.0, 0.0, 0.0))
                    
                    depth = np.zeros((CAM_RESOLUTION[0], CAM_RESOLUTION[1]), dtype=np.float32)
                    
                    # In Episode-Buffer speichern (nicht direkt im Logger)
                    # Action wird im Logger aus EE-Bewegung berechnet, nicht hier!
                    episode_data[i]["observations"].append(rgb)
                    episode_data[i]["depths"].append(depth)
                    episode_data[i]["ee_positions"].append(ee_pos)
                    episode_data[i]["ee_quaternions"].append(ee_quat)
                    episode_data[i]["cube_positions"].append(cube_positions)
                    
                    # Phase-Tracking: Aktuelle Phase vom Controller abfragen und zählen
                    try:
                        current_phase = controller._pick_place_ctrl.get_current_event()
                        if current_phase < 10:  # Nur Phasen 0-9
                            episode_data[i]["phase_timesteps"][current_phase] += 1
                            episode_data[i]["current_phase"] = current_phase
                    except Exception as e:
                        log.debug(f"Env {i}: Konnte Phase nicht abfragen: {e}")
                    
            except Exception as e:
                log.error(f"Env {i}: Fehler beim Datensammeln!")
                log.error(f"  Exception Type: {type(e).__name__}")
                log.error(f"  Exception Message: {str(e)}")
                log.error(f"  Timestep: {step_counts[i]}")
                log.error(f"  Episode ID: {active_episode_ids[i]}")
                import traceback
                log.error(f"  Full Traceback:\n{traceback.format_exc()}")
                # Re-raise für besseres Debugging
                raise
            
            # Action ausführen
            articulations[i].apply_action(action)
            step_counts[i] += 1
            
            # Episode fertig?
            if controller.is_done():
                # Validiere ob Würfel korrekt gestapelt wurden
                is_valid, reason = validate_stacking(env.task, target_positions[i])
                
                ep_id = active_episode_ids[i]
                
                if is_valid and len(episode_data[i]["observations"]) > 0:
                    # Episode erfolgreich - in zentralen Logger übertragen
                    try:
                        # Sammle echte Phase-Daten vom Controller-Tracking
                        phase_data = {}
                        phase_timesteps = episode_data[i].get("phase_timesteps", {})
                        
                        # Konvertiere Timesteps zu Phase-Daten
                        for phase_idx in range(10):
                            timesteps = phase_timesteps.get(phase_idx, 0)
                            phase_data[phase_idx] = {
                                "waypoints": timesteps,
                                "time": timesteps * (1.0 / 60.0),  # 60 Hz Simulation
                            }
                        
                        logger.start_episode(total_successful)
                        # property_params werden nicht mehr gespeichert (wie gewünscht)
                        # logger.set_episode_params(episode_data[i]["params"])  # Auskommentiert
                        
                        # Alle gesammelten Daten übertragen (neues Format)
                        # Action wird im Logger automatisch aus EE-Bewegung berechnet!
                        log.debug(f"Env {i}: Übertrage {len(episode_data[i]['observations'])} Timesteps in Logger...")
                        for t_idx, (obs, depth, ee_pos, ee_quat, cube_pos) in enumerate(zip(
                            episode_data[i]["observations"],
                            episode_data[i]["depths"],
                            episode_data[i]["ee_positions"],
                            episode_data[i]["ee_quaternions"],
                            episode_data[i]["cube_positions"],
                        )):
                            try:
                                logger.log_step(
                                    rgb_image=obs,
                                    depth_image=depth,
                                    ee_pos=ee_pos,
                                    ee_quat=ee_quat,
                                    cube_positions=cube_pos
                                )
                            except Exception as e:
                                log.error(f"Env {i}: Fehler beim log_step (Timestep {t_idx}):")
                                log.error(f"  Exception: {type(e).__name__}: {str(e)}")
                                import traceback
                                log.error(f"  Traceback:\n{traceback.format_exc()}")
                                raise
                        
                        logger.end_episode()
                        
                        # Schreibe Episode-Daten in CSV
                        try:
                            # Controller-Parameter aus globalen Konstanten (nicht vom Controller-Objekt)
                            controller_params = {
                                "trajectory_resolution": TRAJECTORY_RESOLUTION,
                                "air_speed_multiplier": AIR_SPEED_MULTIPLIER,
                                "height_adaptive_speed": HEIGHT_ADAPTIVE_SPEED,
                                "critical_height_threshold": CRITICAL_HEIGHT_THRESHOLD,
                                "critical_speed_factor": CRITICAL_SPEED_FACTOR,
                                "guarantee_final_position": GUARANTEE_FINAL_POSITION,
                                "guarantee_phases": GUARANTEE_PHASES,
                            }
                            
                            log.debug(f"CSV-Logging: Phase-Daten = {phase_data}")
                            log.debug(f"CSV-Logging: Total Timesteps = {step_counts[i]}, Total Time = {step_counts[i] * (1.0 / 60.0)}")
                            
                            csv_logger.log_episode(
                                episode_seed=total_successful,
                                controller_params=controller_params,
                                phase_data=phase_data,
                                total_timesteps=step_counts[i],
                                total_time=step_counts[i] * (1.0 / 60.0),
                                validation_success=True,
                                notes=f"Seed: {seeds[i]}, Env: {i}",
                            )
                            log.info(f"✅ CSV-Eintrag geschrieben für Episode {total_successful}")
                        except Exception as e:
                            log.warning(f"Fehler beim CSV-Logging für Episode {total_successful}: {e}")
                            import traceback
                            log.warning(f"Traceback: {traceback.format_exc()}")
                        
                        successful_counts[i] += 1
                        total_successful += 1
                        log.info(f"✅ Env {i}: Episode {total_successful} erfolgreich ({step_counts[i]} steps, Seed {seeds[i]})")
                    except Exception as e:
                        log.error(f"Env {i}: Fehler beim Speichern der Episode {total_successful}:")
                        log.error(f"  Exception: {type(e).__name__}: {str(e)}")
                        import traceback
                        log.error(f"  Traceback:\n{traceback.format_exc()}")
                        # Episode verwerfen
                        failed_seeds.append(seeds[i])
                        log.error(f"  Episode wird verworfen")
                else:
                    # Episode fehlgeschlagen - Daten verwerfen
                    failed_seeds.append(seeds[i])
                    log.warning(f"❌ Env {i}: Episode verworfen (Seed {seeds[i]}): {reason}")
                    
                    # Schreibe auch fehlgeschlagene Episode in CSV (für Tracking)
                    try:
                        # Controller-Parameter aus globalen Konstanten (nicht vom Controller-Objekt)
                        controller_params = {
                            "trajectory_resolution": TRAJECTORY_RESOLUTION,
                            "air_speed_multiplier": AIR_SPEED_MULTIPLIER,
                            "height_adaptive_speed": HEIGHT_ADAPTIVE_SPEED,
                            "critical_height_threshold": CRITICAL_HEIGHT_THRESHOLD,
                            "critical_speed_factor": CRITICAL_SPEED_FACTOR,
                            "guarantee_final_position": GUARANTEE_FINAL_POSITION,
                            "guarantee_phases": GUARANTEE_PHASES,
                        }
                        
                        # Auch für fehlgeschlagene Episoden Phase-Daten sammeln
                        failed_phase_data = {}
                        phase_timesteps = episode_data[i].get("phase_timesteps", {})
                        for phase_idx in range(10):
                            timesteps = phase_timesteps.get(phase_idx, 0)
                            failed_phase_data[phase_idx] = {
                                "waypoints": timesteps,
                                "time": timesteps * (1.0 / 60.0),
                            }
                        
                        csv_logger.log_episode(
                            episode_seed=f"FAILED_{total_episodes}",
                            controller_params=controller_params,
                            phase_data=failed_phase_data,
                            total_timesteps=step_counts[i],
                            total_time=step_counts[i] * (1.0 / 60.0),
                            validation_success=False,
                            notes=f"Seed: {seeds[i]}, Env: {i}, Grund: {reason}",
                        )
                        log.info(f"CSV-Eintrag geschrieben für fehlgeschlagene Episode {total_episodes}")
                    except Exception as e:
                        log.warning(f"Fehler beim CSV-Logging für fehlgeschlagene Episode {total_episodes}: {e}")
                        import traceback
                        log.warning(f"Traceback: {traceback.format_exc()}")
                
                episode_counts[i] += 1
                total_episodes += 1
                step_counts[i] = 0
                seeds[i] += 1
                
                # Weitere Episoden?
                if successful_counts[i] < episodes_per_env and total_successful < NUM_EPISODES:
                    controller.reset()
                    cube_pos, target_pos = env.domain_randomization(seeds[i])
                    target_positions[i] = target_pos
                    cube_positions_list[i] = cube_pos
                    
                    # Neuen Episode-Buffer initialisieren
                    # Action wird im Logger automatisch aus EE-Bewegung berechnet!
                    active_episode_ids[i] = global_episode_id
                    episode_data[i] = {
                        "observations": [],
                        "depths": [],
                        "ee_positions": [],
                        "ee_quaternions": [],
                        "cube_positions": [],
                        "phase_timesteps": {p: 0 for p in range(10)},  # Timesteps pro Phase (0-9)
                        "current_phase": 0,  # Aktuelle Phase für Tracking
                        "params": {
                            "seed": seeds[i],
                            "env_idx": i,
                            "cube_positions": [p.tolist() for p in cube_pos],
                            "target_position": target_pos.tolist(),
                        }
                    }
                    global_episode_id += 1
                else:
                    env_done[i] = True
                    log.info(f"Env {i}: Fertig ({successful_counts[i]} erfolgreiche Episoden)")
        
        # World Step
        shared_world.step()
        
        # Alle fertig?
        if all(env_done):
            break
    
    # ================================================================
    # FINALES SPEICHERN - Ein zentraler Datensatz
    # ================================================================
    # Speichere Kamera-Kalibrierung (wird am Ende gespeichert)
    logger.save_camera_calibration()
    log.info(f"Zentraler Datensatz gespeichert: {logger.dataset_path}/")
    log.info(f"  Episoden: {total_successful}")
    log.info(f"  Format: Rope-kompatibel (H5-Dateien pro Timestep)")
    
    # Speichere fehlgeschlagene Seeds
    if failed_seeds:
        failed_seeds_path = logger.dataset_path / "failed_seeds.txt"
        os.makedirs(os.path.dirname(failed_seeds_path), exist_ok=True)
        with open(failed_seeds_path, "w") as f:
            f.write("# Fehlgeschlagene Seeds - Würfel nicht korrekt gestapelt\n")
            f.write(f"# Datum: {datetime.now().isoformat()}\n")
            f.write(f"# Anzahl: {len(failed_seeds)}\n")
            f.write(f"# Parallele Envs: {NUM_ENVS}\n\n")
            for s in failed_seeds:
                f.write(f"{s}\n")
        log.info(f"Fehlgeschlagene Seeds gespeichert: {failed_seeds_path}")
    
    log.info("=" * 60)
    log.info(f"Datensammlung abgeschlossen!")
    log.info(f"  Zentraler Datensatz: {DATASET_NAME}")
    log.info(f"  Erfolgreiche Episoden: {total_successful}")
    log.info(f"  Episoden pro Env: {successful_counts}")
    log.info(f"  Fehlgeschlagene Seeds: {len(failed_seeds)}")
    if total_episodes > 0:
        log.info(f"  Erfolgsrate: {total_successful / total_episodes * 100:.1f}%")
        log.info(f"  Durchschnittliche Steps: {sum(step_counts)/NUM_ENVS}, Controller dt = {event_dt}")
    log.info("=" * 60)
    
    # Speichere CSV-Matrix mit allen gesammelten Episode-Daten
    try:
        log.info("Speichere CSV-Matrix...")
        csv_logger.save_matrix()
    except Exception as e:
        log.warning(f"WARNUNG: Fehler beim Speichern der CSV-Matrix: {e}")
    
    try:
        simulation_app.close()
    except Exception as e:
        log.error(f"Fehler beim Schließen der Simulation:")
        log.error(f"  Exception: {type(e).__name__}: {str(e)}")
        import traceback
        log.error(f"  Traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Programm durch Benutzer unterbrochen (Ctrl+C)")
        raise
    except Exception as e:
        log.error("=" * 80)
        log.error("KRITISCHER FEHLER - Programm wird beendet")
        log.error(f"Exception Type: {type(e).__name__}")
        log.error(f"Exception Message: {str(e)}")
        import traceback
        log.error("Full Traceback:")
        for line in traceback.format_exception(type(e), e, e.__traceback__):
            log.error(line.rstrip())
        log.error("=" * 80)
        raise


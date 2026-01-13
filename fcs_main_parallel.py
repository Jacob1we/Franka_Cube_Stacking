"""
Franka Cube Stacking mit Data Logging für dino_wm Training.
PARALLEL VERSION - Unterstützt mehrere Umgebungen gleichzeitig.

Basiert auf franka_cube_stack_reworked.py mit integriertem DataLogger.
"""

import isaacsim
from isaacsim import SimulationApp

from datetime import datetime
import logging, os, sys
import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

launch_config = {"headless": False}  # True für schnellere Datensammlung
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


# Data Logger Import
from data_logger import FrankaDataLogger, get_franka_state, get_franka_action

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

# ============================================================================
# PARALLELISIERUNG - Setze NUM_ENVS > 1 für parallele Datensammlung
# ============================================================================
NUM_ENVS = 4                 # 1 = Single, >1 = Parallel (z.B. 4 für 2x2 Grid)
ENV_SPACING = 2.5            # Abstand zwischen Umgebungen (Meter)

# Datensammlung Konfiguration
NUM_EPISODES = 10 #100           # Anzahl zu sammelnder Episoden (TOTAL, wird auf Envs verteilt)
DATASET_PATH = "/media/tsp_jw/fc8bca1b-cab8-4522-81d0-06172d2beae8/franka_cube_stack_10" #"./dataset"
DATASET_NAME = "franka_cube_stack_ds"
SAVE_PNG = True              # Speichere alle Bilder auch als PNG

# Kamera
SIDE_CAM_BASE_POS = np.array([1.6, -2.0, 1.27])
SIDE_CAM_EULER = (66.0, 0.0, 32.05)
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
YAW_RANGE = (-45.0, 45.0)  # REDUZIERT! War (-180, 180) - verursachte EE-Rotation Probleme

# Workspace-Grenzen (Franka Panda)
FRANKA_MAX_REACH = 0.75     # Maximale Reichweite (konservativ, echte: ~0.855m)
FRANKA_MIN_REACH = 0.3     # Minimale Reichweite (zu nah = Selbstkollision)

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

# Validierung
XY_TOLERANCE = 0.03          # Toleranz für X/Y Position (3 cm)
Z_MIN_HEIGHT = 0.02          # Mindesthöhe über Boden (2 cm)
Z_STACK_TOLERANCE = 0.02     # Toleranz für Z-Stacking
#endregion


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
            preferred_joints=PRESET_LOCK_THREE,  # Soft constraint: prefer neutral pose   PRESET_MINIMAL_MOTION, PRESET_ESSENTIAL_ONLY
            trajectory_resolution=1.5,               # Base resolution (affects ALL phases)
            air_speed_multiplier=4.0,                # Speed up AIR phases only (0,4,5,8,9)
            height_adaptive_speed=True,              # DYNAMIC: Slow down near ground!
            critical_height_threshold=0.05,           # Below xx cm = critical zone
            critical_speed_factor=0.8,               # slower in critical zone
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


def main():
    """
    Hauptfunktion - unterstützt Single und Parallel Mode.
    
    Single Mode (NUM_ENVS=1): Sequentielle Datensammlung
    Parallel Mode (NUM_ENVS>1): Parallele Datensammlung mit Grid-Layout
    """
    
    log.info("=" * 60)
    log.info(f"Franka Cube Stacking - {'PARALLEL' if NUM_ENVS > 1 else 'SINGLE'} Mode")
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
    
    # Controller und Kameras für alle Envs
    for i, env in enumerate(envs):
        controller = env.setup_post_load()
        controllers.append(controller)
        
        articulation = env.franka.get_articulation_controller()
        articulations.append(articulation)
        
        camera = env.add_scene_cam()
        camera.initialize()
        cameras.append(camera)
    
    simulation_context = SimulationContext()
    
    # ================================================================
    # DATA LOGGER SETUP
    # ================================================================
    current_folder = os.path.basename(os.getcwd())
    if current_folder.lower() == "isaacsim":
        local_save_path = "./00_my_envs/Franka_Cube_Stacking_JW/dataset"
    else: 
        local_save_path = DATASET_PATH
    
    # ================================================================
    # ZENTRALER LOGGER - Ein Datensatz für alle Environments
    # ================================================================
    logger = FrankaDataLogger(
        save_path=local_save_path,
        object_name=DATASET_NAME,  # Ein gemeinsamer Datensatz
        image_size=CAM_RESOLUTION,
        max_timesteps=None,
        save_png=SAVE_PNG,
    )
    
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
        active_episode_ids[i] = global_episode_id
        episode_data[i] = {
            "observations": [],
            "states": [],
            "actions": [],
            "params": {
                "seed": seeds[i],
                "env_idx": i,
                "cube_positions": [p.tolist() for p in cube_pos],
                "target_position": target_pos.tolist(),
            }
        }
        global_episode_id += 1
    
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
                rgba = camera.get_rgba()
                if rgba is not None:
                    rgb = rgba[:, :, :3]
                    state = get_franka_state(env.franka, env.task)
                    action_vec = get_franka_action(action)
                    
                    # In Episode-Buffer speichern (nicht direkt im Logger)
                    episode_data[i]["observations"].append(rgb)
                    episode_data[i]["states"].append(state)
                    episode_data[i]["actions"].append(action_vec)
            except Exception as e:
                log.warning(f"Env {i}: Fehler beim Sammeln: {e}")
            
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
                    logger.start_episode(total_successful)
                    logger.set_episode_params(episode_data[i]["params"])
                    
                    # Alle gesammelten Daten übertragen
                    for obs, st, act in zip(
                        episode_data[i]["observations"],
                        episode_data[i]["states"],
                        episode_data[i]["actions"]
                    ):
                        logger.log_step(rgb_image=obs, state=st, action=act)
                    
                    logger.end_episode()
                    successful_counts[i] += 1
                    total_successful += 1
                    log.info(f"✅ Env {i}: Episode {total_successful} erfolgreich ({step_counts[i]} steps, Seed {seeds[i]})")
                else:
                    # Episode fehlgeschlagen - Daten verwerfen
                    failed_seeds.append(seeds[i])
                    log.warning(f"❌ Env {i}: Episode verworfen (Seed {seeds[i]}): {reason}")
                
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
                    active_episode_ids[i] = global_episode_id
                    episode_data[i] = {
                        "observations": [],
                        "states": [],
                        "actions": [],
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
        
        # Zwischenspeichern alle 10 erfolgreichen Episoden
        if total_successful > 0 and total_successful % 10 == 0:
            logger.save_dataset()
            log.info(f"Zwischenstand: {total_successful}/{NUM_EPISODES} erfolgreiche Episoden")
        
        # Alle fertig?
        if all(env_done):
            break
    
    # ================================================================
    # FINALES SPEICHERN - Ein zentraler Datensatz
    # ================================================================
    logger.save_dataset()
    log.info(f"Zentraler Datensatz gespeichert: {local_save_path}/{DATASET_NAME}/")
    
    # Speichere fehlgeschlagene Seeds
    if failed_seeds:
        failed_seeds_path = os.path.join(local_save_path, DATASET_NAME, "failed_seeds.txt")
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
    log.info("=" * 60)
    
    simulation_app.close()


if __name__ == "__main__":
    main()


import isaacsim
from isaacsim import SimulationApp

from pathlib import Path
from datetime import datetime
import logging, os, sys
import matplotlib.pyplot as plt
import numpy as np


# simulation_app = SimulationApp({"headless": False})
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# See DEFAULT_LAUNCHER_CONFIG for available configuration
# https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
launch_config = {"headless": False}
# Launch the Toolkit
simulation_app = SimulationApp(launch_config)

# Locate any other import statement after this point
import omni

# # from isaacsim.examples.interactive.base_sample import BaseSample   DOES NOT WORK FOR STANDALONE use the following:
# from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("omni.isaac.examples")
# from omni.isaac.examples.base_sample import BaseSample
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

#region: Constants
SEED = 111
Problemseeds =  [10, 11, 12, 14]
WORLD_ROOT = "/World"

NUM_SCENES = 4

# SIDE_CAM_BASE_POS = np.array([2.4, -3.2, 2.2]) # m  
# SIDE_CAM_EULER = (63.2, 0.0, 33.0) # deg
SIDE_CAM_BASE_POS = np.array([1.41, -1.67, 1.27]) # m  
SIDE_CAM_EULER = (60.84, 0.0, 32.05) # deg

SCENE_WIDTH  = 0.60  # m
SCENE_LENGTH = 0.75  # m
FRANKA_BASE_CLEARANCE = 0.3 # m
PLANE_LIFT = 0.001

N_CUBES = 2
CUBE_SIDE = 0.05  # für Mindestabstand und Würfelgröße
MIN_DIST = 1.5 * CUBE_SIDE  # Optional: verhindert Überschneidungen der Startwürfel
MAX_TRIES = 200
YAW_RANGE: tuple = (-180.0, 180.0) #(-5.0, 5.0) #(-180.0, 180.0)   # für mode="yaw"
CAM_FEQUENCY = 20 # Hz
CAM_RESOLUTION = (256,256)

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
        world_xform = prims_utils.get_prim_at_path(prim_path=self.world_prim_path)
        print(world_xform)

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
        log.info("World Resetted")
        return

    def setup_post_load(self):
        log.info("Setup Post Load")
        robot_name = self.task.get_params()["robot_name"]["value"]
        self.franka = self.world.scene.get_object(robot_name)

        base_pos, base_quat = self.franka.get_local_pose()
        self.base_pos = base_pos
        self.base_quat = base_quat
        log.info(f"Base Position: {base_pos}")

        controller = StackingController_JW(
            name=f"stacking_controller",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
            picking_order_cube_names=self.task.get_cube_names(),
            robot_observation_name=robot_name,
        )

        return controller
    
    def set_scene_light(self, light_seed, light_prim_path = None):

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
    
        # Kamerapfad (eindeutiger Pfad je Szene)
        cam_xform_path = f"{self.task_root}/camera_xform"
        cam_prim_path = f"{cam_xform_path}/camera"
        
        UsdGeom.Xform.Define(self.stage, cam_xform_path)
        UsdGeom.Camera.Define(self.stage, cam_prim_path)

        # Kamera relativ zur Szene platzieren
        cam_xform_api = UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(cam_xform_path))
        cam_xform_api.SetTranslate(Gf.Vec3d(*SIDE_CAM_BASE_POS))     
        cam_xform_api.SetRotate(Gf.Vec3f(*SIDE_CAM_EULER))

        cam = Camera(
            prim_path=cam_prim_path,
            frequency=CAM_FEQUENCY,
            resolution=CAM_RESOLUTION,
        )
        return cam

    def get_materials(self,plane_root):
        self.materials = []
        for name,rgba in ALLOWED_AREA_MATS:
            mat_path = f"{plane_root}/{name}"
            mat = UsdShade.Material.Get(self.stage, mat_path)
            if not mat:
                mat = UsdShade.Material.Define(self.stage, mat_path)
                shader = UsdShade.Shader.Define(self.stage, f"{plane_root}/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgba[:3])
                shader.CreateInput("roughness",   Sdf.ValueTypeNames.Float).Set(0.6)
                shader.CreateInput("metallic",    Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("opacity",     Sdf.ValueTypeNames.Float).Set(rgba[3])

                surf_out   = mat.CreateSurfaceOutput()
                shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                surf_out.ConnectToSource(shader_out)
            self.materials.append(mat)
    
    def add_or_update_plane(self,seed):
        plane_root = f"{self.task_root}/Plane"

        mesh = UsdGeom.Mesh.Get(self.stage, plane_root)
        if not mesh:
            mesh = UsdGeom.Mesh.Define(self.stage, plane_root)
        plane_prim = mesh.GetPrim()

        def make_point(x, y):
            p = self.base_pos + np.array([x, y, PLANE_LIFT])
            return Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
        

        p0_w = make_point( 0.0,-SCENE_WIDTH/2)
        p1_w = make_point( 0.0, SCENE_WIDTH/2)
        p2_w = make_point( SCENE_LENGTH, SCENE_WIDTH/2)
        p3_w = make_point( SCENE_LENGTH,-SCENE_WIDTH/2)
    

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
        log.info(self.materials)
        log.info(random_material_index)
        log.info(material)
        UsdShade.MaterialBindingAPI(plane_prim).Bind(material)

        return mesh

    def domain_randomization(self, seed):
        log.info("Start Randomization")
        self.rng = np.random.default_rng(seed)

        def sample_cube_positions():
            
            points = []
            w = 0.6*CUBE_SIDE

            def point_valid(point_xy):
                if np.linalg.norm(point_xy-self.base_pos[:2]) < FRANKA_BASE_CLEARANCE:
                    return False
                for p in points:
                    if np.linalg.norm(point_xy - p[:2]) < MIN_DIST:
                        return False
                return True
            
            for _ in range(N_CUBES+1): #+1 wegen stacking target 
                fallback_point = np.array([0.5, 0.0, 1.1*CUBE_SIDE])
                found_valid = False
                for _ in range(MAX_TRIES):
                    u = self.rng.uniform(0, SCENE_LENGTH)
                    v = self.rng.uniform(-SCENE_WIDTH/2, SCENE_WIDTH/2)
                    # u = self.rng.uniform(-SCENE_WIDTH/2, SCENE_WIDTH/2)
                    # v = self.rng.uniform(0, SCENE_LENGTH)
                    
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
                cube_rng = np.random.default_rng(seed+n)
                yaw_deg = float(cube_rng.uniform(*YAW_RANGE))
                orientation = rotations_utils.euler_angles_to_quat(np.array([0.0, 0.0, yaw_deg]), degrees=True)
                orientations.append(orientation)
            return orientations
        
        def cube_randomization_in_existing_task():
            cube_names =self.task.get_cube_names()
            sample_points = sample_cube_positions()
            log.info(f"Sample Points: \n{sample_points}")

            cube_sample_points = sample_points[:N_CUBES]
            stack_target_sample_point = sample_points[N_CUBES]
            sample_orientations = sample_cube_orientation(seed)

            for n, name in enumerate(cube_names):
                cube_name = name
                cube_pos = cube_sample_points[n]
                cube_ori = sample_orientations[n]
                cube_target = stack_target_sample_point
                self.task.set_params(cube_name, cube_pos, cube_ori, cube_target)
            return

        cube_randomization_in_existing_task()

        self.add_or_update_plane(seed)
        
        self.set_scene_light(seed)

        return
    
def main():
    seed = SEED+50
    env = Franka_Cube_Stack()
    env.setup_world()
    controller = env.setup_post_load()
    env.domain_randomization(seed)

    simulation_context = SimulationContext()
    articulation = env.franka.get_articulation_controller()
    camera = env.add_scene_cam()
    camera.initialize()

    while simulation_app.is_running():
        simulation_app.update()
        env.world.step()
        obs = env.task.get_observations()
        # joint_positions = env.franka.get_joint_positions()
        # joint_velocities = env.franka.get_joint_velocities()
        # t = np.round(simulation_context.current_time,2)
        
        # if t%1 == 0:
        #     log.info(f"============================ Sekunde {np.round(t,2)} ============================")
        #     log.info(f"------------- Observations -------------")
        #     cube_0_ori = rotations_utils.quat_to_euler_angles(obs[env.task.get_cube_names()[0]]["orientation"])
        #     log.info(cube_0_ori)
        #     log.info(f"------------- Joint Positions -------------")
        #     log.info(joint_positions)
        #     log.info(f"------------- Joint Velocities -------------")
        #     log.info(joint_velocities)

        action = controller.forward(observations=obs)
        articulation.apply_action(action)

        if controller.is_done():
            try:
                rgba = camera.get_rgba()
                image_path = f"{env.logdir}/00_Screenshots/Episode_{seed:03d}.png"
                plt.imsave(image_path, rgba)
            except Exception as e:
                log.warning(f"Screenshot fehlgeschlagen: {e}")

            env.world.reset()
            controller.reset()
            env.world.step()
            seed += 1
            env.domain_randomization(seed)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
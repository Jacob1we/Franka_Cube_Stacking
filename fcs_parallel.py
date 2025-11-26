"""
Franka Cube Stacking - Parallele Datensammlung mit Multiple Tasks.

Verwendet mehrere Stacking Tasks mit Offset für parallele Datensammlung.
Dieser Ansatz ist stabiler als reines Cloning, da jeder Task seinen eigenen
vollständigen Roboter und Controller hat.
"""

import isaacsim
from isaacsim import SimulationApp

from pathlib import Path
from datetime import datetime
import logging, os, sys
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ============================================================================
# KONFIGURATION
# ============================================================================
NUM_ENVS = 4                    # Anzahl paralleler Umgebungen (2x2 Grid)
ENV_SPACING = 2.0               # Abstand zwischen Umgebungen (Meter)
NUM_EPISODES_PER_ENV = 25       # Episoden pro Umgebung
HEADLESS = False                # True für schnellere Sammlung
SAVE_IMAGES = True              # Bilder speichern

SEED = 42
CUBE_SIDE = 0.05
N_CUBES = 2
CAM_RESOLUTION = (256, 256)
DATASET_PATH = "./dataset_parallel"

launch_config = {"headless": HEADLESS}
simulation_app = SimulationApp(launch_config)

from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.sensors.camera import Camera

# Custom imports
from Franka_Env_JW import Stacking_JW, StackingController_JW
from Franka_Env_JW.rmpflow_controller_jw import PRESET_MINIMAL_MOTION
from data_logger import FrankaDataLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("FrankaParallel")


class ParallelStackingEnv:
    """
    Parallele Franka Stacking Umgebung mit mehreren Tasks.
    
    Jeder Task hat:
    - Eigenen Franka Roboter
    - Eigene Würfel
    - Eigenen Controller
    - Eigene Kamera (optional)
    
    Alle Tasks laufen synchron im selben Physics Step.
    """
    
    def __init__(
        self, 
        num_envs: int = 4, 
        env_spacing: float = 2.0,
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.base_seed = seed
        
        self.world = None
        self.stage = None
        
        # Pro-Umgebung Objekte
        self.tasks = []
        self.frankas = []
        self.controllers = []
        self.articulation_controllers = []
        self.cameras = []
        
        # Episode-Tracking
        self.episode_counts = [0] * num_envs
        self.episode_steps = [0] * num_envs
        self.episode_seeds = [seed + i * 1000 for i in range(num_envs)]
        self.env_done = [False] * num_envs
        
        # Grid-Layout berechnen (2D)
        self.grid_size = int(np.ceil(np.sqrt(num_envs)))
        self.offsets = self._compute_offsets()
        
        log.info(f"Grid size: {self.grid_size}x{self.grid_size}")
        log.info(f"Offsets: {self.offsets}")
    
    def _compute_offsets(self) -> list:
        """Berechnet die Offsets für das Grid-Layout."""
        offsets = []
        for i in range(self.num_envs):
            row = i // self.grid_size
            col = i % self.grid_size
            offset = np.array([
                col * self.env_spacing,
                row * self.env_spacing,
                0.0
            ])
            offsets.append(offset)
        return offsets
    
    def setup(self):
        """Erstellt alle parallelen Umgebungen."""
        log.info(f"Setting up {self.num_envs} parallel stacking environments...")
        
        # Welt erstellen
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = get_current_stage()
        
        # ================================================================
        # Tasks erstellen (jeder Task ist eine vollständige Umgebung)
        # ================================================================
        for i in range(self.num_envs):
            log.info(f"Creating environment {i}...")
            
            # Task mit Offset erstellen
            task = Stacking_JW(
                name=f"stacking_task_{i}",
                cube_size=[CUBE_SIDE] * 3,
                offset=self.offsets[i].tolist(),
                parent_prim_path=f"/World/Env_{i}",
                cube_amount=N_CUBES,
            )
            self.world.add_task(task)
            self.tasks.append(task)
        
        # Welt initialisieren
        self.world.reset()
        
        # ================================================================
        # Roboter und Controller für jede Umgebung
        # ================================================================
        for i in range(self.num_envs):
            task = self.tasks[i]
            task_params = task.get_params()
            robot_name = task_params["robot_name"]["value"]
            
            # Franka aus der Szene holen
            franka = self.world.scene.get_object(robot_name)
            self.frankas.append(franka)
            
            # Controller erstellen
            controller = StackingController_JW(
                name=f"controller_{i}",
                gripper=franka.gripper,
                robot_articulation=franka,
                picking_order_cube_names=task.get_cube_names(),
                robot_observation_name=robot_name,
                preferred_joints=PRESET_MINIMAL_MOTION,
                trajectory_resolution=1.0,
                air_speed_multiplier=4.0,
                height_adaptive_speed=True,
                critical_height_threshold=0.12,
                critical_speed_factor=0.25,
            )
            self.controllers.append(controller)
            
            # Articulation Controller
            art_ctrl = franka.get_articulation_controller()
            self.articulation_controllers.append(art_ctrl)
        
        # ================================================================
        # Kameras (optional)
        # ================================================================
        if SAVE_IMAGES:
            for i in range(self.num_envs):
                cam = self._create_camera(i)
                self.cameras.append(cam)
        
        log.info("Setup complete!")
        log.info(f"Total Frankas: {len(self.frankas)}")
        log.info(f"Total Controllers: {len(self.controllers)}")
    
    def _create_camera(self, env_idx: int) -> Camera:
        """Erstellt eine Kamera für eine Umgebung."""
        offset = self.offsets[env_idx]
        cam_pos = np.array([1.5, -1.5, 1.2]) + offset
        
        cam_path = f"/World/Env_{env_idx}/Camera"
        xform_path = f"{cam_path}_xform"
        
        UsdGeom.Xform.Define(self.stage, xform_path)
        xform_api = UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(xform_path))
        xform_api.SetTranslate(Gf.Vec3d(*cam_pos))
        xform_api.SetRotate(Gf.Vec3f(60.0, 0.0, 30.0))
        
        UsdGeom.Camera.Define(self.stage, cam_path)
        
        camera = Camera(
            prim_path=cam_path,
            frequency=20,
            resolution=CAM_RESOLUTION,
        )
        return camera
    
    def get_observations(self) -> dict:
        """Sammelt Beobachtungen von allen Umgebungen."""
        all_obs = self.world.get_observations()
        return all_obs
    
    def step(self) -> list:
        """
        Führt einen Simulationsschritt für alle Umgebungen aus.
        
        Returns:
            list: Indizes der Umgebungen, die ihre Episode beendet haben
        """
        observations = self.get_observations()
        completed_envs = []
        
        for i in range(self.num_envs):
            if self.env_done[i]:
                continue
            
            task = self.tasks[i]
            controller = self.controllers[i]
            franka = self.frankas[i]
            
            # Action berechnen
            action = controller.forward(
                observations=observations,
                end_effector_orientation=None,
                end_effector_offset=np.array([0, 0, 0]),
            )
            
            # Action anwenden
            self.articulation_controllers[i].apply_action(action)
            
            self.episode_steps[i] += 1
            
            # Prüfen ob Episode fertig
            if controller.is_done():
                completed_envs.append(i)
                self.episode_counts[i] += 1
                self.env_done[i] = True
        
        # Simulation schritt
        self.world.step(render=not HEADLESS)
        
        return completed_envs
    
    def reset_env(self, env_idx: int):
        """Setzt eine einzelne Umgebung zurück."""
        self.episode_seeds[env_idx] += 1
        seed = self.episode_seeds[env_idx]
        
        # Controller zurücksetzen
        self.controllers[env_idx].reset()
        
        # Episode-Tracking
        self.episode_steps[env_idx] = 0
        self.env_done[env_idx] = False
        
        log.debug(f"Env {env_idx} reset with seed {seed}")
    
    def reset_all(self):
        """Setzt alle Umgebungen zurück."""
        self.world.reset()
        
        for i in range(self.num_envs):
            self.controllers[i].reset()
            self.episode_steps[i] = 0
            self.env_done[i] = False
    
    def get_camera_images(self) -> list:
        """Sammelt Bilder von allen Kameras."""
        images = []
        for cam in self.cameras:
            rgba = cam.get_rgba()
            images.append(rgba)
        return images
    
    def run_data_collection(
        self, 
        num_episodes_per_env: int = 25,
        max_steps_per_episode: int = 2000,
    ) -> dict:
        """
        Führt die parallele Datensammlung durch.
        
        Args:
            num_episodes_per_env: Anzahl Episoden pro Umgebung
            max_steps_per_episode: Maximale Schritte pro Episode
            
        Returns:
            dict: Statistiken über die Datensammlung
        """
        log.info("=" * 60)
        log.info(f"Starting parallel data collection")
        log.info(f"  - Environments: {self.num_envs}")
        log.info(f"  - Episodes per env: {num_episodes_per_env}")
        log.info(f"  - Total episodes: {self.num_envs * num_episodes_per_env}")
        log.info("=" * 60)
        
        # Initialisieren
        self.reset_all()
        
        total_steps = 0
        total_completed = 0
        start_time = datetime.now()
        
        # Hauptschleife
        while simulation_app.is_running():
            if self.world.is_stopped():
                break
            
            if not self.world.is_playing():
                self.world.step(render=True)
                continue
            
            # Einen Schritt ausführen
            completed_envs = self.step()
            total_steps += 1
            
            # Abgeschlossene Episoden verarbeiten
            for env_idx in completed_envs:
                total_completed += 1
                log.info(f"Env {env_idx}: Episode {self.episode_counts[env_idx]} completed "
                        f"(steps: {self.episode_steps[env_idx]})")
                
                # Noch mehr Episoden?
                if self.episode_counts[env_idx] < num_episodes_per_env:
                    self.reset_env(env_idx)
            
            # Timeout-Check für steckengebliebene Episoden
            for i in range(self.num_envs):
                if not self.env_done[i] and self.episode_steps[i] > max_steps_per_episode:
                    log.warning(f"Env {i}: Episode timeout after {self.episode_steps[i]} steps")
                    self.env_done[i] = True
                    self.reset_env(i)
            
            # Alle fertig?
            if all(count >= num_episodes_per_env for count in self.episode_counts):
                log.info("All episodes completed!")
                break
            
            # Fortschritt loggen
            if total_steps % 500 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                eps_per_sec = total_completed / elapsed if elapsed > 0 else 0
                log.info(f"Step {total_steps} | Completed: {self.episode_counts} | "
                        f"Rate: {eps_per_sec:.2f} eps/sec")
        
        # Statistiken
        elapsed = (datetime.now() - start_time).total_seconds()
        stats = {
            "total_episodes": sum(self.episode_counts),
            "episodes_per_env": self.episode_counts,
            "total_steps": total_steps,
            "elapsed_seconds": elapsed,
            "episodes_per_second": sum(self.episode_counts) / elapsed if elapsed > 0 else 0,
        }
        
        log.info("=" * 60)
        log.info("Data Collection Complete!")
        log.info(f"  - Total episodes: {stats['total_episodes']}")
        log.info(f"  - Total steps: {stats['total_steps']}")
        log.info(f"  - Time: {elapsed:.1f}s")
        log.info(f"  - Rate: {stats['episodes_per_second']:.2f} episodes/second")
        log.info("=" * 60)
        
        return stats


def main():
    log.info("=" * 60)
    log.info("Franka Cube Stacking - Parallel Data Collection")
    log.info(f"Number of parallel environments: {NUM_ENVS}")
    log.info(f"Episodes per environment: {NUM_EPISODES_PER_ENV}")
    log.info(f"Total episodes: {NUM_ENVS * NUM_EPISODES_PER_ENV}")
    log.info("=" * 60)
    
    # Umgebung erstellen
    env = ParallelStackingEnv(
        num_envs=NUM_ENVS,
        env_spacing=ENV_SPACING,
        seed=SEED,
    )
    
    try:
        # Setup
        env.setup()
        
        # Simulation starten
        env.world.play()
        
        # Einige Schritte zur Stabilisierung
        for _ in range(10):
            env.world.step(render=True)
        
        # Datensammlung starten
        stats = env.run_data_collection(NUM_EPISODES_PER_ENV)
        
        log.info(f"Final statistics: {stats}")
        
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

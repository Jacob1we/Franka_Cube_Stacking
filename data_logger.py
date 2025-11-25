"""
Data Logger für Isaac Sim - Franka Panda Datensatz-Generierung
Struktur kompatibel mit dem deformable/rope Format für dino_wm Training.

Verwendung:
    1. Importiere DataLogger in dein Environment
    2. Rufe logger.start_episode() am Anfang jeder Episode auf
    3. Rufe logger.log_step(...) bei jedem Timestep auf
    4. Rufe logger.end_episode() am Ende jeder Episode auf
    5. Rufe logger.save_dataset() am Ende des Trainings auf
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

# Für PNG-Speicherung
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL nicht verfügbar - versuche matplotlib für PNG-Speicherung")

# h5py ist optional - nur für Kompatibilität mit dem Original-Datensatz
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warning("h5py nicht verfügbar - H5-Dateien werden nicht gespeichert")

log = logging.getLogger("DataLogger")


class FrankaDataLogger:
    """
    Data Logger für Franka Panda Robot in Isaac Sim.
    Speichert Daten im Format kompatibel mit DeformDataset.
    """
    
    def __init__(
        self,
        save_path: str = "./dataset",
        object_name: str = "franka_cube_stack",
        image_size: tuple = (256, 256),
        max_timesteps: Optional[int] = None,  # None = unbegrenzt (bis Controller fertig)
        save_png: bool = True,  # Speichere Bilder auch als PNG
    ):
        """
        Args:
            save_path: Basis-Verzeichnis für den Datensatz
            object_name: Name des Datensatzes (Unterordner)
            image_size: (H, W) der Kamerabilder
            max_timesteps: Maximale Anzahl Timesteps pro Episode (None = unbegrenzt)
            save_png: Wenn True, werden alle Bilder auch als PNG gespeichert
        """
        self.save_path = Path(save_path)
        self.object_name = object_name
        self.dataset_path = self.save_path / object_name
        self.image_size = image_size
        self.max_timesteps = max_timesteps
        self.save_png = save_png
        
        # Temporärer Speicher für aktuelle Episode
        self.current_episode = None
        self.episode_count = 0
        
        # Akkumulierte Daten über alle Episoden
        self.all_states: List[torch.Tensor] = []
        self.all_actions: List[torch.Tensor] = []
        self.all_episode_lengths: List[int] = []
        
        # Kamera-Kalibrierung (optional)
        self.camera_intrinsic: Optional[np.ndarray] = None
        self.camera_extrinsic: Optional[np.ndarray] = None
        
        # Erstelle Verzeichnisstruktur
        self._setup_directories()
        
        log.info(f"DataLogger initialisiert: {self.dataset_path}")
    
    def _setup_directories(self):
        """Erstellt die Verzeichnisstruktur."""
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "cameras").mkdir(exist_ok=True)
    
    def set_camera_calibration(
        self, 
        intrinsic: np.ndarray, 
        extrinsic: np.ndarray
    ):
        """
        Setzt die Kamera-Kalibrierungsmatrizen.
        
        Args:
            intrinsic: 3x3 Kamera-Intrinsik-Matrix
            extrinsic: 4x4 Kamera-Extrinsik-Matrix (World -> Camera)
        """
        self.camera_intrinsic = intrinsic
        self.camera_extrinsic = extrinsic
    
    def start_episode(self, episode_id: Optional[int] = None):
        """
        Startet eine neue Episode für das Logging.
        
        Args:
            episode_id: Optional: Explizite Episode-ID, sonst auto-increment
        """
        if episode_id is not None:
            self.episode_count = episode_id
        
        self.current_episode = {
            "id": self.episode_count,
            "timestep": 0,
            "observations": [],  # RGB Bilder (T, H, W, C)
            "states": [],        # Roboter/Objekt-Zustände (T, state_dim)
            "actions": [],       # Aktionen (T, action_dim)
            "h5_data": [],       # Zusätzliche Daten pro Timestep
            "property_params": {}
        }
        
        log.info(f"Episode {self.episode_count} gestartet")
    
    def log_step(
        self,
        rgb_image: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Loggt einen einzelnen Timestep.
        
        Args:
            rgb_image: RGB Bild (H, W, 3) uint8, Werte 0-255
            state: Zustandsvektor (state_dim,) - z.B. EE-Pose + Joints
            action: Aktionsvektor (action_dim,) - z.B. Joint-Velocities
            depth_image: Optional Tiefenbild (H, W) float32
            additional_data: Optional zusätzliche Daten für H5-Datei
        """
        if self.current_episode is None:
            raise RuntimeError("Keine Episode gestartet! Rufe start_episode() auf.")
        
        # Prüfe max_timesteps nur wenn gesetzt
        if self.max_timesteps is not None and self.current_episode["timestep"] >= self.max_timesteps:
            log.warning(f"Max timesteps ({self.max_timesteps}) erreicht, überspringe...")
            return
        
        # Bild speichern
        if rgb_image.shape[:2] != self.image_size:
            log.warning(f"Bildgröße {rgb_image.shape[:2]} != {self.image_size}")
        self.current_episode["observations"].append(rgb_image.astype(np.uint8))
        
        # State und Action speichern
        self.current_episode["states"].append(state.astype(np.float32))
        self.current_episode["actions"].append(action.astype(np.float32))
        
        # H5 Zusatzdaten (optional)
        h5_entry = {"timestep": self.current_episode["timestep"]}
        if depth_image is not None:
            h5_entry["depth"] = depth_image.astype(np.float32)
        if additional_data:
            h5_entry.update(additional_data)
        self.current_episode["h5_data"].append(h5_entry)
        
        self.current_episode["timestep"] += 1
    
    def set_episode_params(self, params: Dict[str, Any]):
        """
        Setzt zusätzliche Parameter für die aktuelle Episode.
        Wird als property_params.pkl gespeichert.
        
        Args:
            params: Dictionary mit beliebigen Parametern
        """
        if self.current_episode is None:
            raise RuntimeError("Keine Episode gestartet!")
        self.current_episode["property_params"].update(params)
    
    def end_episode(self, save_immediately: bool = True):
        """
        Beendet die aktuelle Episode und speichert die Daten.
        
        Args:
            save_immediately: Wenn True, werden Episode-Daten sofort auf Disk gespeichert
        """
        if self.current_episode is None:
            log.warning("Keine Episode zum Beenden vorhanden")
            return
        
        episode_length = self.current_episode["timestep"]
        if episode_length == 0:
            log.warning("Leere Episode, überspringe...")
            self.current_episode = None
            return
        
        # Konvertiere Listen zu Tensoren
        observations = np.stack(self.current_episode["observations"], axis=0)  # (T, H, W, C)
        states = np.stack(self.current_episode["states"], axis=0)  # (T, state_dim)
        actions = np.stack(self.current_episode["actions"], axis=0)  # (T, action_dim)
        
        # Speichere in akkumulierten Listen
        self.all_states.append(torch.from_numpy(states))
        self.all_actions.append(torch.from_numpy(actions))
        self.all_episode_lengths.append(episode_length)
        
        if save_immediately:
            self._save_episode_folder(
                episode_id=self.current_episode["id"],
                observations=observations,
                h5_data=self.current_episode["h5_data"],
                property_params=self.current_episode["property_params"]
            )
        
        log.info(f"Episode {self.episode_count} beendet: {episode_length} Timesteps")
        
        self.episode_count += 1
        self.current_episode = None
    
    def _save_episode_folder(
        self,
        episode_id: int,
        observations: np.ndarray,
        h5_data: List[Dict],
        property_params: Dict
    ):
        """Speichert die Daten eines einzelnen Rollouts."""
        
        # Erstelle Episode-Ordner (6-stellig, zero-padded)
        episode_path = self.dataset_path / f"{episode_id:06d}"
        episode_path.mkdir(exist_ok=True)
        
        # 1. obses.pth - Bilder als PyTorch Tensor
        obses_tensor = torch.from_numpy(observations)  # (T, H, W, C) uint8
        torch.save(obses_tensor, episode_path / "obses.pth")
        
        # 2. PNG Bilder speichern (optional)
        if self.save_png:
            png_folder = episode_path / "images"
            png_folder.mkdir(exist_ok=True)
            
            for t in range(len(observations)):
                img = observations[t]  # (H, W, C) uint8
                png_path = png_folder / f"frame_{t:04d}.png"
                
                if HAS_PIL:
                    # Schneller mit PIL
                    Image.fromarray(img).save(png_path)
                else:
                    # Fallback mit matplotlib
                    try:
                        import matplotlib.pyplot as plt
                        plt.imsave(str(png_path), img)
                    except Exception as e:
                        log.warning(f"PNG-Speicherung fehlgeschlagen: {e}")
            
            log.info(f"  {len(observations)} PNG-Bilder gespeichert in {png_folder}")
        
        # 3. property_params.pkl
        with open(episode_path / "property_params.pkl", "wb") as f:
            pickle.dump(property_params, f)
        
        # 4. H5 Dateien für jeden Timestep (optional, für Kompatibilität)
        if HAS_H5PY:
            for t, h5_entry in enumerate(h5_data):
                h5_path = episode_path / f"{t:02d}.h5"
                with h5py.File(h5_path, "w") as f:
                    for key, value in h5_entry.items():
                        if isinstance(value, np.ndarray):
                            f.create_dataset(key, data=value)
                        else:
                            f.attrs[key] = value
    
    def save_dataset(self):
        """
        Speichert den gesamten Datensatz (states.pth, actions.pth, cameras/).
        Sollte am Ende der Datensammlung aufgerufen werden.
        """
        if len(self.all_states) == 0:
            log.warning("Keine Daten zum Speichern!")
            return
        
        # Finde maximale Länge für Padding
        max_len = max(self.all_episode_lengths)
        state_dim = self.all_states[0].shape[-1]
        action_dim = self.all_actions[0].shape[-1]
        n_rollouts = len(self.all_states)
        
        log.info(f"Speichere Datensatz: {n_rollouts} Rollouts, max_len={max_len}")
        log.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        
        # Erstelle gepaddte Tensoren
        states_padded = torch.zeros((n_rollouts, max_len, state_dim), dtype=torch.float32)
        actions_padded = torch.zeros((n_rollouts, max_len, action_dim), dtype=torch.float32)
        
        for i, (states, actions, length) in enumerate(
            zip(self.all_states, self.all_actions, self.all_episode_lengths)
        ):
            states_padded[i, :length] = states
            actions_padded[i, :length] = actions
        
        # Speichere Hauptdateien
        torch.save(states_padded, self.dataset_path / "states.pth")
        torch.save(actions_padded, self.dataset_path / "actions.pth")
        
        # Speichere Kamera-Kalibrierung (falls vorhanden)
        if self.camera_intrinsic is not None:
            np.save(self.dataset_path / "cameras" / "intrinsic.npy", self.camera_intrinsic)
        if self.camera_extrinsic is not None:
            np.save(self.dataset_path / "cameras" / "extrinsic.npy", self.camera_extrinsic)
        
        # Speichere Metadaten
        metadata = {
            "n_rollouts": n_rollouts,
            "max_timesteps": max_len,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "episode_lengths": self.all_episode_lengths,
            "image_size": self.image_size,
            "created": datetime.now().isoformat(),
        }
        with open(self.dataset_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        log.info(f"Datensatz gespeichert: {self.dataset_path}")
        log.info(f"  states.pth: {states_padded.shape}")
        log.info(f"  actions.pth: {actions_padded.shape}")


def get_franka_state(franka, task) -> np.ndarray:
    """
    Extrahiert den vollständigen Zustandsvektor vom Franka Roboter.
    
    Returns:
        state: np.ndarray mit Shape (state_dim,)
            - [0:3]: End-Effector Position (x, y, z)
            - [3:7]: End-Effector Orientierung (Quaternion w, x, y, z)
            - [7]: Gripper-Öffnung (0-1)
            - [8:15]: Joint-Positionen (7 DOF)
            - [15:22]: Joint-Velocities (7 DOF)
    """
    try:
        # End-Effector Pose
        ee_pos, ee_quat = franka.end_effector.get_world_pose()
        ee_pos = np.atleast_1d(ee_pos).flatten()[:3]
        ee_quat = np.atleast_1d(ee_quat).flatten()[:4]
        
        # Gripper State
        gripper_pos = franka.gripper.get_joint_positions()
        if gripper_pos is not None:
            gripper_pos = np.atleast_1d(gripper_pos).flatten()
            gripper_opening = float(np.mean(gripper_pos))
        else:
            gripper_opening = 0.0
        
        # Joint States
        joint_positions = franka.get_joint_positions()
        joint_positions = np.atleast_1d(joint_positions).flatten()[:7]
        
        joint_velocities = franka.get_joint_velocities()
        joint_velocities = np.atleast_1d(joint_velocities).flatten()[:7]
        
        # Padding falls nötig
        if len(ee_pos) < 3:
            ee_pos = np.pad(ee_pos, (0, 3 - len(ee_pos)))
        if len(ee_quat) < 4:
            ee_quat = np.pad(ee_quat, (0, 4 - len(ee_quat)))
        if len(joint_positions) < 7:
            joint_positions = np.pad(joint_positions, (0, 7 - len(joint_positions)))
        if len(joint_velocities) < 7:
            joint_velocities = np.pad(joint_velocities, (0, 7 - len(joint_velocities)))
        
        # Kombiniere zu einem Vektor
        state = np.concatenate([
            ee_pos,                    # 3
            ee_quat,                   # 4
            [gripper_opening],         # 1
            joint_positions,           # 7
            joint_velocities,          # 7
        ]).astype(np.float32)
        
        return state  # Total: 22 Dimensionen
        
    except Exception as e:
        # Fallback: Leerer State-Vektor
        log.warning(f"get_franka_state Fehler: {e}")
        return np.zeros(22, dtype=np.float32)


def get_franka_action(controller_action) -> np.ndarray:
    """
    Extrahiert den Aktionsvektor aus der Controller-Aktion.
    
    Args:
        controller_action: ArticulationAction vom Controller
    
    Returns:
        action: np.ndarray mit Shape (action_dim,)
            - [0:7]: Joint-Positions oder Velocities
            - [7:9]: Gripper-Befehle
    """
    if hasattr(controller_action, 'joint_positions') and controller_action.joint_positions is not None:
        joint_cmd = controller_action.joint_positions[:7]
    elif hasattr(controller_action, 'joint_velocities') and controller_action.joint_velocities is not None:
        joint_cmd = controller_action.joint_velocities[:7]
    else:
        joint_cmd = np.zeros(7)
    
    # Gripper
    if hasattr(controller_action, 'joint_positions') and controller_action.joint_positions is not None:
        gripper_cmd = controller_action.joint_positions[7:9] if len(controller_action.joint_positions) > 7 else np.zeros(2)
    else:
        gripper_cmd = np.zeros(2)
    
    action = np.concatenate([joint_cmd, gripper_cmd]).astype(np.float32)
    return action  # Total: 9 Dimensionen


# ============================================================================
# Beispiel-Integration in dein Environment
# ============================================================================

EXAMPLE_INTEGRATION = """
# In franka_cube_stack_reworked.py:

from data_logger import FrankaDataLogger, get_franka_state, get_franka_action

def main():
    # ... bestehender Setup-Code ...
    
    # Data Logger initialisieren
    logger = FrankaDataLogger(
        save_path="./dataset",
        object_name="franka_cube_stack",
        image_size=CAM_RESOLUTION,  # (256, 256)
        max_timesteps=500,
    )
    
    # Optional: Kamera-Kalibrierung setzen
    # logger.set_camera_calibration(intrinsic_matrix, extrinsic_matrix)
    
    episode_num = 0
    
    while simulation_app.is_running():
        # Starte neue Episode
        logger.start_episode(episode_num)
        
        # Optional: Episode-Parameter speichern
        logger.set_episode_params({
            "seed": seed,
            "cube_positions": cube_positions,
            "target_position": target_position,
        })
        
        while not controller.is_done():
            simulation_app.update()
            env.world.step()
            
            # Daten sammeln
            obs = env.task.get_observations()
            action = controller.forward(observations=obs)
            
            # RGB Bild von Kamera holen
            rgba = camera.get_rgba()
            rgb = rgba[:, :, :3]  # Nur RGB, ohne Alpha
            
            # State extrahieren
            state = get_franka_state(env.franka, env.task)
            
            # Action extrahieren
            action_vec = get_franka_action(action)
            
            # Optional: Depth-Bild
            # depth = camera.get_depth()
            
            # Loggen
            logger.log_step(
                rgb_image=rgb,
                state=state,
                action=action_vec,
                # depth_image=depth,  # Optional
                # additional_data={"obs": obs}  # Optional
            )
            
            articulation.apply_action(action)
        
        # Episode beenden
        logger.end_episode()
        
        # Reset für nächste Episode
        env.world.reset()
        controller.reset()
        episode_num += 1
        
        # Nach X Episoden speichern
        if episode_num % 100 == 0:
            logger.save_dataset()
    
    # Finales Speichern
    logger.save_dataset()
    simulation_app.close()
"""

if __name__ == "__main__":
    print(EXAMPLE_INTEGRATION)


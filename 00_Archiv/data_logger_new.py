"""
Data Logger für Isaac Sim - Franka Panda Datensatz-Generierung
Struktur kompatibel mit dem ROPE/DEFORMABLE Dataset Format.

Ausgabe-Format (Rope kompatibel):
    dataset/
    ├── 000000/               # Episode 0
    │   ├── obses.pth         # (T, H, W, C) float32 - RGB Bilder
    │   ├── property_params.pkl
    │   ├── 00.h5             # Timestep 0
    │   ├── 01.h5             # Timestep 1
    │   └── ...
    ├── 000001/               # Episode 1
    │   ├── obses.pth
    │   ├── property_params.pkl
    │   ├── 00.h5
    │   └── ...
    └── ...

H5-Datei Struktur pro Timestep:
    - action: (action_dim,) float64
    - eef_states: (1, 1, 14) float64 - End-Effector States
    - info/
        - n_cams: int64 - Anzahl Kameras
        - timestamp: int64 - Timestep Index
    - observations/
        - color/
            - cam_0: (1, H, W, 3) float32
        - depth/
            - cam_0: (1, H, W) uint16  (falls vorhanden)
    - positions: (1, N, 4) float32 - Optional: Partikel/Objekt-Positionen

Verwendung:
    1. Importiere DataLogger in dein Environment
    2. Rufe logger.start_episode() am Anfang jeder Episode auf
    3. Rufe logger.log_step(...) bei jedem Timestep auf
    4. Rufe logger.end_episode() am Ende jeder Episode auf
    5. Optional: logger.save_dataset() für zusätzliche Metadaten
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

# h5py für Rope-Format erforderlich
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logging.warning("h5py nicht verfügbar - H5-Dateien werden nicht gespeichert!")

# Für PNG-Speicherung (optional)
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

log = logging.getLogger("DataLogger")


class FrankaDataLogger:
    """
    Data Logger für Franka Panda Robot in Isaac Sim.
    Speichert Daten im ROPE/DEFORMABLE Dataset Format.
    """
    
    def __init__(
        self,
        save_path: str = "./dataset",
        object_name: str = "franka_cube_stack",
        image_size: tuple = (256, 256),  # Passend zu config.yaml
        max_timesteps: Optional[int] = None,
        save_png: bool = False,  # Optional: PNG-Preview
        n_cams: int = 1,  # Anzahl Kameras
    ):
        """
        Args:
            save_path: Basis-Verzeichnis für den Datensatz
            object_name: Name des Datensatzes (Unterordner)
            image_size: (H, W) der Kamerabilder (Standard: 224x224 wie Rope)
            max_timesteps: Maximale Anzahl Timesteps pro Episode (None = unbegrenzt)
            save_png: Wenn True, werden Bilder auch als PNG gespeichert
            n_cams: Anzahl der Kameras (für H5-Struktur)
        """
        self.save_path = Path(save_path)
        self.object_name = object_name
        self.dataset_path = self.save_path / object_name
        self.image_size = image_size
        self.max_timesteps = max_timesteps
        self.save_png = save_png
        self.n_cams = n_cams
        
        # Temporärer Speicher für aktuelle Episode
        self.current_episode = None
        self.episode_count = 0
        
        # Statistiken
        self.all_episode_lengths: List[int] = []
        
        # Erstelle Basis-Verzeichnis
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        if not HAS_H5PY:
            raise ImportError("h5py ist erforderlich für das Rope-Format!")
        
        log.info(f"DataLogger initialisiert (Rope-Format): {self.dataset_path}")
    
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
            "observations": [],      # RGB Bilder (T, H, W, C)
            "depth_images": [],      # Tiefenbilder (T, H, W) - optional
            "actions": [],           # (T, action_dim)
            "eef_states": [],        # (T, 14) End-Effector States
            "positions": [],         # (T, N, 4) Objekt-Positionen - optional
            "property_params": {}
        }
        
        log.info(f"Episode {self.episode_count} gestartet")
    
    def log_step(
        self,
        rgb_image: np.ndarray,
        action: np.ndarray,
        eef_state: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,  # Alias für eef_state (Rückwärtskompatibilität)
        depth_image: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Loggt einen einzelnen Timestep.
        
        Args:
            rgb_image: RGB Bild (H, W, 3), Werte 0-255 oder 0-1
            action: Aktionsvektor (action_dim,)
            eef_state: End-Effector State (14,) - [pos(3), quat(4), vel(3), ang_vel(4)]
            state: Alias für eef_state (Rückwärtskompatibilität)
            depth_image: Optional Tiefenbild (H, W)
            positions: Optional Objekt-Positionen (N, 4) - xyz + radius
            additional_data: Optional zusätzliche Daten
        """
        if self.current_episode is None:
            raise RuntimeError("Keine Episode gestartet! Rufe start_episode() auf.")
        
        # Unterstütze beide Parameter-Namen
        if eef_state is None and state is not None:
            eef_state = state
        elif eef_state is None and state is None:
            # Fallback: Leerer State
            eef_state = np.zeros(14, dtype=np.float64)
        
        # Prüfe max_timesteps
        if self.max_timesteps is not None and self.current_episode["timestep"] >= self.max_timesteps:
            log.warning(f"Max timesteps ({self.max_timesteps}) erreicht, überspringe...")
            return
        
        # RGB-Bild normalisieren
        if rgb_image.max() > 1.0:
            # uint8 -> float32 [0, 255]
            rgb_float = rgb_image.astype(np.float32)
        else:
            # [0, 1] -> [0, 255]
            rgb_float = (rgb_image * 255.0).astype(np.float32)
        
        self.current_episode["observations"].append(rgb_float)
        self.current_episode["actions"].append(action.astype(np.float64))
        self.current_episode["eef_states"].append(eef_state.astype(np.float64))
        
        if depth_image is not None:
            self.current_episode["depth_images"].append(depth_image.astype(np.uint16))
        
        if positions is not None:
            self.current_episode["positions"].append(positions.astype(np.float32))
        
        self.current_episode["timestep"] += 1
    
    def set_episode_params(self, params: Dict[str, Any]):
        """
        Setzt zusätzliche Parameter für die aktuelle Episode.
        Wird als property_params.pkl gespeichert.
        """
        if self.current_episode is None:
            raise RuntimeError("Keine Episode gestartet!")
        self.current_episode["property_params"].update(params)
    
    def discard_episode(self):
        """
        Verwirft die aktuelle Episode ohne sie zu speichern.
        """
        if self.current_episode is None:
            log.warning("Keine Episode zum Verwerfen vorhanden")
            return
        
        episode_id = self.current_episode["id"]
        timesteps = self.current_episode["timestep"]
        log.warning(f"Episode {episode_id} verworfen ({timesteps} Timesteps)")
        
        self.current_episode = None
    
    def end_episode(self):
        """
        Beendet die aktuelle Episode und speichert die Daten im Rope-Format.
        """
        if self.current_episode is None:
            log.warning("Keine Episode zum Beenden vorhanden")
            return
        
        episode_length = self.current_episode["timestep"]
        if episode_length == 0:
            log.warning("Leere Episode, überspringe...")
            self.current_episode = None
            return
        
        episode_id = self.current_episode["id"]
        
        # Erstelle Episode-Ordner: 000000/, 000001/, ...
        episode_dir = self.dataset_path / f"{episode_id:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. obses.pth speichern: (T, H, W, C) float32
        observations = np.stack(self.current_episode["observations"], axis=0)
        obses_tensor = torch.from_numpy(observations).float()  # (T, H, W, C)
        torch.save(obses_tensor, episode_dir / "obses.pth")
        
        # 2. property_params.pkl speichern
        if self.current_episode["property_params"]:
            with open(episode_dir / "property_params.pkl", "wb") as f:
                pickle.dump(self.current_episode["property_params"], f)
        
        # 3. H5-Dateien für jeden Timestep: 00.h5, 01.h5, ...
        actions = self.current_episode["actions"]
        eef_states = self.current_episode["eef_states"]
        depth_images = self.current_episode["depth_images"]
        positions = self.current_episode["positions"]
        
        for t in range(episode_length):
            h5_path = episode_dir / f"{t:02d}.h5"
            
            with h5py.File(h5_path, "w") as f:
                # action: (action_dim,) float64
                f.create_dataset("action", data=actions[t])
                
                # eef_states: (1, 1, 14) float64
                eef = eef_states[t].reshape(1, 1, -1)
                # Padding auf 14 Dimensionen falls nötig
                if eef.shape[-1] < 14:
                    eef_padded = np.zeros((1, 1, 14), dtype=np.float64)
                    eef_padded[..., :eef.shape[-1]] = eef
                    eef = eef_padded
                f.create_dataset("eef_states", data=eef)
                
                # info group
                info = f.create_group("info")
                info.create_dataset("n_cams", data=self.n_cams)
                info.create_dataset("timestamp", data=t)
        
                # observations group
                obs_group = f.create_group("observations")
                color_group = obs_group.create_group("color")
                
                # color/cam_0: (1, H, W, 3) float32
                img = observations[t].reshape(1, *observations[t].shape)
                color_group.create_dataset("cam_0", data=img.astype(np.float32))
                
                # depth (optional)
                if depth_images:
                    depth_group = obs_group.create_group("depth")
                    depth = depth_images[t].reshape(1, *depth_images[t].shape)
                    depth_group.create_dataset("cam_0", data=depth.astype(np.uint16))
        
                # positions (optional): (1, N, 4) float32
                if positions:
                    pos = positions[t].reshape(1, -1, 4)
                    f.create_dataset("positions", data=pos.astype(np.float32))
        
        # Optional: PNG-Preview
        if self.save_png:
            png_dir = episode_dir / "png_preview"
            png_dir.mkdir(exist_ok=True)

            frame_step = max(1, episode_length // 10)  # Max 10 Bilder pro Episode
            for t in range(0, episode_length, frame_step):
                img = observations[t].astype(np.uint8)
                if HAS_PIL:
                    Image.fromarray(img).save(png_dir / f"frame_{t:04d}.png")
        
        log.info(f"Episode {episode_id} gespeichert: {episode_length} Timesteps, {episode_length} H5-Dateien")
        
        self.all_episode_lengths.append(episode_length)
        self.episode_count += 1
        self.current_episode = None
    
    def save_dataset(self):
        """
        Optional: Speichert Metadaten über den gesamten Datensatz.
        Im Rope-Format nicht zwingend erforderlich, aber nützlich.
        """
        if len(self.all_episode_lengths) == 0:
            log.warning("Keine Episoden zum Zusammenfassen!")
            return
        
        metadata = {
            "n_episodes": len(self.all_episode_lengths),
            "episode_lengths": self.all_episode_lengths,
            "total_timesteps": sum(self.all_episode_lengths),
            "image_size": self.image_size,
            "n_cams": self.n_cams,
            "created": datetime.now().isoformat(),
            "format": "rope_compatible",
        }
        
        with open(self.dataset_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        
        log.info(f"Datensatz-Metadaten gespeichert:")
        log.info(f"  Episoden: {metadata['n_episodes']}")
        log.info(f"  Gesamt-Timesteps: {metadata['total_timesteps']}")


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
        log.warning(f"get_franka_state Fehler: {e}")
        return np.zeros(22, dtype=np.float32)


def get_franka_eef_state(franka) -> np.ndarray:
    """
    Extrahiert den End-Effector State für das Rope-Format.
    
    Returns:
        eef_state: np.ndarray (14,)
            - [0:3]: Position (x, y, z)
            - [3:7]: Orientierung (Quaternion)
            - [7:10]: Lineare Velocity
            - [10:14]: Angular Velocity (Quaternion-Form oder Axis-Angle)
    """
    try:
        # Position und Orientierung
        ee_pos, ee_quat = franka.end_effector.get_world_pose()
        ee_pos = np.atleast_1d(ee_pos).flatten()[:3]
        ee_quat = np.atleast_1d(ee_quat).flatten()[:4]
        
        # Velocities (falls verfügbar)
        try:
            ee_vel = franka.end_effector.get_linear_velocity()
            ee_vel = np.atleast_1d(ee_vel).flatten()[:3]
        except:
            ee_vel = np.zeros(3)
        
        try:
            ee_ang_vel = franka.end_effector.get_angular_velocity()
            ee_ang_vel = np.atleast_1d(ee_ang_vel).flatten()[:3]
            # Pad to 4 dimensions
            ee_ang_vel = np.pad(ee_ang_vel, (0, 1))
        except:
            ee_ang_vel = np.zeros(4)
        
        # Padding falls nötig
        if len(ee_pos) < 3:
            ee_pos = np.pad(ee_pos, (0, 3 - len(ee_pos)))
        if len(ee_quat) < 4:
            ee_quat = np.pad(ee_quat, (0, 4 - len(ee_quat)))
        if len(ee_vel) < 3:
            ee_vel = np.pad(ee_vel, (0, 3 - len(ee_vel)))
        if len(ee_ang_vel) < 4:
            ee_ang_vel = np.pad(ee_ang_vel, (0, 4 - len(ee_ang_vel)))
        
        eef_state = np.concatenate([
            ee_pos,       # 3
            ee_quat,      # 4
            ee_vel,       # 3
            ee_ang_vel,   # 4
        ]).astype(np.float64)
        
        return eef_state  # Total: 14 Dimensionen
        
    except Exception as e:
        log.warning(f"get_franka_eef_state Fehler: {e}")
        return np.zeros(14, dtype=np.float64)


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

"""
Data Logger für Isaac Sim - Franka Panda + 2 Würfel Datensatz-Generierung
Struktur kompatibel mit dem ROPE/DEFORMABLE Format.

Ausgabe-Format (Rope-kompatibel):
    dataset/
    ├── cameras/
    │   ├── intrinsic.npy      # (4, 4) float64 - Intrinsische Parameter
    │   └── extrinsic.npy      # (4, 4) float64 - Extrinsische Parameter (1 Kamera)
    ├── 000001/                 # Episode 1 (6-stellig, 0-padded)
    │   ├── obses.pth          # (T, H, W, C) uint8 - Alle Beobachtungsbilder
    │   ├── 00.h5              # HDF5 - Timestep 0 Daten
    │   ├── 01.h5              # HDF5 - Timestep 1 Daten
    │   └── ...
    └── ...

Verwendung:
    1. Importiere DataLogger in dein Environment
    2. Lade Config mit load_config_from_yaml()
    3. Rufe logger.start_episode() am Anfang jeder Episode auf
    4. Rufe logger.log_step(...) bei jedem Timestep auf
    5. Rufe logger.end_episode() am Ende jeder Episode auf
"""

import torch
import numpy as np
import pickle
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import logging

# Für PNG-Speicherung
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL nicht verfügbar - versuche matplotlib für PNG-Speicherung")

# h5py ist erforderlich für Rope-Format
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    raise ImportError("h5py ist erforderlich für Rope-Format! Installiere mit: pip install h5py")

log = logging.getLogger("DataLogger")


def load_config_from_yaml(config_path: Optional[str] = None) -> dict:
    """
    Lädt Konfiguration aus YAML-Datei.
    
    Args:
        config_path: Pfad zur config.yaml (None = im gleichen Verzeichnis suchen)
    
    Returns:
        config: Dictionary mit Konfiguration
    """
    if config_path is None:
        # Standard: config.yaml im gleichen Verzeichnis wie dieses Skript
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


class FrankaDataLogger:
    """
    Data Logger für Franka Panda Robot + 2 Würfel in Isaac Sim.
    Speichert Daten im Format kompatibel mit Rope/D deformable Dataset.
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
        action_mode: str = "controller",  # "controller" oder "ee_velocity"
    ):
        """
        Args:
            config: Dictionary mit Konfiguration (wird aus config_path geladen falls None)
            config_path: Pfad zur config.yaml (wird verwendet falls config=None)
            action_mode: "controller" = Controller-Action, "ee_velocity" = EE-Position+Velocity
        """
        # Lade Config
        if config is None:
            config = load_config_from_yaml(config_path)
        
        self.config = config
        
        # Extrahiere Konfiguration
        self.save_path = Path(config["dataset"]["path"])
        self.object_name = config["dataset"]["name"]
        self.dataset_path = self.save_path / self.object_name
        self.image_size = tuple(config["camera"]["resolution"])  # (H, W)
        self.save_png = config["dataset"].get("save_png", True)
        self.n_cubes = config["cubes"]["count"]
        self.action_mode = action_mode
        
        # Kamera-Parameter aus Config
        self.camera_position = np.array(config["camera"]["position"], dtype=np.float64)
        self.camera_euler = np.array(config["camera"]["euler"], dtype=np.float64)
        
        # Temporärer Speicher für aktuelle Episode
        self.current_episode = None
        self.episode_count = 0
        
        # Kamera-Kalibrierung
        self.camera_intrinsic: Optional[np.ndarray] = None
        self.camera_extrinsic: Optional[np.ndarray] = None
        
        # Vorheriger EE-State für Velocity-Berechnung
        self.prev_ee_pos: Optional[np.ndarray] = None
        self.prev_ee_quat: Optional[np.ndarray] = None
        
        # Erstelle Verzeichnisstruktur
        self._setup_directories()
        
        # Prüfe verfügbaren Speicherplatz
        self._check_disk_space()
        
        log.info(f"DataLogger initialisiert: {self.dataset_path}")
        log.info(f"  Action Mode: {self.action_mode}")
        log.info(f"  Image Size: {self.image_size}")
        log.info(f"  Number of Cubes: {self.n_cubes}")
    
    def _setup_directories(self):
        """Erstellt die Verzeichnisstruktur."""
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "cameras").mkdir(exist_ok=True)
    
    def _check_disk_space(self):
        """Prüft verfügbaren Speicherplatz und warnt bei wenig Platz."""
        try:
            total, used, free = shutil.disk_usage(self.dataset_path)
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            free_percent = (free / total) * 100
            
            log.info(f"Speicherplatz: {free_gb:.2f} GB frei von {total_gb:.2f} GB ({free_percent:.1f}%)")
            
            if free_gb < 1.0:
                log.error(f"⚠️  WARNUNG: Weniger als 1 GB Speicherplatz frei! ({free_gb:.2f} GB)")
                log.error("   Datensammlung könnte fehlschlagen!")
            elif free_gb < 5.0:
                log.warning(f"⚠️  WARNUNG: Weniger als 5 GB Speicherplatz frei! ({free_gb:.2f} GB)")
        except Exception as e:
            log.warning(f"Konnte Speicherplatz nicht prüfen: {e}")
    
    def set_camera_calibration(
        self, 
        intrinsic: np.ndarray, 
        extrinsic: np.ndarray
    ):
        """
        Setzt die Kamera-Kalibrierungsmatrizen.
        
        Args:
            intrinsic: 3x3 oder 4x4 Kamera-Intrinsik-Matrix
            extrinsic: 4x4 Kamera-Extrinsik-Matrix (World -> Camera)
        """
        # Konvertiere zu 4x4 falls nötig
        if intrinsic.shape == (3, 3):
            intrinsic_4x4 = np.eye(4, dtype=np.float64)
            intrinsic_4x4[:3, :3] = intrinsic
            self.camera_intrinsic = intrinsic_4x4
        else:
            self.camera_intrinsic = intrinsic.astype(np.float64)
        
        # Extrinsic: (4, 4) für 1 Kamera
        if extrinsic.ndim == 2:
            # Einzelne Kamera: (4, 4) -> (4, 4) (bleibt gleich)
            self.camera_extrinsic = extrinsic.astype(np.float64)
        else:
            # Mehrere Kameras: (N, 4, 4) -> nur erste nehmen
            self.camera_extrinsic = extrinsic[0].astype(np.float64)
    
    def start_episode(self, episode_id: Optional[int] = None):
        """
        Startet eine neue Episode für das Logging.
        
        Args:
            episode_id: Optional: Explizite Episode-ID, sonst auto-increment
        """
        if episode_id is not None:
            self.episode_count = episode_id
        
        # Erstelle Episode-Ordner
        episode_folder = self.dataset_path / f"{self.episode_count:06d}"
        episode_folder.mkdir(exist_ok=True)
        
        self.current_episode = {
            "id": self.episode_count,
            "folder": episode_folder,
            "timestep": 0,
            "observations": [],  # RGB Bilder (T, H, W, C)
            "h5_files": [],        # Pfade zu H5-Dateien
        }
        
        # Reset für Velocity-Berechnung
        self.prev_ee_pos = None
        self.prev_ee_quat = None
        
        log.info(f"Episode {self.episode_count} gestartet")
    
    def log_step(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        controller_action: Optional[Any] = None,
        cube_positions: List[Tuple[float, float, float, float]] = None,  # [(x, y, z, yaw), ...]
    ):
        """
        Loggt einen einzelnen Timestep.
        
        Args:
            rgb_image: RGB Bild (H, W, 3) uint8, Werte 0-255
            depth_image: Tiefenbild (H, W) float32 oder uint16
            ee_pos: Endeffektor Position (3,) float32
            ee_quat: Endeffektor Orientierung (4,) float32 [w, x, y, z]
            controller_action: Optional Controller-Action (für action_mode="controller")
            cube_positions: Liste von (x, y, z, yaw) für jeden Würfel
        """
        if self.current_episode is None:
            raise RuntimeError("Keine Episode gestartet! Rufe start_episode() auf.")
        
        timestep = self.current_episode["timestep"]
        
        # Bild speichern
        if rgb_image.shape[:2] != self.image_size:
            log.warning(f"Bildgröße {rgb_image.shape[:2]} != {self.image_size}, resizing...")
            from PIL import Image
            img = Image.fromarray(rgb_image)
            img = img.resize((self.image_size[1], self.image_size[0]))  # (W, H)
            rgb_image = np.array(img)
        
        self.current_episode["observations"].append(rgb_image.astype(np.uint8))
        
        # Berechne Action basierend auf Mode
        if self.action_mode == "controller":
            if controller_action is None:
                log.warning("controller_action ist None, verwende Null-Action")
                action = np.zeros(4, dtype=np.float64)
            else:
                # Extrahiere Action aus Controller
                action = self._extract_controller_action(controller_action)
        elif self.action_mode == "ee_velocity":
            # Berechne EE-Velocity aus Position
            if self.prev_ee_pos is not None:
                # Velocity = Delta Position (vereinfacht, sollte eigentlich durch dt geteilt werden)
                ee_velocity = ee_pos - self.prev_ee_pos
            else:
                ee_velocity = np.zeros(3, dtype=np.float64)
            
            # Action = [x, y, z, velocity_magnitude]
            action = np.concatenate([
                ee_pos.astype(np.float64),
                [np.linalg.norm(ee_velocity)]
            ])
        else:
            raise ValueError(f"Unbekannter action_mode: {self.action_mode}")
        
        # Berechne EEF States (1, 1, 14) Format
        # Format: [[[x, y, z, x, y, z, qw, qx, qy, qz, qw, qx, qy, qz]]]]
        eef_states = np.array([[[
            ee_pos[0], ee_pos[1], ee_pos[2],  # Position 1
            ee_pos[0], ee_pos[1], ee_pos[2],  # Position 2 (dupliziert)
            ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3],  # Quaternion 1
            ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3],  # Quaternion 2 (dupliziert)
        ]]], dtype=np.float64)
        
        # Cube Positions: (1, n_cubes, 4) - (x, y, z, yaw)
        if cube_positions is None:
            cube_positions = [(0.0, 0.0, 0.0, 0.0)] * self.n_cubes
        
        # Stelle sicher, dass wir genau n_cubes haben
        if len(cube_positions) < self.n_cubes:
            cube_positions = list(cube_positions) + [(0.0, 0.0, 0.0, 0.0)] * (self.n_cubes - len(cube_positions))
        elif len(cube_positions) > self.n_cubes:
            cube_positions = cube_positions[:self.n_cubes]
        
        positions = np.array([cube_positions], dtype=np.float32)  # (1, n_cubes, 4)
        
        # Speichere H5-Datei für diesen Timestep
        h5_path = self.current_episode["folder"] / f"{timestep:02d}.h5"
        self._save_h5_timestep(
            h5_path=h5_path,
            timestep=timestep,
            action=action,
            eef_states=eef_states,
            positions=positions,
            rgb_image=rgb_image,
            depth_image=depth_image,
        )
        
        self.current_episode["h5_files"].append(h5_path)
        
        # Update für nächsten Timestep
        self.prev_ee_pos = ee_pos.copy()
        self.prev_ee_quat = ee_quat.copy()
        self.current_episode["timestep"] += 1
    
    def _extract_controller_action(self, controller_action) -> np.ndarray:
        """
        Extrahiert Action aus Controller-Action.
        
        Returns:
            action: (4,) float64 - [x, y, z, ?] oder ähnlich
        """
        # Versuche verschiedene Attribute
        if hasattr(controller_action, 'joint_positions') and controller_action.joint_positions is not None:
            joint_cmd = np.array(controller_action.joint_positions[:7], dtype=np.float64)
            # Nimm erste 4 Joints als Action
            return joint_cmd[:4]
        elif hasattr(controller_action, 'joint_velocities') and controller_action.joint_velocities is not None:
            joint_cmd = np.array(controller_action.joint_velocities[:7], dtype=np.float64)
            return joint_cmd[:4]
        else:
            log.warning("Konnte Action nicht aus Controller extrahieren, verwende Null-Action")
            return np.zeros(4, dtype=np.float64)
    
    def _save_h5_timestep(
        self,
        h5_path: Path,
        timestep: int,
        action: np.ndarray,
        eef_states: np.ndarray,
        positions: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
    ):
        """
        Speichert einen Timestep als H5-Datei.
        
        Args:
            h5_path: Pfad zur H5-Datei
            timestep: Timestep-Nummer
            action: Action-Vektor (4,) float64
            eef_states: Endeffektor-States (1, 1, 14) float64
            positions: Cube-Positionen (1, n_cubes, 4) float32
            rgb_image: RGB-Bild (H, W, 3) uint8
            depth_image: Tiefenbild (H, W) float32 oder uint16
        """
        if not HAS_H5PY:
            raise RuntimeError("h5py nicht verfügbar!")
        
        try:
            with h5py.File(h5_path, "w") as f:
                # Action
                f.create_dataset("action", data=action.astype(np.float64), dtype=np.float64)
                
                # EEF States
                f.create_dataset("eef_states", data=eef_states, dtype=np.float64)
                
                # Positions
                f.create_dataset("positions", data=positions, dtype=np.float32)
                
                # Info-Gruppe
                info_group = f.create_group("info")
                info_group.create_dataset("n_cams", data=np.int64(1), dtype=np.int64)
                info_group.create_dataset("n_cubes", data=np.int64(self.n_cubes), dtype=np.int64)  # n_particles im Rope-Format, hier n_cubes
                info_group.create_dataset("timestamp", data=np.int64(timestep + 1), dtype=np.int64)  # +1 für Kompatibilität (Rope startet bei 1)
                
                # Observations-Gruppe
                obs_group = f.create_group("observations")
                
                # Color
                color_group = obs_group.create_group("color")
                # RGB-Bild: (H, W, 3) -> (1, H, W, 3) für Kompatibilität
                # Konvertiere uint8 zu float32, Werte bleiben 0-255
                rgb_float = rgb_image.astype(np.float32)
                rgb_expanded = np.expand_dims(rgb_float, axis=0)
                color_group.create_dataset("cam_0", data=rgb_expanded, dtype=np.float32)
                
                # Depth
                depth_group = obs_group.create_group("depth")
                # Depth-Bild: (H, W) -> (1, H, W) für Kompatibilität
                if depth_image.dtype == np.uint16:
                    depth_expanded = np.expand_dims(depth_image, axis=0)
                else:
                    # Konvertiere float32 zu uint16 (Millimeter)
                    depth_mm = (depth_image * 1000.0).astype(np.uint16)
                    depth_expanded = np.expand_dims(depth_mm, axis=0)
                depth_group.create_dataset("cam_0", data=depth_expanded, dtype=np.uint16)
        except OSError as e:
            if "No space left on device" in str(e):
                log.error("=" * 80)
                log.error("❌ SPEICHERPLATZ VOLL!")
                log.error(f"   Konnte H5-Datei nicht speichern: {h5_path}")
                log.error(f"   Episode: {self.current_episode['id']}")
                log.error(f"   Timestep: {timestep}")
                log.error(f"   Episode wird abgebrochen. Bitte Speicherplatz freigeben.")
                log.error("=" * 80)
                raise RuntimeError("Speicherplatz voll - Episode abgebrochen") from e
            else:
                log.error(f"OSError beim Speichern von {h5_path}:")
                log.error(f"  Error Code: {e.errno}")
                log.error(f"  Error Message: {str(e)}")
                import traceback
                log.error(f"  Traceback:\n{traceback.format_exc()}")
                raise
        except Exception as e:
            log.error(f"Unerwarteter Fehler beim Speichern von {h5_path}:")
            log.error(f"  Exception Type: {type(e).__name__}")
            log.error(f"  Exception Message: {str(e)}")
            log.error(f"  Episode: {self.current_episode['id']}")
            log.error(f"  Timestep: {timestep}")
            import traceback
            log.error(f"  Traceback:\n{traceback.format_exc()}")
            raise
    
    def discard_episode(self):
        """
        Verwirft die aktuelle Episode ohne sie zu speichern.
        """
        if self.current_episode is None:
            log.warning("Keine Episode zum Verwerfen vorhanden")
            return
        
        episode_id = self.current_episode["id"]
        timesteps = self.current_episode["timestep"]
        
        # Lösche Episode-Ordner
        import shutil
        if self.current_episode["folder"].exists():
            shutil.rmtree(self.current_episode["folder"])
        
        log.warning(f"Episode {episode_id} verworfen ({timesteps} Timesteps)")
        
        self.current_episode = None
        # episode_count wird NICHT erhöht
    
    def end_episode(self):
        """
        Beendet die aktuelle Episode und speichert die Daten.
        """
        if self.current_episode is None:
            log.warning("Keine Episode zum Beenden vorhanden")
            return
        
        episode_length = self.current_episode["timestep"]
        episode_id = self.current_episode["id"]
        
        if episode_length == 0:
            log.warning(f"Episode {episode_id}: Leere Episode, überspringe...")
            self.current_episode = None
            return
        
        try:
            # Speichere obses.pth
            log.debug(f"Episode {episode_id}: Speichere obses.pth ({episode_length} Timesteps)...")
            observations = np.stack(self.current_episode["observations"], axis=0)  # (T, H, W, C)
            obses_tensor = torch.from_numpy(observations)  # (T, H, W, C) uint8
            obses_path = self.current_episode["folder"] / "obses.pth"
            torch.save(obses_tensor, obses_path)
            log.debug(f"Episode {episode_id}: obses.pth gespeichert: {obses_tensor.shape}")
            
            log.info(f"Episode {episode_id} beendet: {episode_length} Timesteps")
            log.info(f"  obses.pth: {obses_tensor.shape}")
            log.info(f"  H5-Dateien: {episode_length} Dateien (00.h5 bis {episode_length-1:02d}.h5)")
            log.info(f"  Speicherort: {self.current_episode['folder']}")
        except Exception as e:
            log.error(f"Episode {episode_id}: Fehler beim Beenden der Episode!")
            log.error(f"  Exception Type: {type(e).__name__}")
            log.error(f"  Exception Message: {str(e)}")
            log.error(f"  Episode Length: {episode_length}")
            import traceback
            log.error(f"  Traceback:\n{traceback.format_exc()}")
            raise
        
        self.episode_count += 1
        self.current_episode = None
    
    def save_camera_calibration(self):
        """
        Speichert die Kamera-Kalibrierung.
        """
        if self.camera_intrinsic is None or self.camera_extrinsic is None:
            log.warning("Kamera-Kalibrierung nicht gesetzt, überspringe Speicherung")
            return
        
        # Rope-Format: intrinsic (4, 4), extrinsic (4, 4) für 1 Kamera
        # Aber Format erwartet (4, 4) für intrinsic und (4, 4, 4) für extrinsic (4 Kameras)
        # Wir speichern für 1 Kamera, also (4, 4) für beide
        
        intrinsic_path = self.dataset_path / "cameras" / "intrinsic.npy"
        extrinsic_path = self.dataset_path / "cameras" / "extrinsic.npy"
        
        # Intrinsic: (4, 4) - bleibt gleich
        np.save(intrinsic_path, self.camera_intrinsic)
        
        # Extrinsic: (4, 4) -> erweitere zu (1, 4, 4) für Kompatibilität
        # Aber Rope-Format erwartet (4, 4, 4) für 4 Kameras
        # Wir haben nur 1 Kamera, also (1, 4, 4) -> aber Format erwartet (4, 4, 4)
        # Lösung: Wiederhole die Matrix 4x für Kompatibilität
        extrinsic_expanded = np.stack([self.camera_extrinsic] * 4, axis=0)  # (4, 4, 4)
        np.save(extrinsic_path, extrinsic_expanded)
        
        log.info(f"Kamera-Kalibrierung gespeichert:")
        log.info(f"  intrinsic: {self.camera_intrinsic.shape}")
        log.info(f"  extrinsic: {extrinsic_expanded.shape}")


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

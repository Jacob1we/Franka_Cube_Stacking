"""
Minimaler Data Logger für Isaac Sim - Franka Panda + Würfel
Kompatibel mit dem DINO WM Rope/Deformable Format (data.py).

Action-Format: "ee_pos" (6D)
    action = [x_start, y_start, z_start, x_end, y_end, z_end]

Ausgabe-Format (wie deformable/rope Datensatz):
    dataset/
    ├── cameras/
    │   ├── intrinsic.npy      # (4, 4) float64
    │   └── extrinsic.npy      # (4, 4, 4) float64
    └── 000000/                 # Episode 0
        ├── obses.pth          # (T, H, W, 3) float32 - RGB Bilder (Werte 0-255)
        ├── 000.h5             # HDF5 - Timestep 0
        ├── 001.h5             # HDF5 - Timestep 1
        ├── ...                # HDF5 - Weitere Timesteps
        │   ├── action         # (6,) float64 - [prev_ee_pos, current_ee_pos]
        │   ├── eef_states     # (1, 14) float64
        │   ├── positions      # (1, N, 3) float32
        │   ├── info/          # n_cams, n_particles, timestamp
        │   └── observations/  # color/cam_0 (1,H,W,3), depth/cam_0 (1,H,W)
        ├── first.png          # Erstes Frame
        └── last.png           # Letztes Frame
"""

import numpy as np
import h5py
import yaml
import torch
from pathlib import Path
from typing import Optional, List, Tuple
import logging

log = logging.getLogger("MinDataLogger")


def load_config(config_path: Optional[str] = None) -> dict:
    """Lädt Konfiguration aus YAML-Datei."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_imgs(imgs_list: np.ndarray):
    """
    Verarbeitet Bilder wie in data.py.
    
    Args:
        imgs_list: (T, num_cameras, H, W, 5) - RGB(3) + pad(1) + Depth(1)
    
    Returns:
        color_imgs: dict mit cam_X -> (T, H, W, 3)
        depth_imgs: dict mit cam_X -> (T, H, W) uint16
    """
    T, n_cam, H, W, _ = imgs_list.shape
    color_imgs = {}
    depth_imgs = {}

    for cam_idx in range(n_cam):
        img = imgs_list[:, cam_idx]  # (T, H, W, 5)
        color_imgs[f"cam_{cam_idx}"] = img[:, :, :, :3][..., ::-1]  # BGR->RGB
        depth_imgs[f"cam_{cam_idx}"] = (img[:, :, :, -1] * 1000).astype(np.uint16)

    return color_imgs, depth_imgs


def save_h5(filename: Path, data: dict):
    """Speichert Daten als H5 wie in data.py."""
    with h5py.File(filename, "w") as f:
        for key, value in data.items():
            if key == "observations":
                for sub_key, sub_value in value.items():
                    for subsub_key, subsub_value in sub_value.items():
                        f.create_dataset(f"{key}/{sub_key}/{subsub_key}", data=subsub_value)
            elif key == "info":
                for sub_key, sub_value in value.items():
                    f.create_dataset(f"{key}/{sub_key}", data=sub_value)
            else:
                f.create_dataset(key, data=value)


class MinDataLogger:
    """Minimaler Data Logger für Franka + Würfel im data.py Format."""
    
    def __init__(
        self, 
        config: Optional[dict] = None, 
        config_path: Optional[str] = None,
        action_mode: str = "ee_pos",  # Ignoriert, nur ee_pos unterstützt
        dt: float = 1.0 / 60.0,       # Ignoriert, nicht benötigt
    ):
        if config is None:
            config = load_config(config_path)
        
        self.config = config
        self.object_name = config["dataset"]["name"]  # Für Kompatibilität mit fcs_main_parallel
        self.dataset_path = Path(config["dataset"]["path"]) / self.object_name
        self.image_size = tuple(config["camera"]["resolution"])
        self.save_png = config["dataset"].get("save_png", True)
        self.n_cubes = config["cubes"]["count"]
        
        self.camera_intrinsic: Optional[np.ndarray] = None
        self.camera_extrinsic: Optional[np.ndarray] = None
        
        self.current_episode = None
        self.episode_count = 0
        
        self._setup_directories()
        log.info(f"MinDataLogger initialisiert: {self.dataset_path}")
        log.info(f"  Action Mode: ee_pos (6D: start + end Position)")
    
    def _setup_directories(self):
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        (self.dataset_path / "cameras").mkdir(exist_ok=True)
    
    def set_camera_calibration(self, intrinsic: np.ndarray, extrinsic: np.ndarray):
        """Setzt Kamera-Kalibrierung."""
        if intrinsic.shape == (3, 3):
            intrinsic_4x4 = np.eye(4, dtype=np.float64)
            intrinsic_4x4[:3, :3] = intrinsic
            self.camera_intrinsic = intrinsic_4x4
        else:
            self.camera_intrinsic = intrinsic.astype(np.float64)
        
        self.camera_extrinsic = extrinsic.astype(np.float64) if extrinsic.ndim == 2 else extrinsic[0].astype(np.float64)
    
    def save_camera_calibration(self):
        """Speichert Kamera-Kalibrierung."""
        if self.camera_intrinsic is None or self.camera_extrinsic is None:
            return
        
        np.save(self.dataset_path / "cameras" / "intrinsic.npy", self.camera_intrinsic)
        extrinsic_expanded = np.stack([self.camera_extrinsic] * 4, axis=0)
        np.save(self.dataset_path / "cameras" / "extrinsic.npy", extrinsic_expanded)
    
    def start_episode(self, episode_id: Optional[int] = None):
        """Startet eine neue Episode."""
        if episode_id is not None:
            self.episode_count = episode_id
        
        episode_folder = self.dataset_path / f"{self.episode_count:06d}"
        episode_folder.mkdir(exist_ok=True)
        
        self.current_episode = {
            "id": self.episode_count,
            "folder": episode_folder,
            "imgs_list": [],       # wird zu (T, 1, H, W, 5) für obses.pth
            "timestep": 0,         # aktueller Timestep-Zähler
            "prev_ee_pos": None,   # vorherige EE-Position für Action
        }
        log.info(f"Episode {self.episode_count} gestartet")
    
    def log_step(
        self,
        rgb_image: np.ndarray,      # (H, W, 3) uint8
        depth_image: np.ndarray,    # (H, W) float32
        ee_pos: np.ndarray,         # (3,)
        ee_quat: np.ndarray,        # (4,) wxyz
        cube_positions: List[Tuple[float, float, float]] = None,
    ):
        """Loggt einen Timestep und speichert sofort eine .h5 Datei."""
        if self.current_episode is None:
            raise RuntimeError("Keine Episode gestartet!")
        
        ep = self.current_episode
        timestep = ep["timestep"]
        
        # Vorherige Position für Action (beim ersten Step = aktuelle Position)
        if ep["prev_ee_pos"] is None:
            ep["prev_ee_pos"] = ee_pos.astype(np.float64).copy()
        
        # Bilder: (H, W, 5) - RGB + pad + Depth
        H, W = rgb_image.shape[:2]
        img_combined = np.zeros((H, W, 5), dtype=np.float32)
        img_combined[:, :, :3] = rgb_image.astype(np.float32)
        img_combined[:, :, 4] = depth_image.astype(np.float32)
        
        # Für obses.pth speichern
        ep["imgs_list"].append(img_combined[np.newaxis, ...])  # (1, H, W, 5)
        
        # Positionen: (1, N, 3)
        if cube_positions is None:
            cube_positions = [(0.0, 0.0, 0.0)] * self.n_cubes
        positions = np.array([p[:3] if len(p) >= 3 else (p[0], p[1], 0.0) for p in cube_positions], dtype=np.float32)
        positions = positions[np.newaxis, ...]  # (1, N, 3)
        
        # EEF States: (1, 14) - pos1(3), pos2(3), quat1(4), quat2(4)
        eef_state = np.concatenate([
            ee_pos, ee_pos,
            ee_quat, ee_quat
        ]).astype(np.float64)
        eef_states = eef_state[np.newaxis, ...]  # (1, 14)
        
        # Action: (6,) = [prev_ee_pos, current_ee_pos]
        action = np.concatenate([ep["prev_ee_pos"], ee_pos.astype(np.float64)])
        
        # Bilder verarbeiten für .h5
        imgs_array = img_combined[np.newaxis, np.newaxis, ...]  # (1, 1, H, W, 5)
        color_imgs, depth_imgs = process_imgs(imgs_array)
        
        # Timestep-Daten für .h5
        timestep_data = {
            "info": {
                "n_cams": 1,
                "timestamp": 1,
                "n_particles": self.n_cubes,
            },
            "action": action,
            "positions": positions,
            "eef_states": eef_states,
            "observations": {"color": color_imgs, "depth": depth_imgs},
        }
        
        # .h5 Datei speichern (000.h5, 001.h5, etc.)
        h5_path = ep["folder"] / f"{timestep:03d}.h5"
        save_h5(h5_path, timestep_data)
        log.debug(f"Timestep {timestep} gespeichert: {h5_path}")
        
        # Für nächsten Step vorbereiten
        ep["prev_ee_pos"] = ee_pos.astype(np.float64).copy()
        ep["timestep"] += 1
    
    def end_episode(self):
        """Beendet Episode und speichert obses.pth sowie PNGs."""
        if self.current_episode is None:
            return
        
        ep = self.current_episode
        T = ep["timestep"]  # Anzahl der gespeicherten Timesteps
        
        if T == 0:
            log.warning(f"Episode {ep['id']}: Leer, überspringe...")
            self.current_episode = None
            return
        
        # Daten stacken für obses.pth
        imgs_list = np.stack(ep["imgs_list"], axis=0)  # (T, 1, H, W, 5)
        
        # Bilder verarbeiten
        color_imgs, depth_imgs = process_imgs(imgs_list)
        
        # ===== obses.pth speichern (wie im rope Datensatz) =====
        # Format: (T, H, W, 3) float32, Werte 0-255
        obses = color_imgs["cam_0"].astype(np.float32)  # (T, H, W, 3)
        obses_tensor = torch.from_numpy(obses)
        obses_path = ep["folder"] / "obses.pth"
        torch.save(obses_tensor, obses_path)
        log.debug(f"obses.pth gespeichert: {obses_tensor.shape}, dtype={obses_tensor.dtype}")
        
        # PNGs speichern
        if self.save_png:
            self._save_pngs(ep["folder"], color_imgs["cam_0"])
        
        log.info(f"Episode {ep['id']} beendet: {T} Timesteps ({T} .h5 Dateien), obses.pth gespeichert")
        
        self.episode_count += 1
        self.current_episode = None
    
    def _save_pngs(self, folder: Path, color_imgs: np.ndarray):
        """Speichert erstes und letztes Frame als PNG."""
        try:
            from PIL import Image
            for idx, name in [(0, "first"), (-1, "last")]:
                img = Image.fromarray(color_imgs[idx].astype(np.uint8))
                img.save(folder / f"{name}.png")
        except ImportError:
            log.warning("PIL nicht verfügbar für PNG-Speicherung")
    
    def discard_episode(self):
        """Verwirft aktuelle Episode."""
        if self.current_episode is None:
            return
        
        import shutil
        if self.current_episode["folder"].exists():
            shutil.rmtree(self.current_episode["folder"])
        
        log.warning(f"Episode {self.current_episode['id']} verworfen")
        self.current_episode = None

"""
Keyframe-basierter Data Logger für Isaac Sim - Franka Panda + Würfel
Löst das Problem des Pick-Point-Verlusts bei Trajektorien-Komprimierung.

PROBLEM:
    Bei naivem action_interval Subsampling gehen kritische Punkte verloren:
    - Pick-Point (tiefster Punkt beim Greifen)
    - Place-Point (tiefster Punkt beim Ablegen)
    - Greifer-Aktionen (öffnen/schließen)

LÖSUNG:
    Keyframe-basiertes Logging:
    1. IMMER speichern bei Phase-Wechsel (besonders 1→2 und 6→7)
    2. IMMER speichern bei Greifer-Aktionen
    3. IMMER speichern bei lokalen Z-Minima (tiefste Punkte)
    4. Zwischen Keyframes: Sparse Sampling (nur jeden N-ten Frame)

Verwendung:
    logger = KeyframeDataLogger(config)
    logger.start_episode()
    
    for frame in simulation:
        logger.log_step(
            rgb_image=rgb,
            depth_image=depth,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            cube_positions=cube_positions,
            phase=controller.get_current_event(),      # 0-9
            gripper_state=gripper.get_joint_positions() # für Greifer-Erkennung
        )
    
    logger.end_episode()

Ausgabe-Format: Identisch zu MinDataLogger (DINO-WM kompatibel)
"""

import numpy as np
import h5py
import yaml
import torch
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from enum import IntEnum
import logging

log = logging.getLogger("KeyframeDataLogger")


class Phase(IntEnum):
    """Controller-Phasen für Pick-and-Place."""
    MOVE_ABOVE = 0      # AIR - Move EE above cube
    LOWER_TO_GRIP = 1   # CRITICAL - Lower to pick (★ KEYFRAME: Ende = Pick-Point!)
    WAIT = 2            # WAIT - Wait for settle
    CLOSE_GRIPPER = 3   # GRIP - Close gripper (★ KEYFRAME!)
    LIFT_WITH_CUBE = 4  # AIR - Lift up with cube
    MOVE_TO_TARGET = 5  # AIR - Move to target XY
    LOWER_TO_PLACE = 6  # CRITICAL - Lower to place (★ KEYFRAME: Ende = Place-Point!)
    OPEN_GRIPPER = 7    # RELEASE - Open gripper (★ KEYFRAME!)
    LIFT_UP = 8         # AIR - Lift up
    RETURN = 9          # AIR - Return to start


# Phasen bei denen IMMER ein Keyframe gesetzt wird
KEYFRAME_PHASES = {
    Phase.LOWER_TO_GRIP,    # Ende = Pick-Point
    Phase.WAIT,             # Nach Pick-Point
    Phase.CLOSE_GRIPPER,    # Greifer schließt
    Phase.LOWER_TO_PLACE,   # Ende = Place-Point  
    Phase.OPEN_GRIPPER,     # Greifer öffnet
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Lädt Konfiguration aus YAML-Datei."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_imgs(imgs_list: np.ndarray):
    """Verarbeitet Bilder wie in data.py."""
    T, n_cam, H, W, _ = imgs_list.shape
    color_imgs = {}
    depth_imgs = {}

    for cam_idx in range(n_cam):
        img = imgs_list[:, cam_idx]
        color_imgs[f"cam_{cam_idx}"] = img[:, :, :, :3][..., ::-1]  # BGR->RGB
        depth_imgs[f"cam_{cam_idx}"] = (img[:, :, :, -1] * 1000).astype(np.uint16)

    return color_imgs, depth_imgs


def save_h5(filename: Path, data: dict):
    """Speichert Daten als H5."""
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


class KeyframeDataLogger:
    """
    Keyframe-basierter Data Logger für Franka + Würfel.
    
    Garantiert, dass kritische Punkte (Pick/Place) IMMER im Datensatz sind,
    unabhängig vom Sampling-Interval.
    """
    
    def __init__(
        self, 
        config: Optional[dict] = None, 
        config_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        # Keyframe-Einstellungen
        sparse_interval: int = 5,           # Frames zwischen Keyframes (AIR-Phasen)
        min_z_threshold: float = 0.001,     # Mindest-Z-Änderung für lokales Minimum
        always_save_phase_transitions: bool = True,  # Phase-Wechsel = Keyframe
    ):
        if config is None:
            config = load_config(config_path)
        
        self.config = config
        self.dataset_path = Path(dataset_path) if dataset_path else Path("./dataset")
        self.image_size = tuple(config["camera"]["resolution"])
        self.save_png = config["dataset"].get("save_png", False)  # Default: False für Kompression
        self.n_cubes = config["cubes"]["count"]
        
        # Keyframe-Einstellungen
        self.sparse_interval = sparse_interval
        self.min_z_threshold = min_z_threshold
        self.always_save_phase_transitions = always_save_phase_transitions
        
        self.camera_intrinsic: Optional[np.ndarray] = None
        self.camera_extrinsic: Optional[np.ndarray] = None
        
        self.current_episode = None
        self.episode_count = 0
        
        # Globale Listen für states.pth und actions.pth
        self.all_actions: List[List[np.ndarray]] = []
        self.all_states: List[List[np.ndarray]] = []
        
        self._setup_directories()
        log.info(f"KeyframeDataLogger initialisiert: {self.dataset_path}")
        log.info(f"  Sparse Interval: {sparse_interval} (Frames zwischen Keyframes in AIR-Phasen)")
        log.info(f"  Min Z Threshold: {min_z_threshold}m (für lokale Minima-Erkennung)")
        log.info(f"  Phase-Wechsel = Keyframe: {always_save_phase_transitions}")
        log.info(f"  Keyframe-Phasen: {[p.name for p in KEYFRAME_PHASES]}")
    
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
            # Temporärer Buffer für ALLE Frames (für Z-Minima Analyse)
            "all_frames": [],
            # Finaler Output (nur Keyframes)
            "keyframes": [],
            "keyframe_indices": [],  # Original-Indizes der Keyframes
            # Tracking
            "frame_count": 0,
            "last_phase": None,
            "last_z": None,
            "z_decreasing": False,  # War Z im letzten Frame noch am sinken?
            "frames_since_keyframe": 0,
            # Stats
            "stats": {
                "total_frames": 0,
                "keyframes_phase": 0,
                "keyframes_z_min": 0,
                "keyframes_sparse": 0,
            }
        }
        log.info(f"Episode {self.episode_count} gestartet (Keyframe-Modus)")
    
    def log_step(
        self,
        rgb_image: np.ndarray,      # (H, W, 3) uint8
        depth_image: np.ndarray,    # (H, W) float32
        ee_pos: np.ndarray,         # (3,)
        ee_quat: np.ndarray,        # (4,) wxyz
        cube_positions: List[Tuple[float, float, float]] = None,
        phase: int = None,          # Controller-Phase 0-9
        gripper_state: float = None, # Greifer-Position (optional)
    ):
        """
        Loggt einen Timestep und entscheidet ob es ein Keyframe ist.
        
        Ein Keyframe wird gesetzt bei:
        1. Phase-Wechsel (besonders zu/von kritischen Phasen)
        2. Lokalem Z-Minimum (tiefster Punkt erreicht)
        3. Nach sparse_interval Frames ohne Keyframe
        """
        if self.current_episode is None:
            raise RuntimeError("Keine Episode gestartet!")
        
        ep = self.current_episode
        frame_idx = ep["frame_count"]
        
        # Frame-Daten zusammenstellen
        frame_data = {
            "rgb": rgb_image.copy(),
            "depth": depth_image.copy(),
            "ee_pos": ee_pos.copy(),
            "ee_quat": ee_quat.copy(),
            "cube_positions": cube_positions.copy() if cube_positions else [(0.0, 0.0, 0.0)] * self.n_cubes,
            "phase": phase,
            "gripper_state": gripper_state,
            "frame_idx": frame_idx,
        }
        
        # In temporären Buffer speichern (für Post-Processing)
        ep["all_frames"].append(frame_data)
        
        # === Keyframe-Entscheidung ===
        is_keyframe = False
        keyframe_reason = ""
        
        current_z = ee_pos[2]
        
        # 1. Phase-Wechsel?
        if self.always_save_phase_transitions and phase is not None:
            if ep["last_phase"] is not None and phase != ep["last_phase"]:
                is_keyframe = True
                keyframe_reason = f"Phase {ep['last_phase']}→{phase}"
                ep["stats"]["keyframes_phase"] += 1
                
                # Speziell: Am ENDE von Phase 1 oder 6 (tiefster Punkt!)
                if ep["last_phase"] in [Phase.LOWER_TO_GRIP, Phase.LOWER_TO_PLACE]:
                    keyframe_reason += " (PICK/PLACE-POINT!)"
        
        # 2. Lokales Z-Minimum? (EE war am sinken, jetzt steigt es)
        if ep["last_z"] is not None and not is_keyframe:
            z_diff = current_z - ep["last_z"]
            was_decreasing = ep["z_decreasing"]
            
            # Jetzt steigend, aber vorher sinkend = lokales Minimum!
            if was_decreasing and z_diff > self.min_z_threshold:
                is_keyframe = True
                keyframe_reason = f"Z-Minimum bei z={ep['last_z']:.4f}m"
                ep["stats"]["keyframes_z_min"] += 1
            
            # Update z_decreasing Status
            ep["z_decreasing"] = z_diff < -self.min_z_threshold
        
        # 3. Sparse Sampling (falls lange kein Keyframe)
        if not is_keyframe:
            ep["frames_since_keyframe"] += 1
            if ep["frames_since_keyframe"] >= self.sparse_interval:
                is_keyframe = True
                keyframe_reason = f"Sparse (nach {self.sparse_interval} Frames)"
                ep["stats"]["keyframes_sparse"] += 1
        
        # 4. Erster und letzter Frame sind IMMER Keyframes
        if frame_idx == 0:
            is_keyframe = True
            keyframe_reason = "Erster Frame"
        
        # === Keyframe speichern ===
        if is_keyframe:
            ep["keyframes"].append(frame_data)
            ep["keyframe_indices"].append(frame_idx)
            ep["frames_since_keyframe"] = 0
            log.debug(f"  Keyframe #{len(ep['keyframes'])} @ Frame {frame_idx}: {keyframe_reason}")
        
        # Update Tracking
        ep["last_phase"] = phase
        ep["last_z"] = current_z
        ep["frame_count"] += 1
        ep["stats"]["total_frames"] += 1
    
    def end_episode(self, property_params: Optional[Dict[str, Any]] = None):
        """
        Beendet Episode und speichert NUR die Keyframes.
        """
        if self.current_episode is None:
            return
        
        ep = self.current_episode
        
        # Letzten Frame als Keyframe markieren (falls noch nicht)
        if ep["all_frames"] and ep["keyframe_indices"][-1] != ep["frame_count"] - 1:
            last_frame = ep["all_frames"][-1]
            ep["keyframes"].append(last_frame)
            ep["keyframe_indices"].append(last_frame["frame_idx"])
            log.debug(f"  Keyframe #{len(ep['keyframes'])} @ Frame {last_frame['frame_idx']}: Letzter Frame")
        
        T = len(ep["keyframes"])
        
        if T == 0:
            log.warning(f"Episode {ep['id']}: Keine Keyframes, überspringe...")
            self.current_episode = None
            return
        
        # === H5-Dateien für jeden Keyframe speichern ===
        imgs_list = []
        actions_list = []
        states_list = []
        prev_ee_pos = None
        
        for kf_idx, kf in enumerate(ep["keyframes"]):
            rgb = kf["rgb"]
            depth = kf["depth"]
            ee_pos = kf["ee_pos"]
            ee_quat = kf["ee_quat"]
            cube_positions = kf["cube_positions"]
            
            # Bilder: (H, W, 5) - RGB + pad + Depth
            H, W = rgb.shape[:2]
            img_combined = np.zeros((H, W, 5), dtype=np.float32)
            img_combined[:, :, :3] = rgb.astype(np.float32)
            img_combined[:, :, 4] = depth.astype(np.float32)
            imgs_list.append(img_combined[np.newaxis, ...])
            
            # Action: (6,) = [prev_ee_pos, current_ee_pos]
            if prev_ee_pos is None:
                prev_ee_pos = ee_pos.astype(np.float64)
            action = np.concatenate([prev_ee_pos, ee_pos.astype(np.float64)])
            actions_list.append(action)
            prev_ee_pos = ee_pos.astype(np.float64).copy()
            
            # Positionen: (N, 4) - [x, y, z, 1.0]
            pos_xyz = np.array([p[:3] if len(p) >= 3 else (p[0], p[1], 0.0) for p in cube_positions], dtype=np.float32)
            pos_ones = np.ones((pos_xyz.shape[0], 1), dtype=np.float32)
            positions = np.concatenate([pos_xyz, pos_ones], axis=1)
            states_list.append(positions)
            
            # EEF States: (1, 1, 14)
            eef_state = np.concatenate([ee_pos, ee_pos, ee_quat, ee_quat]).astype(np.float64)
            eef_states = eef_state[np.newaxis, np.newaxis, ...]
            
            # Bilder für H5
            imgs_array = img_combined[np.newaxis, np.newaxis, ...]
            color_imgs, depth_imgs = process_imgs(imgs_array)
            
            # H5 speichern
            timestep_data = {
                "info": {
                    "n_cams": 1,
                    "timestamp": 1,
                    "n_particles": self.n_cubes,
                    "original_frame_idx": kf["frame_idx"],  # Original-Index für Debugging
                },
                "action": action,
                "positions": positions[np.newaxis, ...],
                "eef_states": eef_states,
                "observations": {"color": color_imgs, "depth": depth_imgs},
            }
            
            h5_path = ep["folder"] / f"{kf_idx:03d}.h5"
            save_h5(h5_path, timestep_data)
        
        # === obses.pth speichern ===
        imgs_array = np.stack(imgs_list, axis=0)[:, 0]  # (T, H, W, 5)
        color_imgs, _ = process_imgs(imgs_array[:, np.newaxis, ...])
        obses = color_imgs["cam_0"].astype(np.float32)
        obses_tensor = torch.from_numpy(obses)
        torch.save(obses_tensor, ep["folder"] / "obses.pth")
        
        # === property_params.pkl speichern ===
        if property_params is None:
            property_params = {
                "n_cubes": self.n_cubes,
                "keyframe_indices": ep["keyframe_indices"],  # Für Debugging
            }
        property_params["keyframe_stats"] = ep["stats"]
        
        with open(ep["folder"] / "property_params.pkl", "wb") as f:
            pickle.dump(property_params, f)
        
        # Für globale Dateien
        self.all_actions.append(actions_list)
        self.all_states.append(states_list)
        
        # PNGs speichern
        if self.save_png:
            self._save_pngs(ep["folder"], obses)
        
        # Stats ausgeben
        stats = ep["stats"]
        compression = stats["total_frames"] / T if T > 0 else 1
        log.info(f"Episode {ep['id']} beendet:")
        log.info(f"  Total Frames: {stats['total_frames']} → Keyframes: {T} ({compression:.1f}x Kompression)")
        log.info(f"  Keyframe-Verteilung:")
        log.info(f"    - Phase-Wechsel: {stats['keyframes_phase']}")
        log.info(f"    - Z-Minima: {stats['keyframes_z_min']}")
        log.info(f"    - Sparse: {stats['keyframes_sparse']}")
        log.info(f"  Keyframe-Indizes: {ep['keyframe_indices'][:10]}{'...' if len(ep['keyframe_indices']) > 10 else ''}")
        
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
    
    def save_global_data(self):
        """Speichert globale actions.pth und states.pth."""
        if len(self.all_actions) == 0:
            log.warning("Keine Episoden vorhanden, überspringe save_global_data()")
            return
        
        T_max = max(len(ep_actions) for ep_actions in self.all_actions)
        N_episodes = len(self.all_actions)
        action_dim = len(self.all_actions[0][0]) if self.all_actions[0] else 6
        
        actions_array = np.zeros((N_episodes, T_max, action_dim), dtype=np.float32)
        states_array = np.zeros((N_episodes, T_max, self.n_cubes, 4), dtype=np.float32)
        
        for ep_idx, (ep_actions, ep_states) in enumerate(zip(self.all_actions, self.all_states)):
            T = len(ep_actions)
            actions_array[ep_idx, :T, :] = np.array(ep_actions, dtype=np.float32)
            states_array[ep_idx, :T, :, :] = np.array(ep_states, dtype=np.float32)
        
        actions_tensor = torch.from_numpy(actions_array)
        states_tensor = torch.from_numpy(states_array)
        
        torch.save(actions_tensor, self.dataset_path / "actions.pth")
        torch.save(states_tensor, self.dataset_path / "states.pth")
        
        log.info(f"Globale Daten gespeichert:")
        log.info(f"  actions.pth: {actions_tensor.shape}")
        log.info(f"  states.pth: {states_tensor.shape}")
    
    def discard_episode(self):
        """Verwirft aktuelle Episode."""
        if self.current_episode is None:
            return
        
        import shutil
        if self.current_episode["folder"].exists():
            shutil.rmtree(self.current_episode["folder"])
        
        log.warning(f"Episode {self.current_episode['id']} verworfen")
        self.current_episode = None

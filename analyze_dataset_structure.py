"""
Analysiere die Struktur des deformable/rope Datensatzes.
Dieses Skript zeigt die genaue Struktur, damit du sie reproduzieren kannst.
"""

import torch
import numpy as np
import h5py
import pickle
from pathlib import Path

# Pfad zum Datensatz
dataset_path = Path("D:/00_Coding/deformable/deformable/rope")

print("=" * 60)
print("DATENSATZ-STRUKTUR ANALYSE")
print("=" * 60)

# 1. Hauptdateien im rope-Ordner
print("\n📁 Hauptordner-Struktur (rope/):")
print("-" * 40)

# States
states_path = dataset_path / "states.pth"
if states_path.exists():
    states = torch.load(states_path)
    print(f"  states.pth:")
    print(f"    - Type: {type(states).__name__}")
    print(f"    - Shape: {states.shape}")
    print(f"    - Dtype: {states.dtype}")
    print(f"    - Bedeutung: (n_rollout, n_timestep, n_particles, 4)")
    print(f"    - n_rollout = {states.shape[0]} (Anzahl Trajektorien)")
    print(f"    - n_timestep = {states.shape[1]} (Schritte pro Trajektorie)")
    print(f"    - n_particles = {states.shape[2]} (Partikel)")
    print(f"    - 4 = (x, y, z, ?) Koordinaten")

# Actions
actions_path = dataset_path / "actions.pth"
if actions_path.exists():
    actions = torch.load(actions_path)
    print(f"\n  actions.pth:")
    print(f"    - Type: {type(actions).__name__}")
    print(f"    - Shape: {actions.shape}")
    print(f"    - Dtype: {actions.dtype}")
    print(f"    - Bedeutung: (n_rollout, n_timestep, action_dim)")
    print(f"    - action_dim = {actions.shape[-1]}")

# Camera Intrinsics/Extrinsics
cameras_path = dataset_path / "cameras"
if cameras_path.exists():
    print(f"\n  cameras/:")
    intrinsic = np.load(cameras_path / "intrinsic.npy")
    extrinsic = np.load(cameras_path / "extrinsic.npy")
    print(f"    intrinsic.npy: Shape {intrinsic.shape}, Dtype {intrinsic.dtype}")
    print(f"    extrinsic.npy: Shape {extrinsic.shape}, Dtype {extrinsic.dtype}")

# 2. Struktur eines einzelnen Rollout-Ordners
print("\n" + "=" * 60)
print("📁 Rollout-Ordner Struktur (z.B. 000000/):")
print("-" * 40)

rollout_path = dataset_path / "000000"
if rollout_path.exists():
    # obses.pth
    obses_path = rollout_path / "obses.pth"
    if obses_path.exists():
        obses = torch.load(obses_path)
        print(f"  obses.pth:")
        print(f"    - Type: {type(obses).__name__}")
        print(f"    - Shape: {obses.shape}")
        print(f"    - Dtype: {obses.dtype}")
        print(f"    - Bedeutung: (n_timestep, H, W, C) - Bilder")
        print(f"    - Bildgröße: {obses.shape[1]}x{obses.shape[2]} mit {obses.shape[3]} Kanälen")
        print(f"    - Wertbereich: [{obses.min()}, {obses.max()}]")

    # property_params.pkl
    params_path = rollout_path / "property_params.pkl"
    if params_path.exists():
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        print(f"\n  property_params.pkl:")
        print(f"    - Type: {type(params).__name__}")
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    print(f"    - {key}: numpy array, shape {value.shape}")
                else:
                    print(f"    - {key}: {type(value).__name__} = {value}")
        else:
            print(f"    - Content: {params}")

    # .h5 Dateien
    h5_files = list(rollout_path.glob("*.h5"))
    if h5_files:
        print(f"\n  .h5 Dateien ({len(h5_files)} Dateien: 00.h5 bis {len(h5_files)-1:02d}.h5):")
        # Analysiere eine H5-Datei als Beispiel
        with h5py.File(h5_files[0], "r") as f:
            print(f"    Beispiel ({h5_files[0].name}):")
            def print_h5_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"      - {name}: shape {obj.shape}, dtype {obj.dtype}")
            f.visititems(print_h5_structure)

# 3. Zusammenfassung für neue Datensatz-Erstellung
print("\n" + "=" * 60)
print("📋 ZUSAMMENFASSUNG - STRUKTUR FÜR NEUEN DATENSATZ:")
print("=" * 60)
print("""
deformable/
└── <object_name>/                    # z.B. "rope", "cloth", etc.
    ├── states.pth                    # torch.Tensor: (N, T, P, 4) float32
    │                                 #   N = Anzahl Trajektorien
    │                                 #   T = Timesteps pro Trajektorie
    │                                 #   P = Anzahl Partikel
    │                                 #   4 = Partikel-Eigenschaften (x,y,z,...)
    │
    ├── actions.pth                   # torch.Tensor: (N, T, action_dim) float32
    │                                 #   action_dim = Dimension der Aktionen
    │
    ├── cameras/                      # (optional) Kamera-Kalibrierung
    │   ├── intrinsic.npy             # numpy array
    │   └── extrinsic.npy             # numpy array
    │
    ├── 000000/                       # Rollout 0 (6-stellig, 0-padded)
    │   ├── obses.pth                 # torch.Tensor: (T, H, W, C) uint8
    │   │                             #   Bilder der Beobachtungen
    │   │                             #   Werte: 0-255 (wird durch 255 geteilt)
    │   ├── property_params.pkl       # pickle: Zusätzliche Parameter
    │   ├── 00.h5                     # HDF5: Timestep 0 Daten
    │   ├── 01.h5                     # HDF5: Timestep 1 Daten
    │   └── ...                       # bis (T-1).h5
    │
    ├── 000001/                       # Rollout 1
    │   └── ...
    │
    └── ...                           # bis 00XXXX (N-1 Rollouts)
""")

print("\n📌 WICHTIGE HINWEISE:")
print("-" * 40)
print(f"  - Anzahl Rollouts (N): {states.shape[0]}")
print(f"  - Timesteps pro Rollout (T): {states.shape[1]}")
print(f"  - Bildgröße: {obses.shape[1]}x{obses.shape[2]}x{obses.shape[3]}")
print(f"  - Action-Dimension: {actions.shape[-1]}")
print(f"  - State-Dimension (pro Partikel): 4")
print(f"  - Anzahl Partikel: {states.shape[2]}")


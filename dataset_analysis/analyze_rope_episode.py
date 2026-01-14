"""
Analysiere die komplette Struktur einer Episode aus dem rope-Datensatz.
Dieses Skript zeigt alle Details, damit die Struktur exakt repliziert werden kann.
"""

import numpy as np
import h5py
import pickle
from pathlib import Path
import json

# Try to import torch, but continue without it if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  WARNING: torch nicht verfügbar - obses.pth wird nicht analysiert")

# Pfad zur Episode
episode_path = Path("/media/tsp_jw/fc8bca1b-cab8-4522-81d0-06172d2beae8/deformable/rope/000001")
cameras_path = Path("/media/tsp_jw/fc8bca1b-cab8-4522-81d0-06172d2beae8/deformable/rope/cameras")

print("=" * 80)
print("KOMPLETTE EPISODE-STRUKTUR ANALYSE: 000001")
print("=" * 80)

# 1. Kamera-Daten
print("\n📷 KAMERA-DATEN:")
print("-" * 80)
if cameras_path.exists():
    intrinsic = np.load(cameras_path / "intrinsic.npy")
    extrinsic = np.load(cameras_path / "extrinsic.npy")
    print(f"  intrinsic.npy:")
    print(f"    - Shape: {intrinsic.shape}")
    print(f"    - Dtype: {intrinsic.dtype}")
    print(f"    - Content:\n{intrinsic}")
    print(f"\n  extrinsic.npy:")
    print(f"    - Shape: {extrinsic.shape}")
    print(f"    - Dtype: {extrinsic.dtype}")
    print(f"    - Content:\n{extrinsic}")
else:
    print("  ❌ cameras/ Ordner nicht gefunden")

# 2. obses.pth - Beobachtungen (Bilder)
print("\n" + "=" * 80)
print("🖼️  OBSERVATIONS (obses.pth):")
print("-" * 80)
obses_path = episode_path / "obses.pth"
obses = None
if obses_path.exists():
    if HAS_TORCH:
        obses = torch.load(obses_path, map_location='cpu')
        print(f"  Type: {type(obses).__name__}")
        print(f"  Shape: {obses.shape}")
        print(f"  Dtype: {obses.dtype}")
        print(f"  Min: {obses.min().item()}, Max: {obses.max().item()}")
        print(f"  Bedeutung: (n_timesteps, height, width, channels)")
        print(f"    - n_timesteps = {obses.shape[0]}")
        print(f"    - height = {obses.shape[1]}")
        print(f"    - width = {obses.shape[2]}")
        print(f"    - channels = {obses.shape[3]}")
        
        # Zeige Beispielwerte
        print(f"\n  Beispiel (Timestep 0, Pixel [0,0]): {obses[0, 0, 0, :].numpy()}")
    else:
        print("  ⚠️  torch nicht verfügbar - kann obses.pth nicht laden")
        print(f"  Dateigröße: {obses_path.stat().st_size / (1024*1024):.2f} MB")
else:
    print("  ❌ obses.pth nicht gefunden")

# 3. property_params.pkl
print("\n" + "=" * 80)
print("⚙️  PROPERTY PARAMETERS (property_params.pkl):")
print("-" * 80)
params_path = episode_path / "property_params.pkl"
if params_path.exists():
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    print(f"  Type: {type(params).__name__}")
    
    if isinstance(params, dict):
        print(f"  Keys: {list(params.keys())}")
        print(f"\n  Details:")
        for key, value in params.items():
            print(f"    - {key}:")
            if isinstance(value, np.ndarray):
                print(f"        Type: numpy.ndarray")
                print(f"        Shape: {value.shape}")
                print(f"        Dtype: {value.dtype}")
                if value.size <= 20:
                    print(f"        Content: {value}")
                else:
                    print(f"        Content (first 10): {value.flat[:10]}")
            elif isinstance(value, (list, tuple)):
                print(f"        Type: {type(value).__name__}")
                print(f"        Length: {len(value)}")
                if len(value) <= 10:
                    print(f"        Content: {value}")
                else:
                    print(f"        Content (first 5): {value[:5]}")
            else:
                print(f"        Type: {type(value).__name__}")
                print(f"        Value: {value}")
    else:
        print(f"  Content: {params}")
        print(f"  Repr: {repr(params)}")
else:
    print("  ❌ property_params.pkl nicht gefunden")

# 4. H5-Dateien - Detaillierte Analyse
print("\n" + "=" * 80)
print("📦 H5-DATEIEN (State-Daten pro Timestep):")
print("-" * 80)
h5_files = sorted(episode_path.glob("*.h5"))
print(f"  Anzahl H5-Dateien: {len(h5_files)}")

if h5_files:
    # Analysiere die erste H5-Datei vollständig
    print(f"\n  Detaillierte Analyse von {h5_files[0].name}:")
    with h5py.File(h5_files[0], "r") as f:
        def print_h5_structure(name, obj, indent=0):
            prefix = "    " + "  " * indent
            if isinstance(obj, h5py.Dataset):
                print(f"{prefix}📄 {name}:")
                print(f"{prefix}    Shape: {obj.shape}")
                print(f"{prefix}    Dtype: {obj.dtype}")
                print(f"{prefix}    Size: {obj.size} elements")
                
                # Zeige Beispielwerte für kleine Arrays
                try:
                    if obj.shape == ():  # Scalar
                        print(f"{prefix}    Content: {obj[()]}")
                    elif obj.size <= 50:
                        print(f"{prefix}    Content:\n{prefix}      {obj[:]}")
                    elif obj.size <= 500:
                        print(f"{prefix}    Content (first 20):\n{prefix}      {obj[:20]}")
                    else:
                        print(f"{prefix}    Content (first 10):\n{prefix}      {obj[:10]}")
                        print(f"{prefix}    Content (last 10):\n{prefix}      {obj[-10:]}")
                except Exception as e:
                    print(f"{prefix}    Content: [Fehler beim Lesen: {e}]")
                
                # Zeige Attribute
                if obj.attrs:
                    print(f"{prefix}    Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{prefix}      - {attr_name}: {attr_value}")
            elif isinstance(obj, h5py.Group):
                print(f"{prefix}📁 {name}/")
                if obj.attrs:
                    print(f"{prefix}    Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{prefix}      - {attr_name}: {attr_value}")
        
        f.visititems(print_h5_structure)
    
    # Vergleiche mehrere H5-Dateien
    print(f"\n  Struktur-Vergleich über mehrere Timesteps:")
    structures = {}
    for h5_file in h5_files[:5]:  # Erste 5 Dateien
        with h5py.File(h5_file, "r") as f:
            structure = {}
            def collect_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    structure[name] = {
                        'shape': obj.shape,
                        'dtype': str(obj.dtype)
                    }
            f.visititems(collect_structure)
            structures[h5_file.name] = structure
    
    # Zeige ob alle gleich sind
    first_structure = structures[h5_files[0].name]
    all_same = all(s == first_structure for s in structures.values())
    print(f"    Alle H5-Dateien haben die gleiche Struktur: {all_same}")
    if all_same:
        print(f"    Gemeinsame Struktur:")
        for key, value in first_structure.items():
            print(f"      - {key}: shape {value['shape']}, dtype {value['dtype']}")

# 5. Zusammenfassung und Replikations-Guide
print("\n" + "=" * 80)
print("📋 ZUSAMMENFASSUNG & REPLIKATIONS-GUIDE:")
print("=" * 80)

# Sammle alle Informationen
summary = {
    "episode_structure": {
        "obses.pth": {
            "shape": list(obses.shape) if obses is not None else None,
            "dtype": str(obses.dtype) if obses is not None else None,
        },
        "property_params.pkl": {
            "type": type(params).__name__ if params_path.exists() else None,
            "keys": list(params.keys()) if params_path.exists() and isinstance(params, dict) else None,
        },
        "h5_files": {
            "count": len(h5_files),
            "structure": first_structure if h5_files else None,
        },
        "cameras": {
            "intrinsic_shape": list(intrinsic.shape) if cameras_path.exists() else None,
            "extrinsic_shape": list(extrinsic.shape) if cameras_path.exists() else None,
        }
    }
}

print(f"""
EPISODE-STRUKTUR FÜR REPLIKATION:

deformable/rope/000001/
├── obses.pth                    # torch.Tensor
│   └── Shape: {summary['episode_structure']['obses.pth']['shape']}
│   └── Dtype: {summary['episode_structure']['obses.pth']['dtype']}
│
├── property_params.pkl          # pickle file
│   └── Type: {summary['episode_structure']['property_params.pkl']['type']}
│   └── Keys: {summary['episode_structure']['property_params.pkl']['keys']}
│
└── 00.h5 bis {len(h5_files)-1:02d}.h5    # HDF5 files ({len(h5_files)} files)
    └── Struktur: {len(first_structure) if h5_files else 0} Datasets pro Datei
""")

# Speichere die Zusammenfassung als JSON
output_path = episode_path.parent / "episode_structure_summary.json"
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\n✅ Detaillierte Zusammenfassung gespeichert in: {output_path}")

print("\n" + "=" * 80)
print("🎯 FÜR 2 WÜRFEL STATT ROPE:")
print("=" * 80)
print("""
Anpassungen die du vornehmen musst:

1. PARTIKEL-ANZAHL:
   - Rope: 20 Kettenglieder (vermutlich 20 Partikel)
   - Würfel: 2 Würfel (vermutlich 2 Partikel pro Würfel = 4 Partikel, oder 8 Ecken = 16 Partikel)
   - Passe die Shape in den H5-Dateien entsprechend an

2. PROPERTY_PARAMS.PKL:
   - Enthält wahrscheinlich physikalische Eigenschaften des Ropes
   - Für Würfel: Masse, Größe, Material-Eigenschaften anpassen

3. STATES IN H5:
   - Aktuelle Struktur zeigt Partikel-Positionen des Ropes
   - Für Würfel: Positionen der Würfel (Center + Rotation, oder Ecken)

4. OBSERVATIONS:
   - obses.pth sollte unverändert bleiben (Bilder)
   - Aber die visuelle Darstellung zeigt jetzt Würfel statt Rope
""")


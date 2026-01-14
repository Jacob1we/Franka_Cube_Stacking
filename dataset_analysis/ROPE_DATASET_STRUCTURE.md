# Rope Dataset Struktur - Komplette Dokumentation

## Übersicht

Dieses Dokument beschreibt die exakte Struktur des `deformable/rope` Datensatzes, damit sie für einen Datensatz mit 2 Würfeln repliziert werden kann.

## Verzeichnisstruktur

```
deformable/rope/
├── cameras/                      # Kamera-Kalibrierung (global für alle Episoden)
│   ├── intrinsic.npy            # (4, 4) float64 - Intrinsische Kameraparameter
│   └── extrinsic.npy            # (4, 4, 4) float64 - Extrinsische Parameter für 4 Kameras
│
└── 000001/                      # Episode 1 (6-stellig, 0-padded)
    ├── obses.pth                # (T, H, W, C) uint8 - Alle Beobachtungsbilder
    ├── property_params.pkl      # dict - Physikalische Eigenschaften des Ropes
    ├── 00.h5                    # HDF5 - Timestep 0 Daten
    ├── 01.h5                    # HDF5 - Timestep 1 Daten
    ├── ...
    └── 20.h5                    # HDF5 - Timestep 20 Daten (21 Timesteps total)
```

## Detaillierte Beschreibung

### 1. Kamera-Daten (`cameras/`)

#### `intrinsic.npy`
- **Shape**: `(4, 4)` 
- **Dtype**: `float64`
- **Bedeutung**: Intrinsische Kameraparameter (4x4 Matrix für jede der 4 Kameras)
- **Inhalt**: 
  ```
  [[270.39, 270.39, 112.0, 112.0],
   [270.39, 270.39, 112.0, 112.0],
   [270.39, 270.39, 112.0, 112.0],
   [270.39, 270.39, 112.0, 112.0]]
  ```

#### `extrinsic.npy`
- **Shape**: `(4, 4, 4)`
- **Dtype**: `float64`
- **Bedeutung**: Extrinsische Kameraparameter (4x4 Transformationsmatrizen für 4 Kameras)
- **Struktur**: `[cam_0, cam_1, cam_2, cam_3]` - jeweils eine 4x4 Transformationsmatrix

### 2. Episode-Daten (`000001/`)

#### `obses.pth`
- **Format**: PyTorch Tensor
- **Shape**: Vermutlich `(T, N_CAMS, H, W, C)` = `(21, 4, 224, 224, 3)` basierend auf Dateigröße
  - Alternative: `(T, H, W, C)` = `(21, 224, 224, 3)` wenn nur eine Kamera-Ansicht
- **Dtype**: `uint8`
- **Bedeutung**: Alle Beobachtungsbilder der Episode zusammengefasst
- **Wertebereich**: `0-255`
- **Dateigröße**: 12.06 MB
- **Berechnung**: 
  - `21 timesteps × 4 cameras × 224×224×3 bytes = 12,644,352 bytes ≈ 12.06 MB` ✓
  - Dies deutet auf Shape `(21, 4, 224, 224, 3)` hin

#### `property_params.pkl`
- **Format**: Python Pickle (dict)
- **Inhalt**:
  ```python
  {
      'particle_radius': 0.03,        # Radius der Partikel
      'length': 2.5,                   # Länge des Ropes
      'thickness': 3.0,                # Dicke
      'dynamic_friction': 0.1,         # Dynamische Reibung
      'cluster_spacing': 7.2,          # Abstand zwischen Clustern
      'global_stiffness': 0.00022,     # Globale Steifigkeit
      'stiffness': 0.8                 # Steifigkeit
  }
  ```

### 3. H5-Dateien (pro Timestep)

Jede H5-Datei (`00.h5` bis `20.h5`) enthält die folgenden Datasets:

#### Root-Level Datasets

1. **`action`**
   - **Shape**: `(4,)`
   - **Dtype**: `float64`
   - **Bedeutung**: Aktion des Roboters (vermutlich Endeffektor-Position/Orientierung)
   - **Beispiel**: `[0.0, 0.0, 0.0, 0.0]`

2. **`eef_states`** (End-Effector States)
   - **Shape**: `(1, 1, 14)`
   - **Dtype**: `float64`
   - **Bedeutung**: Zustand des Endeffektors (Position, Orientierung, etc.)
   - **Struktur**: `[[[x, y, z, x, y, z, qw, qx, qy, qz, qw, qx, qy, qz]]]`
   - **Vermutung**: Position (3D) + Quaternion (4D) + nochmal Position + Quaternion = 14 Werte

3. **`positions`**
   - **Shape**: `(1, 1965, 4)`
   - **Dtype**: `float32`
   - **Bedeutung**: Positionen aller Partikel des Ropes
   - **Struktur**: `(batch, n_particles, 4)`
   - **4 Werte**: Vermutlich `(x, y, z, w)` - homogene Koordinaten (w=1.0)
   - **Wichtig**: `n_particles = 1965` - das sind die Partikel des Ropes (nicht 20!)

#### Info-Gruppe (`info/`)

4. **`info/n_cams`**
   - **Shape**: `()` (scalar)
   - **Dtype**: `int64`
   - **Wert**: `4` (Anzahl Kameras)

5. **`info/n_particles`**
   - **Shape**: `()` (scalar)
   - **Dtype**: `int64`
   - **Wert**: `1965` (Anzahl Partikel im Rope)

6. **`info/timestamp`**
   - **Shape**: `()` (scalar)
   - **Dtype**: `int64`
   - **Wert**: Timestep-Nummer (1, 2, 3, ...)

#### Observations-Gruppe (`observations/`)

##### Color Images (`observations/color/`)

7-10. **`observations/color/cam_0` bis `cam_3`**
   - **Shape**: `(1, 224, 224, 3)`
   - **Dtype**: `float32`
   - **Bedeutung**: RGB-Bilder von 4 Kameras
   - **Wertebereich**: `0-255` (als float32)
   - **Format**: `(batch, height, width, channels)`

##### Depth Images (`observations/depth/`)

11-14. **`observations/depth/cam_0` bis `cam_3`**
   - **Shape**: `(1, 224, 224)`
   - **Dtype**: `uint16`
   - **Bedeutung**: Tiefenbilder von 4 Kameras
   - **Werte**: Tiefenwerte in Millimetern (typisch: 10000-24000)

## Wichtige Erkenntnisse

### Partikel-Anzahl
- **NICHT 20 Partikel!** Der Rope hat **1965 Partikel** (`info/n_particles = 1965`)
- Die 20 Kettenglieder sind wahrscheinlich visuelle/konzeptionelle Einheiten
- Jedes "Kettenglied" besteht aus vielen Partikeln für die Physik-Simulation

### Timesteps
- **21 Timesteps** pro Episode (`00.h5` bis `20.h5`)
- Jeder Timestep hat eine eigene H5-Datei

### Kameras
- **4 Kameras** werden verwendet
- Jede Kamera liefert:
  - Color: `(224, 224, 3)` RGB
  - Depth: `(224, 224)` uint16

## Anpassung für 2 Würfel

### 1. Partikel-Anzahl ändern

**Aktuell (Rope):**
- `positions`: `(1, 1965, 4)` - 1965 Partikel
- `info/n_particles`: `1965`

**Für 2 Würfel:**
- **Option A**: Rigid Bodies (keine Partikel)
  - `positions`: `(1, 2, 7)` - 2 Würfel, jeweils `(x, y, z, qw, qx, qy, qz)` = 7 Werte
  - `info/n_particles`: `2` oder `0` (je nach Interpretation)

- **Option B**: Partikel-basierte Würfel (wenn Würfel deformierbar sind)
  - `positions`: `(1, N, 4)` - N Partikel pro Würfel
  - `info/n_particles`: `N` (z.B. 8 Ecken × 2 Würfel = 16, oder mehr für Volumen)

### 2. Property Parameters anpassen

**Aktuell (Rope):**
```python
{
    'particle_radius': 0.03,
    'length': 2.5,
    'thickness': 3.0,
    'dynamic_friction': 0.1,
    'cluster_spacing': 7.2,
    'global_stiffness': 0.00022,
    'stiffness': 0.8
}
```

**Für 2 Würfel:**
```python
{
    'cube_size': 0.1,              # Größe der Würfel
    'mass': 0.1,                    # Masse pro Würfel
    'dynamic_friction': 0.5,        # Reibung (höher für Würfel)
    'restitution': 0.3,            # Elastizität
    # Entferne: particle_radius, length, thickness, cluster_spacing, stiffness
}
```

### 3. Positions-Format

**Für Rigid Body Würfel:**
- Statt `(x, y, z, w)` homogene Koordinaten
- Verwende `(x, y, z, qw, qx, qy, qz)` für Position + Rotation
- Shape: `(1, 2, 7)` für 2 Würfel

### 4. Observations

- **Unverändert**: `obses.pth` und `observations/color/`, `observations/depth/`
- Die visuelle Darstellung zeigt jetzt Würfel statt Rope
- Bildgröße und Format bleiben gleich: `(224, 224, 3)` für Color, `(224, 224)` für Depth

### 5. Actions & EEF States

- **Unverändert**: `action` und `eef_states` bleiben gleich
- Der Roboter greift jetzt Würfel statt Rope

## Beispiel-Code zum Lesen

```python
import h5py
import torch
import numpy as np
import pickle
from pathlib import Path

episode_path = Path("deformable/rope/000001")

# Lade obses.pth
obses = torch.load(episode_path / "obses.pth")
print(f"Observations: {obses.shape}")

# Lade property_params.pkl
with open(episode_path / "property_params.pkl", "rb") as f:
    params = pickle.load(f)
print(f"Parameters: {params}")

# Lade H5-Datei für Timestep 0
with h5py.File(episode_path / "00.h5", "r") as f:
    action = f["action"][:]
    positions = f["positions"][:]
    n_particles = f["info/n_particles"][()]
    eef_states = f["eef_states"][:]
    
    # Kamera-Bilder
    color_cam0 = f["observations/color/cam_0"][:]
    depth_cam0 = f["observations/depth/cam_0"][:]
    
    print(f"Action: {action}")
    print(f"Positions: {positions.shape}")
    print(f"Particles: {n_particles}")
    print(f"EEF States: {eef_states.shape}")
    print(f"Color Image: {color_cam0.shape}")
    print(f"Depth Image: {depth_cam0.shape}")
```

## Zusammenfassung der kritischen Werte

| Parameter | Rope Wert | Für 2 Würfel |
|-----------|-----------|--------------|
| `n_particles` | 1965 | 2 (rigid) oder 8-16 (partikel-basiert) |
| `positions` shape | `(1, 1965, 4)` | `(1, 2, 7)` (rigid) oder `(1, N, 4)` (partikel) |
| `property_params` | Rope-spezifisch | Würfel-spezifisch |
| `n_timesteps` | 21 | 21 (unverändert) |
| `n_cams` | 4 | 4 (unverändert) |
| Bildgröße | `224×224` | `224×224` (unverändert) |


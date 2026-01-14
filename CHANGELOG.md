# Changelog - Data Logger Entwicklung

Diese Datei dokumentiert alle Änderungen und Entwicklungsfortschritte am Data Logger für das Franka Cube Stacking Projekt.

## [2024-XX-XX] - Integration in fcs_main_parallel.py

### ✅ Hauptskript-Anpassungen

#### 1. Logger-Initialisierung
- **Status**: ✅ Angepasst
- **Änderung**: 
  - Alte Signatur: `FrankaDataLogger(save_path=..., object_name=..., ...)`
  - Neue Signatur: `FrankaDataLogger(config=CFG, action_mode="controller")`
  - Dataset-Name wird nachträglich überschrieben (für Timestamp-Namen)
- **Zeile**: 625-631

#### 2. Kamera-Kalibrierung
- **Status**: ✅ Implementiert
- **Details**:
  - `logger.set_camera_calibration(intrinsic, extrinsic)` wird aufgerufen
  - `logger.save_camera_calibration()` am Ende
  - Verwendet `env.get_camera_matrices(camera)` für erste Kamera
- **Zeile**: 633-637, 791

#### 3. Daten-Sammlung pro Timestep
- **Status**: ✅ Vollständig angepasst
- **Neue Daten die gesammelt werden**:
  - ✅ RGB-Bilder: `camera.get_rgba()[:, :, :3]`
  - ✅ Depth-Bilder: `camera.get_current_frame()["distance_to_image_plane"]` (mit Fallbacks)
  - ✅ EE-Position: `franka.end_effector.get_world_pose()[0]` (3D)
  - ✅ EE-Quaternion: `franka.end_effector.get_world_pose()[1]` (4D)
  - ✅ Würfel-Positionen: Extrahiert aus `task.scene.get_object(cube_name)` mit Yaw-Berechnung
  - ✅ Controller-Action: Direkt übergeben
- **Zeile**: 697-740

#### 4. Episode-Buffer-Struktur
- **Status**: ✅ Angepasst
- **Neue Struktur**:
  ```python
  episode_data[i] = {
      "observations": [],      # RGB-Bilder
      "depths": [],            # Depth-Bilder (NEU)
      "ee_positions": [],      # EE-Positionen (NEU)
      "ee_quaternions": [],    # EE-Quaternionen (NEU)
      "cube_positions": [],    # Würfel-Positionen (NEU)
      "actions": [],           # Controller-Actions
      "params": {...}
  }
  ```
- **Zeile**: 664-675, 759-770

#### 5. log_step() Aufruf
- **Status**: ✅ Vollständig angepasst
- **Alter Aufruf**:
  ```python
  logger.log_step(rgb_image=obs, state=st, action=act)
  ```
- **Neuer Aufruf**:
  ```python
  logger.log_step(
      rgb_image=obs,
      depth_image=depth,
      ee_pos=ee_pos,
      ee_quat=ee_quat,
      controller_action=controller_act,
      cube_positions=cube_pos
  )
  ```
- **Zeile**: 729-740

#### 6. Yaw-Berechnung für Würfel
- **Status**: ✅ Implementiert
- **Details**:
  - Verwendet `scipy.spatial.transform.Rotation`
  - Konvertiert Quaternion → Euler → Yaw (Z-Rotation)
  - Fallback: (0, 0, 0, 0) wenn Würfel nicht gefunden
- **Zeile**: 720-730

#### 7. Depth-Bild-API
- **Status**: ✅ Mit Fallbacks implementiert
- **Details**:
  - Versucht `camera.get_current_frame()["distance_to_image_plane"]`
  - Fallback: `camera.get_depth()`
  - Fallback: Leeres Array falls beide fehlschlagen
- **Zeile**: 708-720

#### 8. Property Parameters
- **Status**: ⚠️ Auskommentiert (wie gewünscht)
- **Details**: 
  - `logger.set_episode_params()` ist auskommentiert
  - property_params.pkl wird nicht gespeichert
- **Zeile**: 726

#### 9. Zwischenspeicherung
- **Status**: ✅ Entfernt
- **Details**: 
  - `logger.save_dataset()` wurde entfernt (war für Point Maze Format)
  - Daten werden direkt bei `end_episode()` gespeichert
- **Zeile**: 778-780

### 🔧 Technische Details

#### Neue Imports
- `from pathlib import Path` - Für Pfad-Operationen
- `from scipy.spatial.transform import Rotation as R` - Für Yaw-Berechnung

#### Abhängigkeiten
- `scipy` ist jetzt erforderlich für Yaw-Berechnung
- Falls nicht verfügbar: Würfel-Yaw wird auf 0 gesetzt

### 🐛 Bekannte Probleme / Offene Punkte

1. **Depth-Bild-API**:
   - Die genaue API für Depth-Bilder in Isaac Sim könnte variieren
   - Aktuell mit mehreren Fallbacks implementiert
   - **Status**: Sollte funktionieren, aber könnte optimiert werden

2. **Yaw-Berechnung**:
   - Aktuell: Quaternion → Euler → Yaw
   - Könnte direkt aus Quaternion berechnet werden (effizienter)
   - **Status**: Funktioniert, aber könnte optimiert werden

3. **Parallele Umgebungen**:
   - Aktuell: Nur erste Kamera wird für Kalibrierung verwendet
   - Alle Umgebungen verwenden die gleiche Kalibrierung
   - **Status**: Funktioniert, aber könnte pro Env unterschiedlich sein

### 📝 Nächste Schritte

- [ ] Testen mit echten Daten aus Isaac Sim
- [ ] Validierung der Depth-Bild-API
- [ ] Optimierung der Yaw-Berechnung
- [ ] Optional: Pro-Env Kamera-Kalibrierung

---

## [2024-XX-XX] - Initiale Anpassung für Rope-Format

### 🎯 Ziel
Anpassung des Data Loggers von Point Maze Format auf Rope/D deformable Format für Kompatibilität mit bestehenden Datensätzen.

### ✅ Implementierte Features

#### 1. Config-Loading aus YAML
- **Status**: ✅ Implementiert
- **Details**: 
  - `load_config_from_yaml()` Funktion hinzugefügt
  - Lädt Konfiguration aus `config.yaml` im gleichen Verzeichnis
  - Extrahiert alle relevanten Parameter (Kamera, Würfel, Dataset-Pfade, etc.)
- **Verwendung**: 
  ```python
  config = load_config_from_yaml()
  logger = FrankaDataLogger(config=config)
  ```

#### 2. H5-Dateien pro Timestep
- **Status**: ✅ Implementiert
- **Details**:
  - Jeder Timestep wird als separate H5-Datei gespeichert (`00.h5`, `01.h5`, ...)
  - Dateien werden im Episode-Ordner gespeichert (z.B. `000001/00.h5`)
  - Struktur kompatibel mit Rope-Format
- **Format**: 
  ```
  000001/
  ├── obses.pth
  ├── 00.h5
  ├── 01.h5
  └── ...
  ```

#### 3. Action-Modi
- **Status**: ✅ Implementiert
- **Details**:
  - **Mode 1: "controller"** (Standard)
    - Verwendet die vom Controller übergebene Action
    - Extrahiert Joint-Positions oder Joint-Velocities
    - Format: `(4,) float64`
  - **Mode 2: "ee_velocity"**
    - Berechnet Endeffektor-Position + Velocity
    - Format: `[x, y, z, velocity_magnitude]` (4,) float64
- **Verwendung**:
  ```python
  logger = FrankaDataLogger(config=config, action_mode="controller")
  # oder
  logger = FrankaDataLogger(config=config, action_mode="ee_velocity")
  ```

#### 4. Endeffektor States (eef_states)
- **Status**: ✅ Implementiert
- **Details**:
  - Format: `(1, 1, 14) float64`
  - Struktur: `[[[x, y, z, x, y, z, qw, qx, qy, qz, qw, qx, qy, qz]]]`
  - Enthält: Position (2x dupliziert) + Quaternion (2x dupliziert)
  - Kompatibel mit Rope-Format
- **Speicherung**: In jeder H5-Datei als Dataset `eef_states`

#### 5. Würfel-Positionen (positions)
- **Status**: ✅ Implementiert
- **Details**:
  - Format: `(1, n_cubes, 4) float32`
  - Für jeden Würfel: `(x, y, z, yaw)`
  - Anzahl Würfel aus Config geladen (`config["cubes"]["count"]`)
  - Standard: 2 Würfel → `(1, 2, 4)`
- **Speicherung**: In jeder H5-Datei als Dataset `positions`

#### 6. Info-Gruppen in H5-Dateien
- **Status**: ✅ Implementiert
- **Details**:
  - `info/n_cams`: `int64` - Anzahl Kameras (1)
  - `info/n_cubes`: `int64` - Anzahl Würfel (2)
  - `info/timestamp`: `int64` - Timestep-Nummer
- **Struktur**:
  ```python
  info/
  ├── n_cams: 1
  ├── n_cubes: 2
  └── timestamp: 0, 1, 2, ...
  ```

#### 7. Observations in H5-Dateien
- **Status**: ✅ Implementiert
- **Details**:
  - **Color Images**: 
    - Pfad: `observations/color/cam_0`
    - Format: `(1, H, W, 3) float32`
    - Wertebereich: `0-255` (als float32)
    - Auflösung: `256×256` (aus Config)
  - **Depth Images**:
    - Pfad: `observations/depth/cam_0`
    - Format: `(1, H, W) uint16`
    - Werte: Tiefenwerte in Millimetern
    - Konvertierung: float32 → uint16 (×1000 für mm)
- **Struktur**:
  ```python
  observations/
  ├── color/
  │   └── cam_0: (1, 256, 256, 3) float32
  └── depth/
      └── cam_0: (1, 256, 256) uint16
  ```

#### 8. obses.pth Format
- **Status**: ✅ Implementiert
- **Details**:
  - Format: `(T, H, W, C) uint8`
  - Enthält alle RGB-Bilder einer Episode
  - Gespeichert im Episode-Ordner: `000001/obses.pth`
  - Kompatibel mit Rope-Format
- **Beispiel**: Bei 950 Timesteps → `(950, 256, 256, 3) uint8`

#### 9. Kamera-Kalibrierung
- **Status**: ✅ Implementiert
- **Details**:
  - **Intrinsic**: `(4, 4) float64` - Intrinsische Parameter
  - **Extrinsic**: `(4, 4, 4) float64` - Extrinsische Parameter (4x für Kompatibilität, obwohl nur 1 Kamera)
  - Gespeichert in: `cameras/intrinsic.npy` und `cameras/extrinsic.npy`
  - Kamera-Parameter aus Config geladen

#### 10. Property Parameters
- **Status**: ⚠️ Weggelassen (wie gewünscht)
- **Details**: 
  - `property_params.pkl` wird NICHT gespeichert
  - Kann später hinzugefügt werden falls benötigt

### 📋 Datenstruktur

#### Episode-Ordner
```
dataset_name/
├── cameras/
│   ├── intrinsic.npy      # (4, 4) float64
│   └── extrinsic.npy      # (4, 4, 4) float64
└── 000001/                 # Episode 1
    ├── obses.pth          # (T, 256, 256, 3) uint8
    ├── 00.h5              # Timestep 0
    ├── 01.h5              # Timestep 1
    └── ...
```

#### H5-Datei-Struktur (pro Timestep)
```python
00.h5
├── action: (4,) float64
├── eef_states: (1, 1, 14) float64
├── positions: (1, 2, 4) float32
├── info/
│   ├── n_cams: int64 (1)
│   ├── n_cubes: int64 (2)
│   └── timestamp: int64
└── observations/
    ├── color/
    │   └── cam_0: (1, 256, 256, 3) float32
    └── depth/
        └── cam_0: (1, 256, 256) uint16
```

### 🔧 Technische Details

#### Abhängigkeiten
- `torch`: Für `obses.pth` Speicherung
- `h5py`: **Erforderlich** für H5-Dateien
- `numpy`: Für alle Array-Operationen
- `yaml`: Für Config-Loading
- `PIL`: Optional für Bild-Resizing

#### Config-Parameter verwendet
- `dataset.path`: Speicherpfad
- `dataset.name`: Datensatz-Name
- `camera.resolution`: Bildauflösung `[256, 256]`
- `camera.position`: Kamera-Position
- `camera.euler`: Kamera-Orientierung
- `cubes.count`: Anzahl Würfel (2)

### 🐛 Bekannte Probleme / Offene Punkte

1. **Extrinsic-Format**: 
   - Aktuell: `(4, 4, 4)` für 1 Kamera (4x dupliziert für Kompatibilität)
   - Möglicherweise sollte es `(1, 4, 4)` sein
   - **Status**: Funktioniert, aber Format könnte optimiert werden

2. **EE-Velocity-Berechnung**:
   - Aktuell: Delta-Position (ohne dt)
   - Sollte eigentlich durch dt geteilt werden
   - **Status**: Funktioniert, aber könnte präziser sein

3. **Action-Extraktion**:
   - Aktuell: Nimmt erste 4 Joints
   - Könnte spezifischer sein je nach Controller-Typ
   - **Status**: Funktioniert, aber könnte verbessert werden

4. **Timestep-Limit**:
   - User erwähnt ~950 Timesteps, möglicherweise zu reduzieren
   - **Status**: Kein Limit implementiert, kann in Config hinzugefügt werden

### 📝 Nächste Schritte

- [ ] Testen mit echten Daten aus Isaac Sim
- [ ] Validierung der H5-Struktur gegen Rope-Datensatz
- [ ] Optimierung der Action-Extraktion
- [ ] Präzisere Velocity-Berechnung (mit dt)
- [ ] Optional: property_params.pkl wieder hinzufügen falls benötigt
- [ ] Dokumentation der Verwendung im Haupt-Skript

### 🔄 Migration von Point Maze Format

#### Alte Struktur (Point Maze):
```
dataset/
├── states.pth
├── actions.pth
├── seq_lengths.pth
└── obses/
    └── episode_XXX.pth
```

#### Neue Struktur (Rope):
```
dataset/
├── cameras/
│   ├── intrinsic.npy
│   └── extrinsic.npy
└── 000001/
    ├── obses.pth
    └── XX.h5 (pro Timestep)
```

### 📊 Vergleich mit Rope-Format

| Feature | Rope-Format | Unser Format | Status |
|---------|-------------|--------------|--------|
| obses.pth | ✅ (T, H, W, C) | ✅ (T, H, W, C) | ✅ Kompatibel |
| H5 pro Timestep | ✅ | ✅ | ✅ Kompatibel |
| action | ✅ (4,) | ✅ (4,) | ✅ Kompatibel |
| eef_states | ✅ (1, 1, 14) | ✅ (1, 1, 14) | ✅ Kompatibel |
| positions | ✅ (1, 1965, 4) | ✅ (1, 2, 4) | ✅ Kompatibel (2 Würfel) |
| info/n_cams | ✅ | ✅ | ✅ Kompatibel |
| info/n_particles | ✅ | ❌ (n_cubes statt n_particles) | ⚠️ Unterschiedlich |
| observations/color | ✅ | ✅ | ✅ Kompatibel |
| observations/depth | ✅ | ✅ | ✅ Kompatibel |
| cameras/intrinsic | ✅ (4, 4) | ✅ (4, 4) | ✅ Kompatibel |
| cameras/extrinsic | ✅ (4, 4, 4) | ✅ (4, 4, 4) | ✅ Kompatibel |

### 🎓 Lektionen gelernt

1. **H5-Format**: Sehr flexibel, aber Struktur muss exakt eingehalten werden
2. **Config-Loading**: YAML macht Konfiguration viel einfacher
3. **Action-Modi**: Flexibilität wichtig für verschiedene Anwendungsfälle
4. **Kompatibilität**: Rope-Format hat spezifische Anforderungen (z.B. duplizierte Werte in eef_states)

---

## Versionen

- **v1.0.0** (2024-XX-XX): Initiale Anpassung für Rope-Format


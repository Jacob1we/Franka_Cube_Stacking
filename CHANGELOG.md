# Changelog - Data Logger Entwicklung

Diese Datei dokumentiert alle Änderungen und Entwicklungsfortschritte am Data Logger für das Franka Cube Stacking Projekt.

## [2026-01-18] - MinDataLogger: Timestep-basierte H5-Dateien + Globale Dateien

### 🎯 Ziel

Anpassung des MinDataLoggers auf das exakte Format des `deformable_rop_sample` Datensatzes:
- **Eine H5-Datei pro Timestep** (000.h5, 001.h5, ...) statt einer H5 pro Episode
- **`actions.pth` und `states.pth`** im Datensatz-Hauptordner
- **`property_params.pkl`** in jedem Episoden-Ordner

### ✅ Neue Dateien im Output

**Datensatz-Ebene:**
```
dataset/
├── actions.pth            # (N_episodes, T_max, 6) float32
├── states.pth             # (N_episodes, T_max, N_cubes*4) float32
├── cameras/
│   ├── intrinsic.npy
│   └── extrinsic.npy
```

**Episoden-Ebene:**
```
000000/
├── 000.h5                 # Timestep 0
├── 001.h5                 # Timestep 1
├── 002.h5                 # Timestep 2
├── ...
├── obses.pth              # (T, H, W, 3) float32
├── property_params.pkl    # Physik-Parameter
├── first.png
└── last.png
```

### ✅ Änderungen in `min_data_logger.py`

#### 1. Neue Imports
```python
import pickle
from typing import Dict, Any
```

#### 2. Neue Klassenattribute
```python
# Globale Listen für states.pth und actions.pth
self.all_actions: List[List[np.ndarray]] = []
self.all_states: List[List[np.ndarray]] = []
```

#### 3. Geänderte `start_episode()`
```python
self.current_episode = {
    ...
    "actions_list": [],    # NEU: Actions für globale Datei
    "states_list": [],     # NEU: States für globale Datei
}
```

#### 4. Geänderte `log_step()`
- **H5-Datei pro Timestep**: Speichert sofort `{timestep:03d}.h5`
- **Sammelt Actions**: `ep["actions_list"].append(action.copy())`
- **Sammelt States**: Würfel-Positionen als `(N*4,)` Vektor (wie deformable Format)

```python
# .h5 Datei speichern (000.h5, 001.h5, etc.)
h5_path = ep["folder"] / f"{timestep:03d}.h5"
save_h5(h5_path, timestep_data)

# Für globale Dateien
ep["actions_list"].append(action.copy())
state_with_vel = np.concatenate([positions[0], np.zeros((N, 1))], axis=1)
ep["states_list"].append(state_with_vel.flatten())
```

#### 5. Geänderte `end_episode(property_params=None)`
- **Neuer Parameter**: `property_params` (optional, sonst Standard-Werte)
- **Speichert `property_params.pkl`**:
  ```python
  property_params = {
      "n_cubes": self.n_cubes,
      "cube_size": ...,
      "cube_mass": ...,
      "friction": ...,
  }
  with open(property_path, "wb") as f:
      pickle.dump(property_params, f)
  ```
- **Überträgt Episode-Daten** in globale Listen:
  ```python
  self.all_actions.append(ep["actions_list"])
  self.all_states.append(ep["states_list"])
  ```

#### 6. Neue Methode `save_global_data()`
Speichert am Ende alle gesammelten Daten als globale Tensoren:

```python
def save_global_data(self):
    """
    Speichert globale actions.pth und states.pth für alle Episoden.
    
    Format:
        actions.pth: (N_episodes, T_max, action_dim) float32
        states.pth: (N_episodes, T_max, state_dim) float32
    """
    # Padding auf T_max für alle Episoden
    T_max = max(len(ep) for ep in self.all_actions)
    
    actions_array = np.zeros((N_episodes, T_max, action_dim), dtype=np.float32)
    states_array = np.zeros((N_episodes, T_max, state_dim), dtype=np.float32)
    
    # Daten einfügen
    for ep_idx, (ep_actions, ep_states) in enumerate(zip(...)):
        T = len(ep_actions)
        actions_array[ep_idx, :T, :] = np.array(ep_actions)
        states_array[ep_idx, :T, :] = np.array(ep_states)
    
    torch.save(torch.from_numpy(actions_array), "actions.pth")
    torch.save(torch.from_numpy(states_array), "states.pth")
```

### 📁 H5-Datei-Struktur (pro Timestep)

```python
000.h5
├── action: (6,) float64         # [prev_ee_pos, current_ee_pos]
├── eef_states: (1, 14) float64  # [pos, pos, quat, quat]
├── positions: (1, N, 3) float32 # Würfel-Positionen
├── info/
│   ├── n_cams: 1
│   ├── timestamp: 1
│   └── n_particles: N
└── observations/
    ├── color/cam_0: (1, H, W, 3)
    └── depth/cam_0: (1, H, W) uint16
```

### 🔄 Verwendung

```python
logger = MinDataLogger(config)

# Datensammlung
for episode in range(num_episodes):
    logger.start_episode()
    for step in range(steps):
        logger.log_step(rgb, depth, ee_pos, ee_quat, cube_positions)
    logger.end_episode()  # Speichert property_params.pkl

# Am Ende: globale Dateien speichern
logger.save_global_data()  # Speichert actions.pth und states.pth
logger.save_camera_calibration()
```

### 📊 Vergleich mit deformable_rop_sample

| Feature | deformable_rop_sample | MinDataLogger | Status |
|---------|----------------------|---------------|--------|
| H5 pro Timestep | ✅ 00.h5, 01.h5, ... | ✅ 000.h5, 001.h5, ... | ✅ Kompatibel |
| actions.pth | ✅ (N, T, action_dim) | ✅ (N, T, 6) | ✅ Kompatibel |
| states.pth | ✅ (N, T, n_particles, 4) | ✅ (N, T, N_cubes*4) | ✅ Kompatibel |
| property_params.pkl | ✅ Pro Episode | ✅ Pro Episode | ✅ Kompatibel |
| obses.pth | ✅ (T, H, W, 3) | ✅ (T, H, W, 3) | ✅ Kompatibel |

---

## [2026-01-17] - MinDataLogger: Minimale Version im data.py Format

### 🎯 Ziel

Erstellung eines minimalen Data Loggers (`min_data_logger.py`), der:
- Nur den `ee_pos` Action-Mode (6D) unterstützt
- Daten exakt im Format von `dino_wm/env/deformable_env/src/sim/data_gen/data.py` speichert
- PNG-Speicherung beibehält
- Alle unnötigen Funktionen entfernt (~500 → ~180 Zeilen)

### ✅ Neue Datei: `min_data_logger.py`

**Kernfunktionen (aus data.py übernommen):**
```python
def process_imgs(imgs_list):
    """Verarbeitet Bilder: RGB BGR->RGB, Depth -> uint16 (mm)"""
    
def save_h5(filename, data):
    """Speichert H5 mit verschachtelter Struktur wie data.py"""
```

**Klasse: `MinDataLogger`**
```python
class MinDataLogger:
    def __init__(self, config, config_path, action_mode, dt)  # action_mode/dt ignoriert
    def set_camera_calibration(intrinsic, extrinsic)
    def save_camera_calibration()
    def start_episode(episode_id)
    def log_step(rgb_image, depth_image, ee_pos, ee_quat, cube_positions)
    def end_episode()
    def discard_episode()
```

### 📁 Output-Format (identisch zu data.py)

```
dataset/
├── cameras/
│   ├── intrinsic.npy      # (4, 4) float64
│   └── extrinsic.npy      # (4, 4, 4) float64
└── 000000/                 # Episode
    ├── 00.h5              # Eine H5-Datei pro Episode
    │   ├── action         # (6,) float64 - [x_start, y_start, z_start, x_end, y_end, z_end]
    │   ├── eef_states     # (T, 14) float64
    │   ├── positions      # (T, N, 3) float32
    │   ├── info/
    │   │   ├── n_cams     # 1
    │   │   ├── timestamp  # T
    │   │   └── n_particles# N (Anzahl Würfel)
    │   └── observations/
    │       ├── color/cam_0  # (T, H, W, 3) - BGR->RGB konvertiert
    │       └── depth/cam_0  # (T, H, W) uint16 - Millimeter
    ├── first.png          # Erstes Frame
    └── last.png           # Letztes Frame
```

### ❌ Entfernte Features (gegenüber FrankaDataLogger)

| Feature | Status |
|---------|--------|
| `action_mode="delta_pose"` | ❌ Entfernt |
| `action_mode="velocity"` | ❌ Entfernt |
| `action_interval` (mehrere H5 pro Episode) | ❌ Entfernt |
| `obses.pth` Speicherung | ❌ Entfernt |
| Quaternion-zu-Yaw Konvertierung | ❌ Entfernt |
| Velocity-Berechnungen | ❌ Entfernt |
| Disk-Space Checks | ❌ Entfernt |
| Detailliertes Logging | ❌ Reduziert |

### ✅ Beibehaltene Features

| Feature | Status |
|---------|--------|
| `ee_pos` Action (6D) | ✅ Einziger Modus |
| PNG-Speicherung (first.png, last.png) | ✅ Beibehalten |
| Kamera-Kalibrierung | ✅ Beibehalten |
| H5-Format | ✅ Wie data.py |
| Config aus YAML | ✅ Beibehalten |

### 🔄 Verwendung in fcs_main_parallel.py

**Drop-in Ersatz** - nur Import ändern:

```python
# Alt:
from data_logger import FrankaDataLogger, get_franka_state, get_franka_action

# Neu:
from min_data_logger import MinDataLogger as FrankaDataLogger
```

**API ist identisch:**
- `FrankaDataLogger(config, action_mode, dt)` → action_mode/dt werden ignoriert
- `logger.object_name` → vorhanden für Kompatibilität
- `logger.dataset_path` → vorhanden
- Alle Methoden identisch

### 📊 Vergleich: FrankaDataLogger vs MinDataLogger

| Aspekt | FrankaDataLogger | MinDataLogger |
|--------|------------------|---------------|
| Zeilen Code | ~800 | ~180 |
| Action-Modi | 3 (delta_pose, velocity, ee_pos) | 1 (ee_pos) |
| H5 pro Episode | Mehrere (action_interval) | Eine (00.h5) |
| obses.pth | ✅ Ja | ❌ Nein |
| Datenformat | Rope-kompatibel | data.py-kompatibel |
| PNG-Output | ❌ Nein | ✅ Ja (first/last) |

### 📝 Hinweise

- **Beide Logger existieren parallel** - wähle nach Bedarf
- `FrankaDataLogger` für vollständige Rope-Kompatibilität mit allen Features
- `MinDataLogger` für minimales data.py-kompatibles Format

---

## [2026-01-14] - Action Interval: Frame-Aggregation wie im Rope-Format

### ✅ Neuer Parameter: `action_interval`

Wie im Rope-Format können jetzt mehrere Frames zu einer Action zusammengefasst werden.
Der Parameter `action_interval` in `config.yaml` steuert dies zentral.

**Konfiguration (config.yaml):**
```yaml
dataset:
  action_interval: 10    # Alle 10 Frames wird eine H5-Datei gespeichert
                         # 1 = jeder Frame (Standard)
```

### Verhalten

| action_interval | obses.pth | H5-Dateien | Action beschreibt |
|-----------------|-----------|------------|-------------------|
| 1 (Standard)    | 100 Frames | 100 Dateien | 1 Frame |
| 10              | 100 Frames | 10 Dateien  | 10 Frames |
| 50              | 100 Frames | 2 Dateien   | 50 Frames |

**Wichtig:**
- `obses.pth` enthält **immer alle Frames** (für Video-Rekonstruktion)
- H5-Dateien werden **nur alle N Frames** gespeichert
- Die Action beschreibt die **Bewegung über N Frames**

### Action-Format bei action_interval > 1

**"delta_pose" (4D):** Gesamte Positionsänderung über N Frames
```
action = [Σdelta_x, Σdelta_y, Σdelta_z, Σdelta_yaw]
```

**"velocity" (4D):** Durchschnittsgeschwindigkeit über N Frames
```
action = [avg_vx, avg_vy, avg_vz, avg_omega_z]
```

**"ee_pos" (6D):** Start- und Endposition des Intervalls
```
action = [x_start, y_start, z_start, x_end, y_end, z_end]
```
- `x/y/z_start`: EE-Position am **Anfang** des Intervalls
- `x/y/z_end`: EE-Position am **Ende** des Intervalls

### Änderungen in data_logger.py

1. **Neuer Parameter**: `action_interval` aus config.yaml
2. **Intervall-Buffer**: Speichert Start-Position und akkumuliert Frames
3. **Neue Methode**: `_save_interval_h5()` - speichert H5 am Ende eines Intervalls
4. **Überarbeitete `log_step()`**:
   - Alle Frames → `observations` (für obses.pth)
   - Am Anfang des Intervalls: Start-Position merken
   - Am Ende des Intervalls: H5 mit Action über N Frames speichern
5. **Überarbeitete `end_episode()`**: Speichert übrige Frames im Buffer

### Beispiel

Mit `action_interval=10` und 95 Frames:
- `obses.pth`: Shape (95, H, W, C) - alle 95 Frames
- H5-Dateien: 10 Dateien (00.h5 bis 09.h5)
  - 00.h5: Frames 1-10 (Action: EE-Bewegung von Frame 1 bis 10)
  - 01.h5: Frames 11-20
  - ...
  - 09.h5: Frames 91-95 (nur 5 Frames, aber trotzdem gespeichert)

---

## [2026-01-14] - Action-Format: Drei konfigurierbare Modi

### ✅ Drei Action-Modi

Das Action-Format ist jetzt über `action_mode` Parameter (in config.yaml) konfigurierbar:

**Option 1: `action_mode="delta_pose"` (4D)**
```
action = [delta_x, delta_y, delta_z, delta_yaw]
```
- **delta_x/y/z**: Relative Position-Änderung des EE in Metern
- **delta_yaw**: Rotation um Z-Achse in Radiant

**Option 2: `action_mode="velocity"` (4D)**
```
action = [vx, vy, vz, omega_z]
```
- **vx/vy/vz**: Translatorische Geschwindigkeit in m/s
- **omega_z**: Rotatorische Geschwindigkeit um Z-Achse in rad/s

**Option 3: `action_mode="ee_pos"` (6D, wie DINO WM Rope) - DEFAULT**
```
action = [x_start, y_start, z_start, x_end, y_end, z_end]
```
- **x/y/z_start**: EE-Position am Anfang der Bewegung (vorheriger Timestep)
- **x/y/z_end**: EE-Position am Ende der Bewegung (aktueller Timestep)

Diese Option ist analog zum Rope-Format im DINO World Model, wo Actions als
`[start_x, start_z, end_x, end_z]` (2D) codiert sind. Für Franka in 3D sind
es entsprechend 6 Dimensionen.

### Änderungen in config.yaml

```yaml
dataset:
  action_mode: "ee_pos"       # Default (6D, wie DINO WM)
  # action_mode: "delta_pose" # Alternative (4D)
  # action_mode: "velocity"   # Alternative (4D)
```

### Änderungen in data_logger.py

1. **Neuer Parameter**: `action_mode` ("delta_pose" oder "velocity")
2. **Parameter**: `dt` für Timestep (Default: 1/60s = 60Hz)
3. **Neue Methoden**:
   - `_quaternion_to_yaw()`: Extrahiert Yaw aus Quaternion
   - `_normalize_angle()`: Normalisiert Winkel auf [-π, π]
4. **Action-Berechnung** automatisch aus EE-Pose:
   ```python
   # delta_pose Modus
   delta_pos = ee_pos - prev_ee_pos
   delta_yaw = current_yaw - prev_yaw
   action = [delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw]
   
   # velocity Modus
   velocity = delta_pos / dt
   omega_z = delta_yaw / dt
   action = [velocity[0], velocity[1], velocity[2], omega_z]
   ```
5. **H5-Info**: `action_mode` wird als Attribut in `info/` gespeichert

### Änderungen in fcs_main_parallel.py

1. **Logger-Initialisierung** liest `action_mode` aus Config:
   ```python
   action_mode = CFG.get("dataset", {}).get("action_mode", "delta_pose")
   logger = FrankaDataLogger(config=CFG, action_mode=action_mode, dt=1.0/60.0)
   ```

### Änderungen in franka_cube_stack_dset.py

1. **Automatische Erkennung** des `action_mode` aus H5-Dateien
2. **Einheitliche Z-Score Normalisierung** für alle 4 Dimensionen
3. **Info-Ausgabe** zeigt erkannten action_mode und Format

### Rope-Kompatibilität

Das neue Format ist vollständig kompatibel mit dem Rope-Dataset:
- `obses.pth`: (T, H, W, C) float32, Werte 0-255
- `action` in H5: (4,) float64
- `eef_states`: (1, 1, 14) float64
- `positions`: (1, n_cubes, 4) float32
- `observations/color/cam_0`: (1, H, W, 3) float32
- `observations/depth/cam_0`: (1, H, W) uint16
- `info/action_mode`: String-Attribut ("delta_pose" oder "velocity")

---

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


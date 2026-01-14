# FrankaDataLogger - Dokumentation

## Übersicht

Der `FrankaDataLogger` ist ein Datensammlungs-Tool für Isaac Sim, das Roboter-Trajektorien im **ROPE/DEFORMABLE Dataset Format** speichert. Dieses Format ist kompatibel mit dem DINO World Model Training.

---

## 1. Ausgabe-Datenstruktur

```
dataset/
├── 000000/                    # Episode 0
│   ├── obses.pth              # (T, H, W, C) float32 - Alle RGB Bilder
│   ├── property_params.pkl    # Optional: Episode-Parameter
│   ├── 00.h5                  # Timestep 0
│   ├── 01.h5                  # Timestep 1
│   ├── ...
│   └── XX.h5                  # Timestep T-1
├── 000001/                    # Episode 1
│   └── ...
├── ...
└── metadata.pkl               # Optional: Datensatz-Statistiken
```

### H5-Datei Struktur (pro Timestep)

```
XX.h5
├── action              # (action_dim,) float64 - Aktionsvektor
├── eef_states          # (1, 1, 14) float64 - End-Effector State
├── info/
│   ├── n_cams          # int64 - Anzahl Kameras
│   └── timestamp       # int64 - Timestep Index
├── observations/
│   ├── color/
│   │   └── cam_0       # (1, H, W, 3) float32 - RGB Bild
│   └── depth/          # Optional
│       └── cam_0       # (1, H, W) uint16 - Tiefenbild
└── positions           # Optional: (1, N, 4) float32 - Objekt-Positionen
```

---

## 2. Chronologischer Ablauf

### Phase 1: Initialisierung

```
┌─────────────────────────────────────────────────────────────────┐
│  FrankaDataLogger.__init__()                                    │
├─────────────────────────────────────────────────────────────────┤
│  1. Parameter setzen:                                           │
│     - save_path: Basis-Verzeichnis                              │
│     - object_name: Datensatz-Name (Unterordner)                 │
│     - image_size: (H, W) für Bilder                             │
│     - max_timesteps: Optional Limit pro Episode                 │
│     - save_png: PNG-Preview aktivieren                          │
│     - n_cams: Anzahl Kameras                                    │
│                                                                 │
│  2. Interne Variablen initialisieren:                           │
│     - current_episode = None                                    │
│     - episode_count = 0                                         │
│     - all_episode_lengths = []                                  │
│                                                                 │
│  3. Verzeichnis erstellen:                                      │
│     - dataset_path.mkdir(parents=True, exist_ok=True)           │
│                                                                 │
│  4. h5py-Check:                                                 │
│     - Fehler wenn h5py nicht verfügbar                          │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Episode starten

```
┌─────────────────────────────────────────────────────────────────┐
│  logger.start_episode(episode_id=None)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Episode-ID setzen:                                          │
│     - Falls episode_id gegeben → episode_count = episode_id     │
│     - Sonst → auto-increment                                    │
│                                                                 │
│  2. Episode-Dictionary initialisieren:                          │
│     current_episode = {                                         │
│         "id": episode_count,                                    │
│         "timestep": 0,                                          │
│         "observations": [],    # RGB Bilder                     │
│         "depth_images": [],    # Tiefenbilder (optional)        │
│         "actions": [],         # Aktionsvektoren                │
│         "eef_states": [],      # End-Effector States            │
│         "positions": [],       # Objekt-Positionen (optional)   │
│         "property_params": {}  # Episode-Parameter              │
│     }                                                           │
│                                                                 │
│  3. Log-Ausgabe: "Episode X gestartet"                          │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Timestep loggen (wird N mal aufgerufen)

```
┌─────────────────────────────────────────────────────────────────┐
│  logger.log_step(rgb_image, action, state/eef_state, ...)       │
├─────────────────────────────────────────────────────────────────┤
│  1. Validierung:                                                │
│     - Prüfe ob Episode gestartet                                │
│     - Prüfe max_timesteps Limit                                 │
│                                                                 │
│  2. State-Parameter auflösen:                                   │
│     - eef_state ODER state akzeptieren (Rückwärtskompatibel)    │
│     - Fallback: zeros(14) wenn nichts übergeben                 │
│                                                                 │
│  3. RGB-Bild normalisieren:                                     │
│     - Wenn max > 1.0: bereits uint8 → als float32 speichern     │
│     - Wenn max <= 1.0: [0,1] → [0,255] skalieren                │
│                                                                 │
│  4. Daten an Listen anhängen:                                   │
│     - observations.append(rgb_float)                            │
│     - actions.append(action.float64)                            │
│     - eef_states.append(eef_state.float64)                      │
│     - Optional: depth_images, positions                         │
│                                                                 │
│  5. Timestep-Counter erhöhen:                                   │
│     - timestep += 1                                             │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 4: Episode beenden

```
┌─────────────────────────────────────────────────────────────────┐
│  logger.end_episode()                                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Validierung:                                                │
│     - Prüfe ob Episode existiert                                │
│     - Prüfe ob Episode nicht leer                               │
│                                                                 │
│  2. Episode-Ordner erstellen:                                   │
│     - Pfad: dataset_path / f"{episode_id:06d}"                  │
│     - Beispiel: dataset/000000/                                 │
│                                                                 │
│  3. obses.pth speichern:                                        │
│     - observations → np.stack → torch.Tensor                    │
│     - Shape: (T, H, W, C) float32                               │
│     - torch.save(tensor, episode_dir / "obses.pth")             │
│                                                                 │
│  4. property_params.pkl speichern (falls vorhanden):            │
│     - pickle.dump(params, file)                                 │
│                                                                 │
│  5. H5-Dateien für jeden Timestep erstellen:                    │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  for t in range(episode_length):                        │ │
│     │      h5_path = episode_dir / f"{t:02d}.h5"              │ │
│     │                                                         │ │
│     │      with h5py.File(h5_path, "w") as f:                 │ │
│     │          f["action"] = actions[t]                       │ │
│     │          f["eef_states"] = eef_states[t] → (1,1,14)     │ │
│     │          f["info/n_cams"] = n_cams                      │ │
│     │          f["info/timestamp"] = t                        │ │
│     │          f["observations/color/cam_0"] = img → (1,H,W,3)│ │
│     │          Optional: depth, positions                     │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                                                 │
│  6. Optional: PNG-Preview speichern                             │
│     - Jeden n-ten Frame als PNG                                 │
│                                                                 │
│  7. Statistiken aktualisieren:                                  │
│     - all_episode_lengths.append(episode_length)                │
│     - episode_count += 1                                        │
│     - current_episode = None                                    │
│                                                                 │
│  8. Log-Ausgabe:                                                │
│     - "Episode X gespeichert: Y Timesteps, Y H5-Dateien"        │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 5: Datensatz finalisieren (optional)

```
┌─────────────────────────────────────────────────────────────────┐
│  logger.save_dataset()                                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Metadaten erstellen:                                        │
│     metadata = {                                                │
│         "n_episodes": len(all_episode_lengths),                 │
│         "episode_lengths": all_episode_lengths,                 │
│         "total_timesteps": sum(all_episode_lengths),            │
│         "image_size": (H, W),                                   │
│         "n_cams": n_cams,                                       │
│         "created": datetime.now().isoformat(),                  │
│         "format": "rope_compatible"                             │
│     }                                                           │
│                                                                 │
│  2. metadata.pkl speichern:                                     │
│     - pickle.dump(metadata, dataset_path / "metadata.pkl")      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Sequenzdiagramm

```
┌──────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│ Main.py  │     │ DataLogger   │     │ Filesystem  │     │ Episode  │
└────┬─────┘     └──────┬───────┘     └──────┬──────┘     └────┬─────┘
     │                  │                    │                 │
     │  __init__()      │                    │                 │
     │─────────────────>│                    │                 │
     │                  │  mkdir()           │                 │
     │                  │───────────────────>│                 │
     │                  │                    │                 │
     │  start_episode() │                    │                 │
     │─────────────────>│                    │                 │
     │                  │         create episode dict          │
     │                  │─────────────────────────────────────>│
     │                  │                    │                 │
     │  log_step() x N  │                    │                 │
     │─────────────────>│                    │                 │
     │                  │              append data             │
     │                  │─────────────────────────────────────>│
     │                  │                    │                 │
     │  end_episode()   │                    │                 │
     │─────────────────>│                    │                 │
     │                  │  save obses.pth    │                 │
     │                  │───────────────────>│                 │
     │                  │  save XX.h5 x T    │                 │
     │                  │───────────────────>│                 │
     │                  │                    │                 │
     │  save_dataset()  │                    │                 │
     │─────────────────>│                    │                 │
     │                  │  save metadata.pkl │                 │
     │                  │───────────────────>│                 │
     │                  │                    │                 │
```

---

## 4. Datenformate im Detail

### Action Vector (9 Dimensionen)
```
action = [
    joint_0,    # Joint Position/Velocity
    joint_1,
    joint_2,
    joint_3,
    joint_4,
    joint_5,
    joint_6,
    gripper_0,  # Gripper Command
    gripper_1,
]
```

### EEF State (14 Dimensionen)
```
eef_state = [
    pos_x, pos_y, pos_z,           # Position (3)
    quat_w, quat_x, quat_y, quat_z, # Orientierung (4)
    vel_x, vel_y, vel_z,           # Lineare Velocity (3)
    ang_x, ang_y, ang_z, _pad,     # Angular Velocity (4)
]
```

### Franka State (22 Dimensionen) - Alternative
```
state = [
    ee_pos (3),          # End-Effector Position
    ee_quat (4),         # End-Effector Quaternion
    gripper_opening (1), # Gripper Öffnung [0-1]
    joint_pos (7),       # Joint Positionen
    joint_vel (7),       # Joint Velocities
]
```

---

## 5. Hilfsfunktionen

### `get_franka_state(franka, task)`
Extrahiert 22-dim State-Vektor vom Franka Roboter.

### `get_franka_eef_state(franka)`
Extrahiert 14-dim EEF-State für Rope-Format.

### `get_franka_action(controller_action)`
Extrahiert 9-dim Action-Vektor aus Controller-Aktion.

---

## 6. Verwendungsbeispiel

```python
from data_logger import FrankaDataLogger, get_franka_state, get_franka_action

# 1. Logger initialisieren
logger = FrankaDataLogger(
    save_path="/path/to/datasets",
    object_name="my_dataset",
    image_size=(256, 256),
    save_png=True,
)

# 2. Pro Episode
for episode in range(num_episodes):
    logger.start_episode()
    
    # Optional: Episode-Parameter setzen
    logger.set_episode_params({"difficulty": "hard"})
    
    # 3. Pro Timestep
    for step in range(max_steps):
        # Daten sammeln
        rgb = camera.get_rgb()
        state = get_franka_state(franka, task)
        action = get_franka_action(controller_action)
        
        # Loggen
        logger.log_step(
            rgb_image=rgb,
            action=action,
            state=state,  # oder eef_state=...
        )
        
        if done:
            break
    
    # 4. Episode beenden
    if success:
        logger.end_episode()
    else:
        logger.discard_episode()

# 5. Metadaten speichern
logger.save_dataset()
```

---

## 7. Abhängigkeiten

| Paket | Version | Zweck |
|-------|---------|-------|
| torch | ≥1.9 | Tensor-Speicherung (.pth) |
| numpy | <2.0 | Array-Operationen |
| h5py | ≥3.0 | HDF5-Dateien |
| PIL | Optional | PNG-Export |

**Wichtig:** numpy 2.x ist inkompatibel mit Isaac Sim!

---

# Entwicklungslog

## Changelog / Commit-Historie

### 2026-01-14: Rope-Format Implementation

**Commit: `feat(dataset): Rope-Format für Franka Cube Stack Pipeline`**

- ✅ Komplette Neuimplementierung im ROPE/DEFORMABLE Format
- ✅ Episode-Ordner-Struktur: `000000/`, `000001/`, ...
- ✅ `obses.pth`: (T, H, W, C) float32 RGB-Bilder
- ✅ H5-Dateien pro Timestep mit vollständiger Struktur
- ✅ Neue Funktion `get_franka_eef_state()` für 14-dim EEF State
- ✅ Rückwärtskompatibilität: `state` Parameter als Alias für `eef_state`

### 2026-01-14: h5py Installation & numpy Fix

**Problem:** h5py Installation hat numpy auf 2.x aktualisiert → Isaac Sim Crash

**Lösung:**
```bash
pip install 'numpy<2.0' 'h5py' --force-reinstall
```

**Ergebnis:**
- numpy: 1.26.4 (kompatibel)
- h5py: 3.15.1

### 2026-01-14: Parameter-Kompatibilität

**Problem:** `TypeError: log_step() got an unexpected keyword argument 'state'`

**Lösung:** `state` als Alias für `eef_state` hinzugefügt:
```python
def log_step(self, ..., eef_state=None, state=None, ...):
    if eef_state is None and state is not None:
        eef_state = state
```

### 2026-01-14: Bildgröße angepasst

**Änderung:** Default `image_size` von (224, 224) auf (256, 256) geändert
- Passend zu `config.yaml: resolution: [256, 256]`
- Training resized auf 224x224 (DINOv2 Standard)

---

## Bekannte Issues

| Issue | Status | Workaround |
|-------|--------|------------|
| h5py upgradet numpy | ✅ Gelöst | `numpy<2.0` forcieren |
| `state` vs `eef_state` | ✅ Gelöst | Beide Parameter akzeptiert |
| PNG-Export langsam | ⚠️ Offen | `save_png=False` nutzen |

---

## Geplante Features

- [ ] Multi-Kamera Support (cam_0, cam_1, ...)
- [ ] Kompression für H5-Dateien
- [ ] Streaming-Mode (direktes Schreiben ohne RAM-Puffer)
- [ ] Validierungs-Tool für Datensatz-Integrität



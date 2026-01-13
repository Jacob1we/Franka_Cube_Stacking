# Franka Cube Stacking – Data Collection Environment

> **Isaac Sim Umgebung für Franka Panda Roboter mit parallelisierter Datensammlung für DINO-WM World Model Training**

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.x-green)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-orange)

## Überblick

Diese Umgebung ermöglicht die automatisierte Datensammlung für das Training von DINO-WM World Models. Ein Franka Panda Roboter stapelt Würfel in einer physikalisch simulierten Umgebung, während Kamerabilder, Roboter-Zustände und Aktionen aufgezeichnet werden.

### Hauptmerkmale

- **Parallelisierung**: Bis zu N Umgebungen gleichzeitig für schnellere Datensammlung
- **Domain Randomization**: Zufällige Würfelpositionen, Materialien, Beleuchtung
- **RMPFlow Controller**: Intelligente Trajektorienplanung mit Joint-Präferenzen
- **Validierung**: Automatische Prüfung erfolgreicher Stacking-Episoden
- **DINO-WM kompatibel**: Datensatzformat direkt für World Model Training verwendbar

---

## Projektstruktur

```
Franka_Cube_Stacking_JW/
├── fcs_main_parallel.py      # Hauptskript (Single & Parallel Mode)
├── config.yaml               # Zentrale Konfigurationsdatei
├── data_logger.py            # FrankaDataLogger für DINO-WM Format
├── Franka_Env_JW/            # Eigene Isaac Sim Module
│   ├── __init__.py
│   ├── stacking_jw.py              # Stacking Task Definition
│   ├── stacking_controller_jw.py   # Enhanced Stacking Controller
│   ├── pick_place_controller_jw.py # Pick & Place Logik
│   ├── rmpflow_controller_jw.py    # RMPFlow mit Joint-Präferenzen
│   ├── franka_jw.py                # Franka Roboter Wrapper
│   ├── base_stacking_jw.py         # Basis-Stacking Task
│   └── base_task_jw.py             # Basis-Task Klasse
├── dataset/                  # Lokaler Datensatz-Output
└── 00_Archiv/                # Archivierte Entwicklungsversionen
```

---

## Voraussetzungen

### Software

- **NVIDIA Isaac Sim 4.x** (getestet mit Isaac Sim 4.2)
- **Python 3.10+** (Isaac Sim Python Environment)
- **CUDA-fähige GPU** (RTX 3000+ empfohlen)

### Python-Pakete (im Isaac Sim Environment)

```
numpy
torch
pyyaml
scipy
PIL (Pillow)
h5py (optional)
```

---

## Konfiguration

Alle Parameter werden über `config.yaml` gesteuert:

```yaml
# Simulation
simulation:
  headless: true              # GUI aus für schnellere Datensammlung
  seed: 111

# Parallelisierung
parallel:
  num_envs: 10                # Anzahl paralleler Umgebungen
  env_spacing: 2.5            # Abstand zwischen Environments (m)

# Datensammlung
dataset:
  num_episodes: 100           # Anzahl zu sammelnder Episoden
  path: "/path/to/datasets"   # Speicherort
  name: "franka_cube_stack"   # Datensatzname
  save_png: true              # PNG-Bilder speichern

# Kamera
camera:
  position: [1.6, -2.0, 1.27]
  euler: [66.0, 0.0, 32.05]
  resolution: [256, 256]

# Würfel
cubes:
  count: 2                    # Anzahl Würfel zum Stapeln
  side: 0.05                  # Kantenlänge (m)

# Roboter Workspace
robot:
  max_reach: 0.75             # Maximale Reichweite (m)
  min_reach: 0.3              # Minimale Reichweite (m)
```

---

## Verwendung

### Starten der Datensammlung

```bash
# Mit Isaac Sim Python
cd /path/to/isaacsim
./python.sh /path/to/Franka_Cube_Stacking_JW/fcs_main_parallel.py

# Mit eigener Config-Datei
./python.sh /path/to/fcs_main_parallel.py --config /path/to/custom_config.yaml
```

### Single vs. Parallel Mode

| Mode | `num_envs` | Beschreibung |
|------|------------|--------------|
| Single | 1 | Sequentielle Datensammlung, einfacher zu debuggen |
| Parallel | >1 | N Umgebungen im Grid, N-fache Geschwindigkeit |

---

## Controller-Architektur

### Stacking Controller Phasen

Der `StackingController_JW` führt einen 10-Phasen Pick-and-Place Zyklus aus:

| Phase | Beschreibung | Typ |
|-------|--------------|-----|
| 0 | Bewege EE über Würfel | AIR |
| 1 | Senke EE zum Würfel | CRITICAL |
| 2 | Warte auf Stabilisierung | WAIT |
| 3 | Schließe Greifer | GRIP |
| 4 | Hebe EE an | AIR |
| 5 | Bewege zu Ziel-XY | AIR |
| 6 | Senke zum Platzieren | CRITICAL |
| 7 | Öffne Greifer | RELEASE |
| 8 | Hebe EE an | AIR |
| 9 | Zurück zur Ausgangsposition | AIR |

### Joint-Präferenzen (Soft Constraints)

Verfügbare Presets für den RMPFlow Controller:

```python
PRESET_LOCK_WRIST_ROTATION  # Neutrale Handgelenk-Rotation
PRESET_LOCK_UPPER_ARM       # Neutraler Oberarm
PRESET_MINIMAL_MOTION       # Minimale Gesamtbewegung
PRESET_LOCK_FOREARM         # Neutraler Unterarm
PRESET_ESSENTIAL_ONLY       # Alle Rotationsgelenke neutral
```

---

## Datensatzformat

### Struktur

```
franka_cube_stack_ds/
├── states.pth              # (N, T, 22) Roboter-Zustände
├── actions.pth             # (N, T, 9)  Aktionen
├── metadata.pkl            # Datensatz-Metadaten
├── seq_lengths.pkl         # Episodenlängen
├── cameras/                # Kamera-Kalibrierung
├── 000000/                 # Episode 0
│   ├── obses.pth           # (T, H, W, 3) RGB-Bilder
│   ├── images/             # PNG-Vorschaubilder
│   └── property_params.pkl # Episode-Parameter
├── 000001/
│   └── ...
└── ...
```

### State-Vektor (22 Dimensionen)

| Index | Beschreibung |
|-------|--------------|
| 0:3 | End-Effector Position (x, y, z) |
| 3:7 | End-Effector Orientierung (Quaternion) |
| 7 | Greifer-Öffnung (0-1) |
| 8:15 | Joint-Positionen (7 DOF) |
| 15:22 | Joint-Velocities (7 DOF) |

### Action-Vektor (9 Dimensionen)

| Index | Beschreibung |
|-------|--------------|
| 0:7 | Joint-Befehle (Position/Velocity) |
| 7:9 | Greifer-Befehle |

---

## DINO-WM Integration

Die gesammelten Datensätze sind direkt kompatibel mit dem DINO-WM Training:

```bash
# Training starten (aus dino_wm2 Verzeichnis)
python train.py \
    --config-name train.yaml \
    env=franka_cube_stack \
    dataset_path=/path/to/fcs_datasets/franka_cube_stack_ds
```

### Empfohlene Trainingsparameter

```yaml
# Für Franka Cube Stacking
frameskip: 3          # Reduziert Sequenzlänge
num_hist: 3           # Historie für World Model
action_dim: 9         # 7 Joints + 2 Gripper
```

---

## Domain Randomization

Folgende Parameter werden pro Episode randomisiert:

| Parameter | Bereich | Beschreibung |
|-----------|---------|--------------|
| Würfel-Position | Workspace | Innerhalb min/max Reichweite |
| Würfel-Rotation | ±45° | Yaw-Rotation |
| Ziel-Position | Workspace | Stack-Zielposition |
| Beleuchtung | 5500-7000 lux | Intensität & Position |
| Material | 7 Presets | Tischoberfläche |

---

## Validierung

Episoden werden nur gespeichert wenn:

1. **X/Y Toleranz**: Alle Würfel innerhalb 3cm der Zielposition
2. **Z Mindesthöhe**: Alle Würfel über 2cm (nicht durchgefallen)
3. **Stacking**: Würfel sind korrekt übereinander gestapelt

Fehlgeschlagene Seeds werden in `failed_seeds.txt` protokolliert.

---

## Troubleshooting

### Häufige Probleme

| Problem | Lösung |
|---------|--------|
| `ModuleNotFoundError: Franka_Env_JW` | Skript aus Isaac Sim Python starten |
| Würfel außerhalb Reichweite | `robot.max_reach` in Config anpassen |
| Langsame Datensammlung | `headless: true` und `num_envs` erhöhen |
| Speicher voll | `save_png: false` oder weniger Episoden |

### Logging

Logs werden nach `data_collection.log` geschrieben:

```bash
tail -f data_collection.log
```

---

## Lizenz

Apache License 2.0 – basiert auf NVIDIA Isaac Sim Beispielen.

---

## Autor

**JW** – Entwickelt für DINO-WM World Model Experimente mit Franka Panda Roboter.

---

## Referenzen

- [DINO-WM Paper](https://arxiv.org/abs/2411.04983)
- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Franka Emika Panda](https://www.franka.de/)


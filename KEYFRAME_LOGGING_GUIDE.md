# Keyframe-basiertes Logging - Integration Guide

## 🎯 Problem

Bei naivem Subsampling (action_interval) gehen kritische Punkte verloren:

```
Dichte Trajektorie:  ●───●───●───●───🔽───●───●───●───●
                     AIR      AIR   PICK  AIR      AIR
                                    ↑
                              Tiefster Punkt!

Nach naivem Subsampling (interval=5):
                     ●───────────────●───────────────●
                     AIR             ???             AIR
                     ❌ PICK-POINT VERLOREN!
```

## ✅ Lösung: KeyframeDataLogger

Der neue `keyframe_data_logger.py` garantiert, dass kritische Punkte IMMER im Datensatz sind:

```
Keyframe-basiert:    ●───────────🔽───●───────────●
                     AIR         PICK AIR         END
                     ✅ Kritische Punkte IMMER dabei!
```

### Keyframe-Regeln

| Bedingung | Beschreibung | Priorität |
|-----------|--------------|-----------|
| **Phase-Wechsel** | Ende von Phase 1 (PICK) oder Phase 6 (PLACE) | ★★★ Höchste |
| **Z-Minimum** | Lokales Minimum der EE-Z-Koordinate | ★★★ Höchste |
| **Greifer-Aktion** | Phase 3 (schließen) oder Phase 7 (öffnen) | ★★ Hoch |
| **Sparse Interval** | Nach N Frames ohne Keyframe | ★ Normal |

---

## 📝 Integration in fcs_main_parallel.py

### 1. Import ändern

```python
# ALT:
from min_data_logger import MinDataLogger as FrankaDataLogger

# NEU:
from keyframe_data_logger import KeyframeDataLogger as FrankaDataLogger
```

### 2. Logger initialisieren

```python
# ALT:
logger = FrankaDataLogger(
    config=CFG,
    dataset_path=dataset_path,
)

# NEU:
logger = FrankaDataLogger(
    config=CFG,
    dataset_path=dataset_path,
    sparse_interval=10,      # Frames zwischen Keyframes in AIR-Phasen
    min_z_threshold=0.001,   # Mindest-Z-Änderung für lokales Minimum
)
```

### 3. log_step erweitern

Der wichtigste Teil: **Phase-Information mitgeben!**

```python
# ALT:
logger.log_step(
    rgb_image=obs,
    depth_image=depth,
    ee_pos=ee_pos,
    ee_quat=ee_quat,
    cube_positions=cube_pos
)

# NEU:
# Phase vom Controller abfragen
current_phase = controller._pick_place_ctrl.get_current_event()

logger.log_step(
    rgb_image=obs,
    depth_image=depth,
    ee_pos=ee_pos,
    ee_quat=ee_quat,
    cube_positions=cube_pos,
    phase=current_phase,  # ★ NEU: Phase für Keyframe-Entscheidung
)
```

---

## 📊 Erwartete Ergebnisse

### Beispiel: 1 Würfel stapeln

| Metrik | MinDataLogger | KeyframeDataLogger |
|--------|---------------|-------------------|
| Total Frames | 150 | 150 |
| Gespeicherte Frames | 150 | **~20-25** |
| H5-Dateien | 150 | **~20-25** |
| Kompression | 1x | **6-7x** |
| Pick-Point erhalten? | ❌ Bei interval>1 | ✅ IMMER |
| Place-Point erhalten? | ❌ Bei interval>1 | ✅ IMMER |

### Keyframe-Verteilung (typisch)

```
Episode beendet:
  Total Frames: 150 → Keyframes: 23 (6.5x Kompression)
  Keyframe-Verteilung:
    - Phase-Wechsel: 10  (Phasen 0→1, 1→2, 2→3, ...)
    - Z-Minima: 2        (Pick-Point, Place-Point)
    - Sparse: 11         (alle 10 Frames in AIR-Phasen)
```

---

## 🔧 Konfiguration

### Parameter in config.yaml (optional)

```yaml
dataset:
  # Keyframe-Einstellungen
  sparse_interval: 10        # Frames zwischen Keyframes (AIR-Phasen)
  min_z_threshold: 0.001     # m, für Z-Minima Erkennung
  save_png: false            # PNG speichern (aus für Kompression)
```

### Sparse Interval Empfehlungen

| Szenario | sparse_interval | Ergebnis |
|----------|-----------------|----------|
| Maximale Kompression | 20 | ~15 Keyframes |
| Ausgewogen | 10 | ~20-25 Keyframes |
| Feine Details | 5 | ~35-40 Keyframes |
| Wie Rope-Datensatz | 7-8 | ~21 Keyframes |

---

## 🔍 Debugging

### Keyframe-Indizes prüfen

```python
import pickle

# Nach der Episode
with open("dataset/000000/property_params.pkl", "rb") as f:
    params = pickle.load(f)

print("Keyframe Stats:", params["keyframe_stats"])
print("Keyframe Indizes:", params["keyframe_indices"])
```

### Visualisierung der Keyframes

```python
import torch
import matplotlib.pyplot as plt

obses = torch.load("dataset/000000/obses.pth")
print(f"Keyframes: {obses.shape[0]}")

fig, axes = plt.subplots(1, min(5, obses.shape[0]), figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(obses[i].numpy().astype('uint8'))
    ax.set_title(f"KF {i}")
plt.show()
```

---

## ⚠️ Wichtige Hinweise

1. **Phase muss übergeben werden!** Ohne Phase-Information funktioniert nur Z-Minima und Sparse Sampling.

2. **Kompatibilität:** Output-Format ist 100% identisch zu MinDataLogger (DINO-WM kompatibel).

3. **Speicherverbrauch:** Der Logger speichert alle Frames temporär im RAM, daher ähnlicher RAM-Verbrauch wie MinDataLogger.

4. **Debugging:** `property_params.pkl` enthält jetzt `keyframe_stats` und `keyframe_indices` für Analyse.

---

## 📁 Dateistruktur (identisch zu MinDataLogger)

```
dataset/
├── cameras/
│   ├── intrinsic.npy
│   └── extrinsic.npy
├── actions.pth              # (N_episodes, T_max, 6)
├── states.pth               # (N_episodes, T_max, N_cubes, 4)
└── 000000/
    ├── obses.pth            # (T_keyframes, H, W, 3)  ← NUR Keyframes!
    ├── property_params.pkl  # + keyframe_stats
    ├── 000.h5               # Keyframe 0
    ├── 001.h5               # Keyframe 1
    ├── ...
    ├── first.png
    └── last.png
```

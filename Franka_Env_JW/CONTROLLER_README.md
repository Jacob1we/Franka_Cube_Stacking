# Franka Cube Stacking - Controller Architektur

## 📋 Inhaltsverzeichnis

1. [Überblick](#überblick)
2. [Hierarchische Struktur](#hierarchische-struktur)
3. [Das Nullspace-Konzept](#das-nullspace-konzept)
4. [Die 3 Controller-Ebenen](#die-3-controller-ebenen)
5. [Detaillierte Funktionsweise](#detaillierte-funktionsweise)
6. [Warum diese Architektur?](#warum-diese-architektur)
7. [Praktische Verwendungsbeispiele](#praktische-verwendungsbeispiele)

---

## 🎯 Überblick

Das Controller-System ist ein **hierarchisch aufgebautes 3-Schichten-System**, das die Franka-Panda-Roboter-Arm für komplexe Stacking-Aufgaben steuert.

**Kernkonzept:** Jede Schicht abstrahiert die darunter liegende Komplexität und löst ein spezifisches Problem:

```
┌──────────────────────────────────────────────────────┐
│ Level 3: StackingController_JW                       │
│ → "Welcher Würfel ist dran?"                         │
│ → Orchestrierung mehrerer Pick&Place-Zyklen         │
└──────────────┬───────────────────────────────────────┘
               │ nutzt (Composition)
               ↓
┌──────────────────────────────────────────────────────┐
│ Level 2: PickPlaceController_JW                      │
│ → "Wie bewege ich den Arm durch die 10 Phasen?"     │
│ → State Machine mit Geschwindigkeit & Genauigkeit  │
└──────────────┬───────────────────────────────────────┘
               │ nutzt
               ↓
┌──────────────────────────────────────────────────────┐
│ Level 1: RMPFlowController_JW                        │
│ → "Welche Gelenk-Positions erreichen diesen Punkt?" │
│ → Inverse Kinematik mit Soft Constraints            │
└──────────────────────────────────────────────────────┘
```

---

## 🏗️ Hierarchische Struktur

```
Dein Roboter-Kontrollsystem
│
├─ StackingController_JW
│  ├─ Verwaltet Pick-Order: ["cube_0", "cube_1", "cube_2"]
│  ├─ Tracked Current Cube Index
│  └─ Delegiert an PickPlaceController
│     │
│     └─ PickPlaceController_JW
│        ├─ 10-Phasen State Machine
│        ├─ Berechnet Ziel-Positionen pro Phase
│        ├─ Modulation: trajectory_resolution, air_speed_multiplier
│        ├─ Height-Adaptive Speed
│        └─ Delegiert an RMPFlowController
│           │
│           └─ RMPFlowController_JW
│              ├─ Inverse Kinematik (RMPFlow)
│              ├─ Soft Constraints (Joint Preferences)
│              ├─ Null-Space Projection
│              └─ Gibt Joint-Positions zurück
```

---

## 🌌 Das Nullspace-Konzept

### Was ist der Nullspace?

Ein **Nullspace** ist ein mathematisches Konzept aus der Robotik, das die Freiheitsgrade beschreibt, die **die End-Effektor-Position nicht beeinflussen**.

### Visuelles Beispiel:

Stell dir vor, du schaust auf einen Roboterarm von oben:

```
Franka Panda mit 7 Gelenken, aber Aufgabe: "Erreiche Position (x, y, z)"
Das sind nur 3 Constraints (x, y, z).

→ Es bleiben 7 - 3 = 4 Freiheitsgrade übrig!
```

Diese 4 Freiheitsgrade können bewegt werden, **ohne die End-Effektor-Position zu ändern**:

```
Position (x,y,z) bleibt gleich ✓
Aber die Gelenke können anders angeordnet sein:

    Konfiguration A              Konfiguration B
    (arm bent forward)           (arm bent backward)
       ●●●                          ●●●
      ●   ●                        ●   ●
     ●     ●                      ●     ●
    ●       ●                    ●       ●
   [Greifer am gleichen Punkt (x,y,z)]

Die Gelenke bewegen sich im Nullspace, aber der Greifer bleibt stehen!
```

### Mathematische Erklärung:

```
Jacobian J(q) * dq = v_ee
─────────────────────────
  7×3      7×1    3×1

Wenn v_ee = 0 (Greifer bewegt sich nicht):
J(q) * dq = 0

Alle dq-Vektoren, die diese Gleichung erfüllen, sind im Nullspace!

Es gibt infinite viele Lösungen → Nullspace ist mehrdimensional
```

### Praktisch: Warum ist das wichtig?

**Im Pick&Place-Prozess:**

```
Problem: Der Roboter kann den Würfel greifen, aber:
- Die Gelenke sind ungünstig angeordnet
- Nächste Phase könnte singular werden
- Bewegung wird "knickerig"

Lösung mit Nullspace:
→ Bevorzugte Gelenkposition setzen (z.B. Gelenk 6 = 0.78 rad)
→ RMPFlow erreicht trotzdem den Zielpunkt (x, y, z)
→ Aber nutzt den Nullspace um die bevorzugte Gelenkposition anzusteuern
```

### Beispiel in unserem Code:

```python
# Soft Constraint über Nullspace
preferred_joints = {
    2: 0.0,    # Upper arm: bevorzuge neutralen Winkel
    6: 0.78,   # Wrist: bevorzuge Neutralrotation
}

# RMPFlow-Verhalten:
# 1. PRIMÄR: Erreiche Position (1.0, 0.5, 0.3)  ← Hard Constraint
# 2. SEKUNDÄR (im Nullspace): Versuche auch Gelenk 2 ≈ 0.0 zu halten
#                             und Gelenk 6 ≈ 0.78 zu halten

# Konflikt? → Position (1.0, 0.5, 0.3) GEWINNT
# Kein Konflikt? → Soft Constraints werden eingehalten
```

---

## 🔧 Die 3 Controller-Ebenen

### Level 1: RMPFlowController_JW (Unterste Ebene)

**Verantwortung:** Direkte Roboter-Kontrolle  
**Eingabe:** End-Effektor Zielposition + -orientierung  
**Ausgabe:** Gelenk-Positionen  

**Initialisierung:**

```python
from Franka_Env_JW import RMPFlowController_JW, PRESET_MINIMAL_MOTION

rmpflow_ctrl = RMPFlowController_JW(
    name="cspace_controller",
    robot_articulation=franka_robot,              # SingleArticulation Objekt
    physics_dt=1.0/60.0,                          # 60 Hz Simulator
    preferred_joints=PRESET_MINIMAL_MOTION,       # {2: 0.0, 6: 0.78}
    trajectory_scale=1.0,                         # 1.0 = Normal, 2.0 = 2x schneller
)
```

**Was macht es intern?**

```python
# 1. RMPFlow laden (NVIDIA Motion Generation Library)
rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
    "Franka", "RMPflow"
)
rmp_flow = mg.lula.motion_policies.RmpFlow(**rmp_flow_config)

# 2. Mit Roboter-Articulation verbinden
articulation_rmp = mg.ArticulationMotionPolicy(
    robot_articulation,      # Der echte Roboter
    self.rmp_flow,          # Das IK-Modell
    adjusted_physics_dt     # physics_dt * trajectory_scale
)

# 3. Soft Constraints (Nullspace) setzen
# RMPFlow wird beim forward() diese bevorzugten Werte anstreben
_update_cspace_attractor()  # Setzt preferred_joints im Nullspace
```

**Verwendung:**

```python
# During simulation loop:
target_pos = np.array([1.0, 0.5, 0.3])
target_orient = np.array([1, 0, 0, 0])  # Quaternion [w,x,y,z]

action = rmpflow_ctrl.forward(
    target_end_effector_position=target_pos,
    target_end_effector_orientation=target_orient
)

# action.joint_positions = np.array([...7 werte...])
# → Diese werden zum Roboter geschickt
```

**Key-Features:**

| Feature | Effekt |
|---------|--------|
| `preferred_joints` | Soft Constraints im Nullspace |
| `trajectory_scale` | Skaliert physics_dt → beeinflusst Geschwindigkeit |
| `set_joint_preference(6, 0.5)` | Zur Laufzeit ändern |
| `clear_all_preferences()` | Zurücksetzen zu Default-Pose |

---

### Level 2: PickPlaceController_JW (Middle Ebene)

**Verantwortung:** 10-Phasen Pick&Place State Machine  
**Eingabe:** Greif-Position, Ablage-Position, aktuelle Joint-Werte  
**Ausgabe:** ArticulationAction (Joint-Positions + Greifer-Befehle)

**Initialisierung:**

```python
from Franka_Env_JW import PickPlaceController_JW, PRESET_ESSENTIAL_ONLY

pick_place_ctrl = PickPlaceController_JW(
    name="pick_place",
    gripper=gripper,                         # ParallelGripper Objekt
    robot_articulation=franka_robot,         # Robot body
    
    # Soft Constraints
    preferred_joints=PRESET_ESSENTIAL_ONLY,  # {2:0.0, 4:0.0, 6:0.78}
    
    # Geschwindigkeit
    trajectory_resolution=1.0,                # Alle Phasen normal speed
    air_speed_multiplier=2.0,                 # AIR-Phasen 2x schneller
    
    # Adaptiver Modus
    height_adaptive_speed=True,               # Dynamic speed bei niedriger Z
    critical_height_threshold=0.15,           # Unter 15cm "kritisch"
    critical_speed_factor=0.25,               # 4x langsamer nahe am Boden
)
```

**Die 10 Phasen:**

```
┌────────────────────────────────────────────────────────────┐
│ Phase │ Action              │ Type     │ Default dt │ Typ  │
├───────┼─────────────────────┼──────────┼────────────┼──────┤
│  0    │ Move above cube     │ Position │ 0.008 s    │ AIR  │
│  1    │ Lower to grip       │ Position │ 0.005 s    │ CRIT │
│  2    │ Wait for settle     │ Time     │ 0.1 s      │ WAIT │
│  3    │ Close gripper       │ Gripper  │ 0.1 s      │ GRIP │
│  4    │ Lift with cube      │ Position │ 0.05 s     │ AIR  │
│  5    │ Move to target XY   │ Position │ 0.05 s     │ AIR  │
│  6    │ Lower to place      │ Position │ 0.0025 s   │ CRIT │
│  7    │ Open gripper        │ Gripper  │ 1 s        │ REL  │
│  8    │ Lift up             │ Position │ 0.008 s    │ AIR  │
│  9    │ Return to start     │ Position │ 0.08 s     │ AIR  │
└───────┴─────────────────────┴──────────┴────────────┴──────┘

AIR Phasen:      0, 4, 5, 8, 9  (können schnell sein)
CRITICAL Phasen: 1, 6           (müssen präzise sein)
```

**Speed-Berechnung Beispiel:**

```
Basis events_dt:
[0.008, 0.005, 1, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]

Mit trajectory_resolution=1.0 (alle × 1.0):
[0.008, 0.005, 1, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]

Mit air_speed_multiplier=2.0 (AIR-Phasen × 2.0):
[0.016, 0.005, 1, 0.1, 0.10, 0.10, 0.0025, 1, 0.016, 0.16]
 ↑                    ↑      ↑       ↑             ↑      ↑
 Phase 0,4,5,8,9 verdoppelt (sind AIR-Phasen)

Mit height_adaptive_speed=True (zusätzliche Z-basierte Anpassung):
if current_z < 0.15:  # Kritische Höhe
    dt *= critical_speed_factor (0.25)  # 4x langsamer
```

**State Machine Ablauf:**

```python
def forward(picking_position, placing_position, current_joint_positions):
    # Jeder Call erhöht interne Zeit
    self._t += effective_dt
    
    # Berechne Ziel-Position basierend auf aktuelle Phase
    if phase == 0:
        # Move above
        target = [picking_position.x, picking_position.y, self._h1]
        gripper_action = "open"
        
    elif phase == 1:
        # Lower to grip
        target = picking_position  # Vollständig hinab
        gripper_action = "open"
        
    elif phase == 2:
        # Wait (keine Bewegung)
        target = current_ee_position
        gripper_action = "hold"
        
    elif phase == 3:
        # Close gripper
        target = current_ee_position
        gripper_action = "close"
        
    # ... Phase 4-9 analog ...
    
    # Wenn Phase-Dauer vorbei, zur nächsten Phase
    if self._t > self._events_dt[self._event]:
        self._event += 1
        self._t = 0
    
    # Sende Zielposition an RMPFlowController
    rmpflow_action = self._rmpflow_controller.forward(target)
    
    # Kombiniere mit Greifer-Aktion
    return ArticulationAction(
        joint_positions=rmpflow_action.joint_positions,
        gripper_action=gripper_action
    )
```

---

### Level 3: StackingController_JW (Oberste Ebene)

**Verantwortung:** Orchestrierung mehrerer Pick&Place-Zyklen  
**Eingabe:** Beobachtungen (Würfel-Positionen, Ziel-Positionen)  
**Ausgabe:** ArticulationAction für aktuelle Phase

**Initialisierung:**

```python
from Franka_Env_JW import StackingController_JW, PRESET_MINIMAL_MOTION

stacking_ctrl = StackingController_JW(
    name="stacking",
    gripper=gripper,
    robot_articulation=franka_robot,
    picking_order_cube_names=["cube_0", "cube_1", "cube_2"],
    robot_observation_name="franka_panda",
    
    # Parameter delegiert an PickPlaceController
    preferred_joints=PRESET_MINIMAL_MOTION,
    trajectory_resolution=1.0,
    air_speed_multiplier=1.5,
    height_adaptive_speed=True,
)
```

**Funktionsweise:**

```python
def forward(observations):
    # Prüfe: Sind alle Würfel fertig?
    if self._current_cube >= len(self._picking_order_cube_names):
        return ArticulationAction(joint_positions=[None]*7)  # Done
    
    # Hole aktuellen Würfel
    cube_name = self._picking_order_cube_names[self._current_cube]
    picking_pos = observations[cube_name]["position"]
    placing_pos = observations[cube_name]["target_position"]
    current_joints = observations[self.robot_observation_name]["joint_positions"]
    
    # Delegiere an PickPlaceController
    action = self._pick_place_controller.forward(
        picking_position=picking_pos,
        placing_position=placing_pos,
        current_joint_positions=current_joints,
    )
    
    # Wenn PickPlace fertig
    if self._pick_place_controller.is_done():
        self._current_cube += 1           # Nächster Würfel
        self._pick_place_controller.reset()  # State Machine zurücksetzen
    
    return action
```

**Simulation Loop:**

```
Episode = Alle Würfel gestacked

while not done:
    obs = task.get_observations()
    action = stacking_ctrl.forward(obs)
    
    # StackingController orchestriert:
    # Iteration 0-500:    Pick cube_0, Phase 0-9
    # Iteration 501-1000: Pick cube_1, Phase 0-9
    # Iteration 1001-1500: Pick cube_2, Phase 0-9
    # Iteration 1501+:    All done
    
    robot.apply_action(action)
    world.step()
```

---

## 📊 Detaillierte Funktionsweise

### Beispiel-Szenario: Greife Würfel

**Initiale Zustand:**
```
cube_0 Position: [0.5, 0.2, 0.05]
Ziel Position:   [0.5, 0.5, 0.1]
Franka Position: [0.0, 0.0, 0.0]
```

**Simulation Ticks:**

```
Tick 1-50: Phase 0 (Move above)
├─ target = [0.5, 0.2, 0.3]  (über dem Würfel)
├─ RMPFlow.forward(target) → joint_positions
├─ Greifer: offen
└─ → Arm bewegt sich nach oben zur Position

Tick 51-100: Phase 1 (Lower to grip) 
├─ target = [0.5, 0.2, 0.05]  (direkt auf Würfel)
├─ RMPFlow mit preferred_joints versucht gleichzeitig:
│  - Ziel (0.5, 0.2, 0.05) zu erreichen (PRIMÄR)
│  - Gelenk 6 ≈ 0.78 zu halten (SEKUNDÄR, im Nullspace)
├─ Greifer: offen
└─ → Arm senkt sich genau zum Würfel

Tick 101-120: Phase 2 (Wait)
├─ target = [0.5, 0.2, 0.05]  (keine Änderung)
├─ Greifer: offen
└─ → Wartet, bis Roboter settelt (Vibrationen abklingen)

Tick 121-140: Phase 3 (Close gripper)
├─ target = [0.5, 0.2, 0.05]  (keine Bewegung)
├─ Greifer: schließen
└─ → Greifer schließt sich um Würfel

Tick 141-200: Phase 4 (Lift with cube)
├─ target = [0.5, 0.2, 0.3]   (hoch mit Würfel)
├─ RMPFlow mit Würfel-Gewicht versucht:
│  - Ziel zu erreichen
│  - preferred_joints zu halten
├─ Greifer: fest geschlossen
└─ → Arm hebt Würfel

Tick 201-300: Phase 5 (Move to target XY)
├─ target = [0.5, 0.5, 0.3]   (XY Bewegung, Z konstant)
├─ height_adaptive_speed: Z=0.3 > 0.15 → Normal speed
├─ Greifer: fest geschlossen
└─ → Arm bewegt sich seitlich zum Ziel

Tick 301-350: Phase 6 (Lower to place)
├─ target = [0.5, 0.5, 0.1]   (down to place height)
├─ height_adaptive_speed: Z sinkt unter 0.15
│  → dt *= 0.25 (4x langsamer, präzisere Bewegung)
├─ Greifer: fest geschlossen
└─ → Arm senkt Würfel sehr präzise ab

Tick 351-370: Phase 7 (Open gripper)
├─ target = [0.5, 0.5, 0.1]   (keine Bewegung)
├─ Greifer: öffnen
└─ → Greifer lässt Würfel los

Tick 371-420: Phase 8 (Lift up)
├─ target = [0.5, 0.5, 0.3]   (hoch ohne Würfel)
├─ Greifer: offen
└─ → Arm hebt sich

Tick 421-500: Phase 9 (Return)
├─ target = [0.0, 0.0, 0.3]   (zurück zur Start-Position)
├─ Greifer: offen
└─ → Arm zurück

⏳ PickPlaceController.is_done() = True
→ StackingController._current_cube += 1
→ Nächster Würfel!
```

---

## 🤔 Warum diese Architektur?

### Problem ohne Hierarchie:

```python
# Schlechte Variante: Alles in einem großen Controller
class BadMonolithicController:
    def forward(self, obs):
        # 2000 Zeilen Code
        # - State Machine
        # - Geschwindigkeit Berechnung
        # - Height Adaptation
        # - IK-Berechnung
        # - Greifer-Steuerung
        # - Würfel-Tracking
        # → Unmögliches zu verstehen und zu debuggen
```

### Lösungsansatz: Separation of Concerns

```
Jeder Controller macht EINE Sache gut:

┌──────────────────────────────┐
│ StackingController           │
│ VERANTWORTUNG:               │
│ ✓ Welcher Würfel ist dran?   │
│ ✓ Tracking der Würfel        │
│ ✗ Wie bewegt man den Arm?    │
│ ✗ Inverse Kinematik          │
└──────────────────────────────┘
         │
         └─→ PickPlaceController
             VERANTWORTUNG:
             ✓ 10-Phasen State Machine
             ✓ Ziel-Positionen pro Phase
             ✓ Geschwindigkeit-Modulation
             ✗ Inverse Kinematik
             
             └─→ RMPFlowController
                 VERANTWORTUNG:
                 ✓ Inverse Kinematik
                 ✓ Soft Constraints
                 ✓ Nullspace Behavior
                 ✗ Welche Zielposition?
```

### Vorteile:

| Vorteil | Erklärung |
|---------|-----------|
| **Testbarkeit** | Jede Ebene kann isoliert getestet werden |
| **Wartbarkeit** | Bug in Phase 1? Nur PickPlaceController ändern |
| **Wiederverwendbarkeit** | RMPFlowController kann für andere Tasks genutzt werden |
| **Modularität** | Neue Features können pro Ebene hinzugefügt werden |
| **Verständlichkeit** | Jeder Code-Teil hat klare Verantwortung |
| **Parameter-Tuning** | Effekte sind isoliert und vorhersagbar |

### Konkrete Beispiele:

**Szenario 1: Schnellere Bewegung gewünscht**

```python
# Alt (hätte überall angepasst werden müssen):
controller.move_speed = 2.0
controller.phase_durations = [...]  # Alle Phase-Zeiten

# Neu (nur eine Parameter):
stacking_ctrl.set_trajectory_resolution(2.0)
# → Wirkt sich auf alle Phasen proportional aus

# Noch präziser (nur Air-Phasen schneller):
stacking_ctrl.air_speed_multiplier = 3.0
# → Phase 0, 4, 5, 8, 9 sind 3x schneller
# → Phase 1, 6 (kritisch) bleiben gleich
```

**Szenario 2: Andere Stack-Reihenfolge**

```python
# Alt (hätte State Machine umgestaltet werden müssen):
# 500 Zeilen Code ändern

# Neu (nur ein Parameter):
stacking_ctrl = StackingController_JW(
    picking_order_cube_names=["cube_2", "cube_0", "cube_1"]  # Andere Reihenfolge
)
```

**Szenario 3: Andere Soft Constraints**

```python
# Zur Laufzeit ändern (auch während Simulation!):
stacking_ctrl.set_preferred_joints({6: 0.0, 4: 1.57})
# → Nächste Pick-Phase nutzt neue Preferences
# → PickPlaceController und RMPFlow passen sich automatisch an
```

---

## 💻 Praktische Verwendungsbeispiele

### Beispiel 1: Standard-Setup für Datensammlung

```python
from Franka_Env_JW import StackingController_JW, PRESET_OPTIMIZED_EVENTS_DT
from Franka_Env_JW import PRESET_MINIMAL_MOTION

# Erstelle Controller mit optimierten Einstellungen
stacking_controller = StackingController_JW(
    name="data_collection_stack",
    gripper=franka_gripper,
    robot_articulation=franka_robot,
    picking_order_cube_names=["Cube_1", "Cube_2", "Cube_3"],
    robot_observation_name="Franka",
    
    # Schnelle Air-Bewegungen, präzise Griffer-Bewegungen
    preferred_joints=PRESET_MINIMAL_MOTION,      # Stabil
    trajectory_resolution=1.0,                    # Normal
    air_speed_multiplier=3.0,                     # 3x schneller in der Luft
    height_adaptive_speed=True,                   # Adaptiv nahe am Boden
    critical_height_threshold=0.15,               # 15cm
    critical_speed_factor=0.1,                    # 10x langsamer unten
)

# Simulation Loop
for episode in range(num_episodes):
    task.reset()
    stacking_controller.reset()
    
    for step in range(max_steps):
        observations = task.get_observations()
        actions = stacking_controller.forward(observations)
        
        robot.apply_action(actions)
        world.step(render=True)
        
        # Daten loggen
        data_logger.log(observations, actions)
        
        if stacking_controller.is_done():
            break
```

### Beispiel 2: Langsame, Präzise Bewegung

```python
# Für Debugging oder sehr präzise Operationen
stacking_controller = StackingController_JW(
    name="precision_stack",
    gripper=gripper,
    robot_articulation=franka_robot,
    picking_order_cube_names=["Cube_1", "Cube_2"],
    robot_observation_name="Franka",
    
    # Alles langsam und präzise
    trajectory_resolution=0.5,                    # 2x langsamer
    air_speed_multiplier=0.5,                     # Selbst Luft langsam
    height_adaptive_speed=True,
    critical_height_threshold=0.2,                # Mehr kritische Zone
    critical_speed_factor=0.05,                   # 20x langsamer unten
)
```

### Beispiel 3: Zur Laufzeit anpassen

```python
# Während Simulation verschiedene Konfigurationen testen
stacking_controller = StackingController_JW(...)

for episode in range(num_episodes):
    # Episode 0-10: Test 1
    if episode < 10:
        stacking_controller.use_preset("minimal")
        stacking_controller.set_trajectory_resolution(1.0)
    
    # Episode 11-20: Test 2 (schneller)
    elif episode < 20:
        stacking_controller.use_preset("essential")
        stacking_controller.set_trajectory_resolution(1.5)
        stacking_controller._air_speed_multiplier = 2.0
    
    # Episode 21+: Test 3 (schnellste Luft)
    else:
        stacking_controller.use_preset("wrist_rotation")
        stacking_controller._air_speed_multiplier = 4.0
    
    # Simuliere Episode
    task.reset()
    stacking_controller.reset()
    # ...
```

### Beispiel 4: Direkter RMPFlow Zugriff (Low-Level)

```python
# Wenn man mehr Kontrolle braucht, kann man direkt
# auf die untere Ebene zugreifen:

from Franka_Env_JW import RMPFlowController_JW

rmpflow = RMPFlowController_JW(
    name="direct_ik",
    robot_articulation=franka_robot,
    preferred_joints={2: 0.0, 6: 0.78}
)

# Direkt Positionen anfahren
target_positions = [
    np.array([0.5, 0.3, 0.2]),
    np.array([0.6, 0.4, 0.25]),
    np.array([0.7, 0.5, 0.3]),
]

for target_pos in target_positions:
    action = rmpflow.forward(target_pos)
    robot.apply_action(action)
    world.step()
```

---

## 🎓 Zusammenfassung

### Kernkonzepte:

1. **Nullspace**: Die Freiheitsgrade, die die End-Effektor-Position nicht beeinflussen
   - Ermöglicht Soft Constraints ohne den Zielpunkt zu beeinflussen
   - RMPFlow nutzt diese automatisch

2. **Hierarchische Struktur**:
   - **Level 1**: "Wie erreiche ich diese Position?" (RMPFlow + Nullspace)
   - **Level 2**: "Welche Positionen brauche ich in welcher Phase?" (State Machine)
   - **Level 3**: "Welcher Würfel ist dran?" (Orchestrierung)

3. **Soft Constraints** vs **Hard Constraints**:
   - Hard: Position (x, y, z) MUSS erreicht werden
   - Soft: Gelenkposition bevorzugt, aber nicht erzwungen

4. **Warum diese Architektur**:
   - Separation of Concerns
   - Leicht testbar und wartbar
   - Parameter-Tuning ist lokal und vorhersagbar
   - Wiederverwendbar für andere Tasks

### Verwendung:

```python
# Standard Setup
stacking_ctrl = StackingController_JW(
    name="stack",
    gripper=gripper,
    robot_articulation=robot,
    picking_order_cube_names=["c0", "c1"],
    robot_observation_name="robot",
    preferred_joints=PRESET_MINIMAL_MOTION,
    air_speed_multiplier=2.0,
)

# In Loop
action = stacking_ctrl.forward(observations)
robot.apply_action(action)
```

---

**Entstanden:** Januar 2026  
**Zugeordnet zu:** `/Franka_Env_JW/`  
**Relevant für:** Data Collection Pipeline, Simulation, Training

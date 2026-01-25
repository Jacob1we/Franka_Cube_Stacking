# Franka Cube Stacking - Configuration

## Simulation
- **headless**: false - True für schnellere Datensammlung (ohne GUI)
- **seed**: 111
- **world_root**: /World

## Parallelization
- **num_envs**: 1 - 1 = Single, >1 = Parallel (z.B. 4 für 2x2 Grid)
- **env_spacing**: 2.5 - Abstand zwischen Umgebungen (Meter)

## Data Collection
- **num_episodes**: 10 - Anzahl zu sammelnder Episoden (TOTAL)
- **path**: E:/00_Coding_SSD/fcs_datasets
- **name**: franka_cube_stack_ds
- **save_png**: true - Speichere alle Bilder auch als PNG
- **action_mode**: ee_pos
  - delta_pose: [dx, dy, dz, d_yaw] - relative Positionsänderung (4D)
  - velocity: [vx, vy, vz, omega_z] - Geschwindigkeiten (4D)
  - ee_pos: [x_s, y_s, z_s, x_e, y_e, z_e] - EE Start/End Position (6D, wie DINO WM)
- **action_interval**: 10
  - 1 = jeder Frame wird gespeichert
  - 10 = alle 10 Frames wird eine Action/H5 gespeichert
  - Die Action beschreibt dann die Bewegung über N Frames

## Camera
- **position**: [1.6, -2.0, 1.27] - SIDE_CAM_BASE_POS
- **euler**: [66.0, 0.0, 32.05] - Roll, Pitch, Yaw in Grad
- **frequency**: 20
- **resolution**: [224, 224] - Breite x Höhe

## Scene
- **width**: 0.60 - SCENE_WIDTH (Meter)
- **length**: 0.75 - SCENE_LENGTH (Meter)
- **plane_lift**: 0.001 - Ebene leicht über Boden

## Controller
- **trajectory_resolution**: 1.0 - TRAJECTORY_RESOLUTION
- **air_speed_multiplier**: 1.0 - Speed up AIR phases only (0,4,5,8,9)
- **height_adaptive_speed**: False - DYNAMIC: Slow down near ground!
- **critical_height_threshold**: 0.1 - Below xx m = critical zone
- **critical_speed_factor**: 0.5 - slower in critical zone

### 🎉 Optimierte dt-Werte (Durchbruch 2026-01-25)
- **air_dt**: 1.0 - Große Schritte für AIR-Phasen (0, 4, 5, 8, 9)
- **critical_dt**: 0.015 - Feinere Schritte für CRITICAL-Phasen (1, 6)
- **wait_dt**: 1.0 - Warten auf Stabilisierung
- **grip_dt**: 0.2 - Greifer schließen
- **release_dt**: 0.2 - Greifer öffnen

**Ergebnis:** ~150 Steps/Episode (1 Würfel) statt ~950 Steps (2 Würfel)
→ Erstes erfolgreiches DINO-WM Training!

## Cubes
- **count**: 2 - N_CUBES
- **side**: 0.05 - Kantenlänge (Meter)
- **min_dist_factor**: 1.5 - MIN_DIST = factor * CUBE_SIDE
- **max_placement_tries**: 200 - MAX_TRIES für Positionierung
- **yaw_range**: [-45.0, 45.0] - Rotation Range in Grad (reduziert wegen EE-Rotation)

## Robot Workspace
- **base_clearance**: 0.3 - FRANKA_BASE_CLEARANCE
- **max_reach**: 0.75 - Maximale Reichweite (konservativ, echte: ~0.855m)
- **min_reach**: 0.3 - Minimale Reichweite (zu nah = Selbstkollision)

## Validation
- **xy_tolerance**: 0.03 - Toleranz für X/Y Position (3 cm)
- **z_min_height**: 0.02 - Mindesthöhe über Boden (2 cm)
- **z_stack_tolerance**: 0.02 - Toleranz für Z-Stacking

## Materials (Domain Randomization)
1. **AllowedArea_Steel_Brushed**: [0.62, 0.62, 0.62, 1.00]
2. **AllowedArea_Aluminum_Mill**: [0.77, 0.77, 0.78, 1.00]
3. **AllowedArea_Wood_Oak**: [0.65, 0.53, 0.36, 1.00]
4. **AllowedArea_Wood_BirchPly**: [0.85, 0.74, 0.54, 1.00]
5. **AllowedArea_Plastic_HDPE_Black**: [0.08, 0.08, 0.08, 1.00]
6. **AllowedArea_Rubber_Mat**: [0.12, 0.12, 0.12, 1.00]
7. **AllowedArea_Acrylic_Frosted**: [1.00, 1.00, 1.00, 1.00]

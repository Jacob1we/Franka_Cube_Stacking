feat(controller): Add joint locking and trajectory resolution control

Implement enhanced motion control features for the Franka stacking controller
to reduce unnecessary movements and allow trajectory granularity adjustment.

## New Features

### Joint Locking System
- Add ability to lock specific robot joints to fixed values during motion
- Eliminates redundant rotational movements for cleaner trajectories
- Supports both static configuration and runtime modification

### Trajectory Resolution Control
- Add `trajectory_resolution` parameter to scale motion speed/granularity
- Values < 1.0: finer trajectories (more interpolation, smoother)
- Values > 1.0: coarser trajectories (fewer points, faster execution)

## New Files

- `rmpflow_controller_jw.py`: Custom RMPFlowController with joint locking
  - Wraps NVIDIA's RMPFlow motion policy
  - Overrides computed joint positions for locked joints
  - Includes 5 predefined joint locking presets

## Modified Files

- `pick_place_controller_jw.py`:
  - Add `locked_joints` and `trajectory_resolution` parameters
  - Scale `events_dt` based on trajectory resolution
  - Add runtime control methods (lock_joint, unlock_joint, etc.)
  - Include preset events_dt configurations (DEFAULT, FAST, FINE)

- `stacking_controller_jw.py`:
  - Forward new parameters to PickPlaceController
  - Add convenience method `use_preset()` for quick configuration
  - Add runtime joint control passthrough methods

- `__init__.py`:
  - Export RMPFlowController_JW and all presets

## Available Presets

| Preset                    | Locked Joints | Use Case                    |
|---------------------------|---------------|-----------------------------|
| PRESET_LOCK_WRIST_ROTATION| 6             | Stable end-effector orient. |
| PRESET_LOCK_UPPER_ARM     | 2             | Reduce shoulder complexity  |
| PRESET_MINIMAL_MOTION     | 2, 6          | Balanced constraint         |
| PRESET_LOCK_FOREARM       | 4             | Planar-like motion          |
| PRESET_ESSENTIAL_ONLY     | 2, 4, 6       | Maximum constraint          |

## Franka Panda Joint Reference

- Joint 0: Shoulder rotation (vertical axis)
- Joint 1: Shoulder tilt (forward/backward)
- Joint 2: Upper arm rotation
- Joint 3: Elbow
- Joint 4: Forearm rotation
- Joint 5: Wrist tilt
- Joint 6: Wrist rotation (end-effector orientation)

## Usage Example

from Franka_Env_JW import StackingController_JW, PRESET_ESSENTIAL_ONLY

controller = StackingController_JW(
    name="stacker",
    gripper=gripper,
    robot_articulation=franka,
    picking_order_cube_names=cubes,
    robot_observation_name="franka",
    locked_joints=PRESET_ESSENTIAL_ONLY,
    trajectory_resolution=1.5,
)

# Runtime adjustments
controller.use_preset("minimal")
controller.lock_joint(4, 0.0)
controller.set_trajectory_resolution(2.0)## Breaking Changes

None - all new parameters are optional with backward-compatible defaults.
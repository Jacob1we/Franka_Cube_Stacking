# isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller_jw.py

from typing import Optional, List, Dict
import logging
import numpy as np

from .base_stacking_controller_jw import Base_StackingController
from isaacsim.core.prims import SingleArticulation
from .pick_place_controller_jw import PickPlaceController_JW as PickPlaceController
from .rmpflow_controller_jw import (
    PRESET_LOCK_WRIST_ROTATION,
    PRESET_LOCK_UPPER_ARM, 
    PRESET_MINIMAL_MOTION,
    PRESET_LOCK_FOREARM,
    PRESET_ESSENTIAL_ONLY,
)

from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper


class StackingController_JW(Base_StackingController):
    """
    Enhanced Stacking Controller with joint locking and trajectory control.

    Args:
        name (str): Controller name
        gripper (ParallelGripper): Gripper controller
        robot_articulation (SingleArticulation): Robot articulation
        picking_order_cube_names (List[str]): Order of cubes to pick
        robot_observation_name (str): Name for robot observations
        locked_joints (Optional[Dict[int, float]]): Joints to lock for simpler motion.
            Available presets (import from rmpflow_controller_jw):
            - PRESET_LOCK_WRIST_ROTATION: Lock joint 6 (wrist rotation)
            - PRESET_LOCK_UPPER_ARM: Lock joint 2 (upper arm rotation)
            - PRESET_MINIMAL_MOTION: Lock joints 2 and 6
            - PRESET_LOCK_FOREARM: Lock joint 4 (forearm rotation)
            - PRESET_ESSENTIAL_ONLY: Lock joints 2, 4, 6 (most constrained)
            
            Manual specification example:
            {0: 0.0, 2: 0.0, 6: 0.0}  # Lock specific joints to 0
            
        trajectory_resolution (float): Controls trajectory granularity.
            < 1.0: Finer trajectory (more interpolation points, smoother but slower)
            > 1.0: Coarser trajectory (fewer points, faster execution)
            Default: 1.0
        events_dt (Optional[List[float]]): Custom phase timings.
            Presets available in PickPlaceController_JW:
            - DEFAULT_EVENTS_DT: Balanced (default)
            - FAST_EVENTS_DT: Faster execution
            - FINE_EVENTS_DT: Smoother motion
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        picking_order_cube_names: List[str],
        robot_observation_name: str,
        locked_joints: Optional[Dict[int, float]] = None,
        trajectory_resolution: float = 1.0,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        self.log = logging.getLogger("StackingController_JW")
        
        self.log.info(f"Initializing StackingController_JW '{name}':")
        self.log.info(f"  - locked_joints: {locked_joints}")
        self.log.info(f"  - trajectory_resolution: {trajectory_resolution}")
        
        # Store for runtime access
        self._locked_joints = locked_joints
        self._trajectory_resolution = trajectory_resolution
        
        # Create the pick-place controller with new parameters
        self._pick_place_ctrl = PickPlaceController(
            name=name + "_pick_place_controller", 
            gripper=gripper, 
            robot_articulation=robot_articulation,
            locked_joints=locked_joints,
            trajectory_resolution=trajectory_resolution,
            events_dt=events_dt,
        )
        
        Base_StackingController.__init__(
            self,
            name=name,
            pick_place_controller=self._pick_place_ctrl,
            picking_order_cube_names=picking_order_cube_names,
            robot_observation_name=robot_observation_name,
        )
        
        self.log.info(f"StackingController_JW '{name}' initialized successfully")
        return
    
    # =========================================================================
    # Runtime Control Methods
    # =========================================================================
    
    def set_locked_joints(self, locked_joints: Dict[int, float]) -> None:
        """Update locked joints at runtime."""
        self._locked_joints = locked_joints
        self._pick_place_ctrl.set_locked_joints(locked_joints)
    
    def lock_joint(self, joint_idx: int, value: float = 0.0) -> None:
        """Lock a specific joint to a value."""
        self._pick_place_ctrl.lock_joint(joint_idx, value)
    
    def unlock_joint(self, joint_idx: int) -> None:
        """Unlock a specific joint."""
        self._pick_place_ctrl.unlock_joint(joint_idx)
    
    def set_trajectory_resolution(self, resolution: float) -> None:
        """Update trajectory resolution (takes effect after reset)."""
        self._trajectory_resolution = resolution
        self._pick_place_ctrl.set_trajectory_resolution(resolution)
    
    def use_preset(self, preset_name: str) -> None:
        """
        Apply a predefined joint locking preset.
        
        Available presets:
        - "wrist_rotation": Lock wrist rotation only
        - "upper_arm": Lock upper arm rotation only
        - "minimal": Lock wrist + upper arm rotation
        - "forearm": Lock forearm rotation only
        - "essential": Maximum constraint (only shoulder, elbow, wrist tilt)
        - "none" / "unlock_all": Unlock all joints
        """
        presets = {
            "wrist_rotation": PRESET_LOCK_WRIST_ROTATION,
            "upper_arm": PRESET_LOCK_UPPER_ARM,
            "minimal": PRESET_MINIMAL_MOTION,
            "forearm": PRESET_LOCK_FOREARM,
            "essential": PRESET_ESSENTIAL_ONLY,
            "none": {},
            "unlock_all": {},
        }
        
        if preset_name not in presets:
            self.log.warning(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
            return
        
        self.set_locked_joints(presets[preset_name])
        self.log.info(f"Applied preset '{preset_name}': {presets[preset_name]}")

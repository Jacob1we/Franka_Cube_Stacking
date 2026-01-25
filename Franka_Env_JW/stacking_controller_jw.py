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
    Enhanced Stacking Controller with joint preferences and trajectory control.

    IMPORTANT: Joint preferences are "soft constraints" (cspace attractors).
    They influence the robot's null-space behavior to prefer certain joint configurations,
    but do NOT prevent the robot from reaching target positions accurately.

    Phase Overview:
    ---------------
    Phase 0: Move EE above cube     [AIR]      - can be fast
    Phase 1: Lower EE to cube       [CRITICAL] - must be precise for gripping
    Phase 2: Wait for settle        [WAIT]     - fixed timing
    Phase 3: Close gripper          [GRIP]     - fixed timing
    Phase 4: Lift EE up             [AIR]      - can be fast
    Phase 5: Move to target XY      [AIR]      - can be fast
    Phase 6: Lower to place         [CRITICAL] - must be precise for placing
    Phase 7: Open gripper           [RELEASE]  - fixed timing
    Phase 8: Lift EE up             [AIR]      - can be fast
    Phase 9: Return to start        [AIR]      - can be fast

    Args:
        name (str): Controller name
        gripper (ParallelGripper): Gripper controller
        robot_articulation (SingleArticulation): Robot articulation
        picking_order_cube_names (List[str]): Order of cubes to pick
        robot_observation_name (str): Name for robot observations
        preferred_joints (Optional[Dict[int, float]]): Preferred joint values (soft constraints).
            Available presets: PRESET_MINIMAL_MOTION, PRESET_ESSENTIAL_ONLY, etc.
        trajectory_resolution (float): Controls ALL phases uniformly.
            > 1.0: Coarser/faster. Default: 1.0
        air_speed_multiplier (float): Extra speed for AIR phases only (0, 4, 5, 8, 9).
            Does NOT affect critical phases (1, 6) or grip phases (2, 3, 7).
            Example: air_speed_multiplier=3.0 → air movements 3x faster
        height_adaptive_speed (bool): Enable DYNAMIC speed based on Z-height!
            Points near the ground automatically get finer resolution.
            The lowest points are ALWAYS preserved. Default: False
        critical_height_threshold (float): Height (m) below which points are critical.
            Default: 0.15m (15cm above ground)
        critical_speed_factor (float): Speed factor for critical heights (0.0-1.0).
            Lower = slower/finer at low heights. Default: 0.25 (4x slower near ground)
        events_dt (Optional[List[float]]): Custom phase timings.
            Presets: DEFAULT_EVENTS_DT, FAST_EVENTS_DT, FINE_EVENTS_DT, OPTIMIZED_EVENTS_DT
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        picking_order_cube_names: List[str],
        robot_observation_name: str,
        preferred_joints: Optional[Dict[int, float]] = None,
        trajectory_resolution: float = 1.0,
        air_speed_multiplier: float = 1.0,
        height_adaptive_speed: bool = False,
        critical_height_threshold: float = 0.15,
        critical_speed_factor: float = 0.25,
        guarantee_final_position: bool = True,
        guarantee_phases: Optional[List[int]] = None,
        events_dt: Optional[List[float]] = None,
        # Backwards compatibility
        locked_joints: Optional[Dict[int, float]] = None,
    ) -> None:
        self.log = logging.getLogger("StackingController_JW")
        
        # Handle backwards compatibility
        if locked_joints is not None and preferred_joints is None:
            preferred_joints = locked_joints
            self.log.warning("'locked_joints' is deprecated, use 'preferred_joints' instead")
        
        self.log.info(f"Initializing StackingController_JW '{name}':")
        self.log.info(f"  - preferred_joints: {preferred_joints}")
        self.log.info(f"  - trajectory_resolution: {trajectory_resolution}")
        self.log.info(f"  - air_speed_multiplier: {air_speed_multiplier}")
        self.log.info(f"  - height_adaptive_speed: {height_adaptive_speed}")
        if height_adaptive_speed:
            self.log.info(f"    - critical_height_threshold: {critical_height_threshold}m")
            self.log.info(f"    - critical_speed_factor: {critical_speed_factor}")
        self.log.info(f"  - guarantee_final_position: {guarantee_final_position}")
        if guarantee_final_position:
            self.log.info(f"    - guarantee_phases: {guarantee_phases if guarantee_phases else [1, 6]}")
        
        # Store for runtime access
        self._preferred_joints = preferred_joints
        self._trajectory_resolution = trajectory_resolution
        self._air_speed_multiplier = air_speed_multiplier
        self._height_adaptive_speed = height_adaptive_speed
        
        # Create the pick-place controller with new parameters
        self._pick_place_ctrl = PickPlaceController(
            name=name + "_pick_place_controller", 
            gripper=gripper, 
            robot_articulation=robot_articulation,
            preferred_joints=preferred_joints,
            trajectory_resolution=trajectory_resolution,
            air_speed_multiplier=air_speed_multiplier,
            height_adaptive_speed=height_adaptive_speed,
            critical_height_threshold=critical_height_threshold,
            critical_speed_factor=critical_speed_factor,
            guarantee_final_position=guarantee_final_position,
            guarantee_phases=guarantee_phases,
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
    
    def set_preferred_joints(self, preferred_joints: Dict[int, float]) -> None:
        """Update preferred joints at runtime (soft constraints)."""
        self._preferred_joints = preferred_joints
        self._pick_place_ctrl.set_preferred_joints(preferred_joints)
    
    def set_joint_preference(self, joint_idx: int, value: float) -> None:
        """Set preference for a specific joint."""
        self._pick_place_ctrl.set_joint_preference(joint_idx, value)
    
    def clear_joint_preference(self, joint_idx: int) -> None:
        """Clear preference for a specific joint."""
        self._pick_place_ctrl.clear_joint_preference(joint_idx)
    
    def clear_all_preferences(self) -> None:
        """Clear all joint preferences."""
        self._pick_place_ctrl.clear_all_preferences()
    
    def set_trajectory_resolution(self, resolution: float) -> None:
        """Update trajectory resolution (takes effect after reset)."""
        self._trajectory_resolution = resolution
        self._pick_place_ctrl.set_trajectory_resolution(resolution)
    
    def use_preset(self, preset_name: str) -> None:
        """
        Apply a predefined joint preference preset.
        
        Available presets:
        - "wrist_rotation": Prefer neutral wrist rotation
        - "upper_arm": Prefer neutral upper arm rotation
        - "minimal": Prefer neutral wrist + upper arm
        - "forearm": Prefer neutral forearm rotation
        - "essential": Prefer neutral for all rotational joints
        - "none" / "clear": Clear all preferences
        """
        presets = {
            "wrist_rotation": PRESET_LOCK_WRIST_ROTATION,
            "upper_arm": PRESET_LOCK_UPPER_ARM,
            "minimal": PRESET_MINIMAL_MOTION,
            "forearm": PRESET_LOCK_FOREARM,
            "essential": PRESET_ESSENTIAL_ONLY,
            "none": {},
            "clear": {},
        }
        
        if preset_name not in presets:
            self.log.warning(f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}")
            return
        
        if preset_name in ["none", "clear"]:
            self.clear_all_preferences()
        else:
            self.set_preferred_joints(presets[preset_name])
        self.log.info(f"Applied preset '{preset_name}': {presets.get(preset_name, {})}")
    
    # Backwards compatibility aliases
    def set_locked_joints(self, locked_joints: Dict[int, float]) -> None:
        """Deprecated: Use set_preferred_joints instead."""
        self.log.warning("set_locked_joints is deprecated, use set_preferred_joints")
        self.set_preferred_joints(locked_joints)
    
    def lock_joint(self, joint_idx: int, value: float = 0.0) -> None:
        """Deprecated: Use set_joint_preference instead."""
        self.set_joint_preference(joint_idx, value)
    
    def unlock_joint(self, joint_idx: int) -> None:
        """Deprecated: Use clear_joint_preference instead."""
        self.clear_joint_preference(joint_idx)

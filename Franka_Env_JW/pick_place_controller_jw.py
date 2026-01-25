# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Dict
import logging
import numpy as np

from .base_pick_place_controller_jw import Base_PickPlaceController_JW as Base_PickPlaceController
from .rmpflow_controller_jw import RMPFlowController_JW, PRESET_MINIMAL_MOTION, PRESET_ESSENTIAL_ONLY

from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper


class PickPlaceController_JW(Base_PickPlaceController):
    """
    Enhanced Pick and Place Controller with joint preferences and trajectory resolution control.

    IMPORTANT: Joint preferences are "soft constraints" - they influence null-space behavior
    but do NOT prevent the robot from reaching the target position accurately.

    Phase Overview (for understanding speed parameters):
    ---------------------------------------------------
    Phase 0: Move EE above cube at initial height     [AIR - can be fast]
    Phase 1: Lower EE down to cube                    [CRITICAL - must be precise]
    Phase 2: Wait for inertia to settle               [WAIT]
    Phase 3: Close gripper                            [GRIP]
    Phase 4: Lift EE up with cube                     [AIR - can be fast]
    Phase 5: Move EE toward target XY                 [AIR - can be fast]
    Phase 6: Lower EE to place cube                   [CRITICAL - must be precise]
    Phase 7: Open gripper                             [RELEASE]
    Phase 8: Lift EE up                               [AIR - can be fast]
    Phase 9: Return EE to starting position           [AIR - can be fast]

    Args:
        name (str): Controller name
        gripper (ParallelGripper): Gripper controller
        robot_articulation (SingleArticulation): Robot articulation
        end_effector_initial_height (Optional[float]): Initial EE height. Defaults to 0.3m.
        events_dt (Optional[List[float]]): Time deltas for each phase. 
            Larger values = faster/coarser movement.
        preferred_joints (Optional[Dict[int, float]]): Preferred joint values (soft constraints).
        trajectory_resolution (float): Controls ALL phases uniformly.
            Values > 1.0 = faster/coarser. Default: 1.0
        air_speed_multiplier (float): Extra speed boost for AIR phases only (0, 4, 5, 8, 9).
            Does NOT affect critical gripper phases (1, 6) or grip/release phases (2, 3, 7).
            Values > 1.0 = faster air movements. Default: 1.0
        height_adaptive_speed (bool): Enable dynamic speed based on Z-height.
            Points near the ground automatically get finer resolution.
            Default: False
        critical_height_threshold (float): Height (m) below which points are critical.
            Only used when height_adaptive_speed=True. Default: 0.15m
        critical_speed_factor (float): Speed factor for critical heights (0.0-1.0).
            Lower = slower/finer at low heights. Default: 0.25
    """

    # Phase indices for different movement types
    AIR_PHASES = [0, 4, 5, 8, 9]      # Movements in the air (can be fast)
    CRITICAL_PHASES = [1, 6]          # Approaching/placing (must be precise)
    GRIPPER_PHASES = [2, 3, 7]        # Wait/grip/release (fixed timing)

    # Default events_dt values (controls speed of each phase)
    # Lower values = slower movement = finer trajectory
    # [0:above, 1:lower, 2:wait, 3:grip, 4:lift, 5:move_xy, 6:place, 7:release, 8:lift, 9:return]
    DEFAULT_EVENTS_DT = [0.008, 0.005, 1, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
    
    # Preset for fast/coarse trajectory (all phases)
    FAST_EVENTS_DT = [0.016, 0.01, 1, 0.15, 0.1, 0.1, 0.005, 1, 0.016, 0.15]
    
    # Preset for slow/fine trajectory (all phases)
    FINE_EVENTS_DT = [0.004, 0.0025, 1, 0.05, 0.025, 0.025, 0.00125, 1, 0.004, 0.04]
    
    # Preset: Fast air, precise gripper (RECOMMENDED for data collection)
    OPTIMIZED_EVENTS_DT = [0.04, 0.005, 1, 0.1, 0.1, 0.1, 0.0025, 1, 0.04, 0.15]

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
        preferred_joints: Optional[Dict[int, float]] = None,
        trajectory_resolution: float = 1.0,
        air_speed_multiplier: float = 1.0,
        height_adaptive_speed: bool = False,
        critical_height_threshold: float = 0.15,
        critical_speed_factor: float = 0.25,
        guarantee_final_position: bool = True,
        guarantee_phases: Optional[List[int]] = None,
        # Backwards compatibility alias
        locked_joints: Optional[Dict[int, float]] = None,
    ) -> None:
        self.log = logging.getLogger("PickAndPlaceController_JW")
        
        # Handle backwards compatibility: locked_joints -> preferred_joints
        if locked_joints is not None and preferred_joints is None:
            preferred_joints = locked_joints
            self.log.warning("'locked_joints' is deprecated, use 'preferred_joints' instead")
        
        # Apply trajectory resolution to events_dt
        if events_dt is None:
            events_dt = self.DEFAULT_EVENTS_DT.copy()
        else:
            events_dt = list(events_dt)  # Make a copy
        
        # Scale events_dt by trajectory_resolution (affects ALL phases)
        scaled_events_dt = [dt * trajectory_resolution for dt in events_dt]
        
        # Apply air_speed_multiplier ONLY to air phases
        if air_speed_multiplier != 1.0:
            for phase_idx in self.AIR_PHASES:
                if phase_idx < len(scaled_events_dt):
                    scaled_events_dt[phase_idx] *= air_speed_multiplier
        
        self.log.info(f"PickPlaceController_JW initialized:")
        self.log.info(f"  - preferred_joints: {preferred_joints}")
        self.log.info(f"  - trajectory_resolution: {trajectory_resolution}")
        self.log.info(f"  - air_speed_multiplier: {air_speed_multiplier}")
        self.log.info(f"  - height_adaptive_speed: {height_adaptive_speed}")
        if height_adaptive_speed:
            self.log.info(f"    - critical_height_threshold: {critical_height_threshold}m")
            self.log.info(f"    - critical_speed_factor: {critical_speed_factor}")
        self.log.info(f"  - events_dt: {scaled_events_dt}")
        self.log.info(f"    (air phases {self.AIR_PHASES} boosted, critical phases {self.CRITICAL_PHASES} preserved)")
        
        # Store references for runtime modification
        self._preferred_joints = preferred_joints
        self._trajectory_resolution = trajectory_resolution
        self._air_speed_multiplier = air_speed_multiplier
        self._base_events_dt = events_dt  # Store original for recalculation
        
        # Create the RMPFlow controller with joint preferences
        self._rmpflow_controller = RMPFlowController_JW(
            name=name + "_cspace_controller", 
            robot_articulation=robot_articulation,
            preferred_joints=preferred_joints,
            trajectory_scale=trajectory_resolution,
        )
        
        # Store guarantee settings
        self._guarantee_final_position = guarantee_final_position
        self._guarantee_phases = guarantee_phases if guarantee_phases is not None else self.CRITICAL_PHASES
        
        self.log.info(f"  - guarantee_final_position: {guarantee_final_position}")
        if guarantee_final_position:
            self.log.info(f"    - guarantee_phases: {self._guarantee_phases}")
        
        Base_PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=self._rmpflow_controller,
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=scaled_events_dt,
            height_adaptive_speed=height_adaptive_speed,
            critical_height_threshold=critical_height_threshold,
            critical_speed_factor=critical_speed_factor,
            guarantee_final_position=guarantee_final_position,
            guarantee_phases=self._guarantee_phases,
        )
        return
    
    def set_preferred_joints(self, preferred_joints: Dict[int, float]) -> None:
        """Update preferred joints at runtime (soft constraints)."""
        self._preferred_joints = preferred_joints
        self._rmpflow_controller.set_preferred_joints(preferred_joints)
    
    def set_joint_preference(self, joint_idx: int, value: float) -> None:
        """Set preference for a specific joint at runtime."""
        self._rmpflow_controller.set_joint_preference(joint_idx, value)
    
    def clear_joint_preference(self, joint_idx: int) -> None:
        """Clear preference for a specific joint at runtime."""
        self._rmpflow_controller.clear_joint_preference(joint_idx)
    
    def clear_all_preferences(self) -> None:
        """Clear all joint preferences."""
        self._rmpflow_controller.clear_all_preferences()
    
    def set_trajectory_resolution(self, resolution: float) -> None:
        """
        Update trajectory resolution at runtime.
        Note: This affects events_dt, changes take effect after next reset().
        """
        self._trajectory_resolution = resolution
        new_events_dt = [dt * resolution for dt in self.DEFAULT_EVENTS_DT]
        self._events_dt = new_events_dt
        self.log.info(f"Trajectory resolution set to {resolution}, new events_dt: {new_events_dt}")
    
    # Backwards compatibility aliases
    def set_locked_joints(self, locked_joints: Dict[int, float]) -> None:
        """Deprecated: Use set_preferred_joints instead."""
        self.log.warning("set_locked_joints is deprecated, use set_preferred_joints")
        self.set_preferred_joints(locked_joints)
    
    def lock_joint(self, joint_idx: int, value: float) -> None:
        """Deprecated: Use set_joint_preference instead."""
        self.set_joint_preference(joint_idx, value)
    
    def unlock_joint(self, joint_idx: int) -> None:
        """Deprecated: Use clear_joint_preference instead."""
        self.clear_joint_preference(joint_idx)

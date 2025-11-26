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
    Enhanced Pick and Place Controller with joint locking and trajectory resolution control.

    Args:
        name (str): Controller name
        gripper (ParallelGripper): Gripper controller
        robot_articulation (SingleArticulation): Robot articulation
        end_effector_initial_height (Optional[float]): Initial EE height. Defaults to 0.3m.
        events_dt (Optional[List[float]]): Time deltas for each phase. 
            Larger values = faster/coarser movement.
        locked_joints (Optional[Dict[int, float]]): Joints to lock.
            Franka joint indices:
            - 0: Shoulder rotation (vertical axis)
            - 1: Shoulder tilt (forward/backward)
            - 2: Upper arm rotation
            - 3: Elbow
            - 4: Forearm rotation  
            - 5: Wrist tilt
            - 6: Wrist rotation (end-effector orientation)
        trajectory_resolution (float): Controls trajectory granularity.
            Values < 1.0 = finer trajectory (more points, smoother but slower)
            Values > 1.0 = coarser trajectory (fewer points, faster)
            Default: 1.0
    """

    # Default events_dt values (controls speed of each phase)
    # Lower values = slower movement = finer trajectory
    DEFAULT_EVENTS_DT = [0.008, 0.005, 1, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
    
    # Preset for fast/coarse trajectory
    FAST_EVENTS_DT = [0.016, 0.01, 1, 0.15, 0.1, 0.1, 0.005, 1, 0.016, 0.15]
    
    # Preset for slow/fine trajectory
    FINE_EVENTS_DT = [0.004, 0.0025, 1, 0.05, 0.025, 0.025, 0.00125, 1, 0.004, 0.04]

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
        locked_joints: Optional[Dict[int, float]] = None,
        trajectory_resolution: float = 1.0,
    ) -> None:
        self.log = logging.getLogger("PickAndPlaceController_JW")
        
        # Apply trajectory resolution to events_dt
        if events_dt is None:
            events_dt = self.DEFAULT_EVENTS_DT.copy()
        
        # Scale events_dt by trajectory_resolution
        # Higher resolution value = faster = multiply dt values
        scaled_events_dt = [dt * trajectory_resolution for dt in events_dt]
        
        self.log.info(f"PickPlaceController_JW initialized:")
        self.log.info(f"  - locked_joints: {locked_joints}")
        self.log.info(f"  - trajectory_resolution: {trajectory_resolution}")
        self.log.info(f"  - events_dt: {scaled_events_dt}")
        
        # Store references for runtime modification
        self._locked_joints = locked_joints
        self._trajectory_resolution = trajectory_resolution
        
        # Create the RMPFlow controller with joint locking
        self._rmpflow_controller = RMPFlowController_JW(
            name=name + "_cspace_controller", 
            robot_articulation=robot_articulation,
            locked_joints=locked_joints,
            trajectory_scale=trajectory_resolution,
        )
        
        Base_PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=self._rmpflow_controller,
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=scaled_events_dt,
        )
        return
    
    def set_locked_joints(self, locked_joints: Dict[int, float]) -> None:
        """Update locked joints at runtime."""
        self._locked_joints = locked_joints
        self._rmpflow_controller.set_locked_joints(locked_joints)
    
    def lock_joint(self, joint_idx: int, value: float) -> None:
        """Lock a specific joint to a value at runtime."""
        self._rmpflow_controller.lock_joint(joint_idx, value)
    
    def unlock_joint(self, joint_idx: int) -> None:
        """Unlock a specific joint at runtime."""
        self._rmpflow_controller.unlock_joint(joint_idx)
    
    def set_trajectory_resolution(self, resolution: float) -> None:
        """
        Update trajectory resolution at runtime.
        Note: This affects events_dt, changes take effect after next reset().
        """
        self._trajectory_resolution = resolution
        new_events_dt = [dt * resolution for dt in self.DEFAULT_EVENTS_DT]
        self._events_dt = new_events_dt
        self.log.info(f"Trajectory resolution set to {resolution}, new events_dt: {new_events_dt}")

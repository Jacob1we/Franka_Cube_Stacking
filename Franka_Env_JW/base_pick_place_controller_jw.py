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
import typing
import logging

import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper
import isaacsim.core.utils.rotations as rotations_utils



class Base_PickPlaceController_JW(BaseController):
    """
    A simple pick and place state machine for tutorials

    Each phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    #JW - Phase 10: Turn EE towards Cube Orientation
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.

    Args:
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
        end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
        events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.
        height_adaptive_speed (bool): If True, dynamically adjust speed based on Z-height.
            Points near the ground are always processed with fine resolution.
        critical_height_threshold (float): Height (in meters) below which points are considered critical.
            Default: 0.15m (15cm above ground)
        critical_speed_factor (float): Speed reduction factor for critical (low) heights.
            Default: 0.25 (4x slower when near ground)

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 10
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
        height_adaptive_speed: bool = False,
        critical_height_threshold: float = 0.15,
        critical_speed_factor: float = 0.25,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008, 0.08]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        
        # Store base events_dt for adaptive speed calculation
        self._base_events_dt = list(self._events_dt)
        
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        
        # Height-adaptive speed settings
        self._height_adaptive_speed = height_adaptive_speed
        self._critical_height_threshold = critical_height_threshold
        self._critical_speed_factor = critical_speed_factor
        self._current_target_height = None  # Track current target height

        self.log = logging.getLogger("Base_PickAndPlaceController")
        self.log.info("Initiation of Base Pick and Place Controller is Done")
        if height_adaptive_speed:
            self.log.info(f"  Height-adaptive speed enabled:")
            self.log.info(f"    - Critical height threshold: {critical_height_threshold}m")
            self.log.info(f"    - Critical speed factor: {critical_speed_factor}")
        
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        picking_orientation: np.ndarray, #JW
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="open")
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(placing_position[2])
            
            # Store current target height for adaptive speed calculation
            self._current_target_height = target_height
            
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            if end_effector_orientation is None:
                # end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0])) #JW
                picking_orientation_euler = rotations_utils.quat_to_euler_angles(quat=picking_orientation) 
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, picking_orientation_euler[2]])) #JW 
                # end_effector_orientation = picking_orientation      #JW
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )
        
        # Get effective dt (with height-adaptive adjustment if enabled)
        effective_dt = self._get_effective_dt()
        
        self._t += effective_dt
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions
    
    def _get_effective_dt(self) -> float:
        """
        Get the effective dt for the current step, with optional height-adaptive adjustment.
        
        When height_adaptive_speed is enabled:
        - Points near the ground (below critical_height_threshold) use slower speed
        - Points high in the air use the normal (possibly accelerated) speed
        - Smooth transition between the two based on height
        
        Returns:
            float: Effective dt value for this step
        """
        base_dt = self._events_dt[self._event]
        
        if not self._height_adaptive_speed:
            return base_dt
        
        # For gripper phases (2, 3, 7), don't apply height adjustment
        if self._event in [2, 3, 7]:
            return base_dt
        
        # If we don't have a target height yet, use base dt
        if self._current_target_height is None:
            return base_dt
        
        # Calculate height factor: 0.0 at ground, 1.0 at threshold and above
        height_ratio = min(1.0, self._current_target_height / self._critical_height_threshold)
        
        # Smooth interpolation using cosine (smoother transition)
        # At height=0: factor = critical_speed_factor (slow)
        # At height>=threshold: factor = 1.0 (normal speed)
        smooth_factor = self._critical_speed_factor + (1.0 - self._critical_speed_factor) * height_ratio
        
        # Apply factor: lower factor = smaller dt = finer trajectory
        effective_dt = base_dt * smooth_factor
        
        return effective_dt

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from. If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        self._current_target_height = None  # Reset height tracking
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            self._base_events_dt = list(events_dt)
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        return
    
    def set_height_adaptive_speed(
        self, 
        enabled: bool,
        critical_height_threshold: typing.Optional[float] = None,
        critical_speed_factor: typing.Optional[float] = None,
    ) -> None:
        """
        Enable or disable height-adaptive speed at runtime.
        
        Args:
            enabled: Whether to enable height-adaptive speed
            critical_height_threshold: Height below which points are critical (meters)
            critical_speed_factor: Speed factor for critical heights (0.0-1.0)
        """
        self._height_adaptive_speed = enabled
        if critical_height_threshold is not None:
            self._critical_height_threshold = critical_height_threshold
        if critical_speed_factor is not None:
            self._critical_speed_factor = critical_speed_factor
        
        self.log.info(f"Height-adaptive speed: {enabled}")
        if enabled:
            self.log.info(f"  - Critical threshold: {self._critical_height_threshold}m")
            self.log.info(f"  - Critical speed factor: {self._critical_speed_factor}")

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return

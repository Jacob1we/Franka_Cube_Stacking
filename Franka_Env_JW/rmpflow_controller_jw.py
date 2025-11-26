# Custom RMPFlow Controller with Joint Preferences and Trajectory Control
# Based on isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller

from typing import Optional, List, Dict
import logging
import numpy as np

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


class RMPFlowController_JW(mg.MotionPolicyController):
    """
    Enhanced RMPFlow Controller with joint preferences and trajectory resolution control.
    
    IMPORTANT: This controller uses "soft constraints" via cspace_attractor.
    Joints are NOT hard-locked (which would break IK), but the robot is 
    attracted towards the preferred joint values in the null-space.
    
    Args:
        name (str): Name of the controller
        robot_articulation (SingleArticulation): The robot articulation
        physics_dt (float): Physics timestep. Defaults to 1.0/60.0
        preferred_joints (Dict[int, float]): Dictionary mapping joint indices to preferred values.
            The robot will try to stay close to these values in the null-space.
            Franka Panda Joint Indices:
            - 0: panda_joint1 (Shoulder rotation around vertical axis)
            - 1: panda_joint2 (Shoulder tilt forward/backward)
            - 2: panda_joint3 (Upper arm rotation)
            - 3: panda_joint4 (Elbow)
            - 4: panda_joint5 (Forearm rotation)
            - 5: panda_joint6 (Wrist tilt)
            - 6: panda_joint7 (Wrist rotation / End-effector orientation)
        trajectory_scale (float): Scale factor for trajectory resolution.
            Values < 1.0 = finer (slower, more interpolation points)
            Values > 1.0 = coarser (faster, fewer interpolation points)
            Default: 1.0
    """
    
    # Default Franka joint positions (neutral pose)
    DEFAULT_CSPACE_TARGET = np.array([0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.78])

    def __init__(
        self, 
        name: str, 
        robot_articulation: SingleArticulation, 
        physics_dt: float = 1.0 / 60.0,
        preferred_joints: Optional[Dict[int, float]] = None,
        trajectory_scale: float = 1.0,
    ) -> None:
        self.log = logging.getLogger("RMPFlowController_JW")
        
        # Load RMPFlow configuration for Franka
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)

        # Adjust physics_dt based on trajectory_scale
        # Larger dt = faster movement = coarser trajectory
        adjusted_physics_dt = physics_dt * trajectory_scale
        
        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, 
            self.rmp_flow, 
            adjusted_physics_dt
        )

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, 
            robot_orientation=self._default_orientation
        )
        
        # Store preferred joints configuration
        self._preferred_joints = preferred_joints or {}
        self._trajectory_scale = trajectory_scale
        self._num_joints = 7  # Franka has 7 DOF (excluding gripper)
        
        # Apply preferred joints as cspace attractor
        self._update_cspace_attractor()
        
        self.log.info(f"RMPFlowController_JW initialized with preferred_joints={self._preferred_joints}, "
                      f"trajectory_scale={trajectory_scale}")
        
        return

    def _update_cspace_attractor(self) -> None:
        """
        Update the cspace attractor based on preferred joint values.
        The attractor pulls the robot towards these values in the null-space,
        without affecting the end-effector position.
        """
        if len(self._preferred_joints) > 0:
            # Start with default pose
            cspace_target = self.DEFAULT_CSPACE_TARGET.copy()
            
            # Apply preferred values
            for joint_idx, preferred_value in self._preferred_joints.items():
                if 0 <= joint_idx < self._num_joints:
                    cspace_target[joint_idx] = preferred_value
            
            # Set the cspace attractor in RMPFlow
            self._motion_policy.set_cspace_target(cspace_target)
            self.log.debug(f"Set cspace attractor: {cspace_target}")
    
    def forward(
        self,
        target_end_effector_position: np.ndarray,
        target_end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Compute joint positions to reach target end-effector pose.
        The preferred joint values influence the null-space behavior.
        
        Args:
            target_end_effector_position: Target position [x, y, z]
            target_end_effector_orientation: Target orientation as quaternion [w, x, y, z]
            
        Returns:
            ArticulationAction with computed joint positions
        """
        # Simply forward to parent - the cspace attractor is already set
        return mg.MotionPolicyController.forward(
            self,
            target_end_effector_position=target_end_effector_position,
            target_end_effector_orientation=target_end_effector_orientation,
        )

    def set_preferred_joints(self, preferred_joints: Dict[int, float]) -> None:
        """
        Update preferred joints configuration at runtime.
        
        Args:
            preferred_joints: Dictionary mapping joint indices to preferred values.
        """
        self._preferred_joints = preferred_joints
        self._update_cspace_attractor()
        self.log.info(f"Updated preferred_joints to {self._preferred_joints}")
    
    def set_joint_preference(self, joint_idx: int, value: float) -> None:
        """Set preference for a specific joint."""
        self._preferred_joints[joint_idx] = value
        self._update_cspace_attractor()
        self.log.info(f"Set joint {joint_idx} preference to {value}")
    
    def clear_joint_preference(self, joint_idx: int) -> None:
        """Clear preference for a specific joint."""
        if joint_idx in self._preferred_joints:
            del self._preferred_joints[joint_idx]
            self._update_cspace_attractor()
            self.log.info(f"Cleared preference for joint {joint_idx}")
    
    def clear_all_preferences(self) -> None:
        """Clear all joint preferences."""
        self._preferred_joints = {}
        # Reset to default cspace target
        self._motion_policy.set_cspace_target(self.DEFAULT_CSPACE_TARGET)
        self.log.info("Cleared all joint preferences")

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, 
            robot_orientation=self._default_orientation
        )
        # Re-apply cspace attractor after reset
        self._update_cspace_attractor()


# ============================================================================
# Convenience presets for common configurations (Soft Constraints / Attractors)
# These define preferred joint values that RMPFlow will try to stay close to
# in the null-space, without affecting end-effector accuracy.
# ============================================================================

# Preset: Prefer neutral wrist rotation for simpler pick-and-place
PRESET_LOCK_WRIST_ROTATION = {6: 0.78}  # Neutral wrist rotation

# Preset: Prefer neutral upper arm rotation for more predictable paths
PRESET_LOCK_UPPER_ARM = {2: 0.0}

# Preset: Prefer neutral for both wrist and upper arm
PRESET_MINIMAL_MOTION = {2: 0.0, 6: 0.78}

# Preset: Prefer neutral forearm rotation
PRESET_LOCK_FOREARM = {4: 0.0}

# Preset: Preferred neutral pose for rotational joints
PRESET_ESSENTIAL_ONLY = {
    2: 0.0,    # Upper arm rotation -> neutral
    4: 0.0,    # Forearm rotation -> neutral
    6: 0.78,   # Wrist rotation -> neutral
}

# Backwards compatibility aliases
PRESET_PREFER_WRIST_NEUTRAL = PRESET_LOCK_WRIST_ROTATION
PRESET_PREFER_MINIMAL_ROTATION = PRESET_MINIMAL_MOTION
PRESET_PREFER_NEUTRAL_POSE = PRESET_ESSENTIAL_ONLY


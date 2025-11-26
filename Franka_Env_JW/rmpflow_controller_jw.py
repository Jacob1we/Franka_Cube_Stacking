# Custom RMPFlow Controller with Joint Locking and Trajectory Control
# Based on isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller

from typing import Optional, List, Dict
import logging
import numpy as np

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


class RMPFlowController_JW(mg.MotionPolicyController):
    """
    Enhanced RMPFlow Controller with joint locking and trajectory resolution control.
    
    Args:
        name (str): Name of the controller
        robot_articulation (SingleArticulation): The robot articulation
        physics_dt (float): Physics timestep. Defaults to 1.0/60.0
        locked_joints (Dict[int, float]): Dictionary mapping joint indices to locked values.
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

    def __init__(
        self, 
        name: str, 
        robot_articulation: SingleArticulation, 
        physics_dt: float = 1.0 / 60.0,
        locked_joints: Optional[Dict[int, float]] = None,
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
        
        # Store locked joints configuration
        self._locked_joints = locked_joints or {}
        self._trajectory_scale = trajectory_scale
        self._num_joints = 7  # Franka has 7 DOF (excluding gripper)
        
        # Create the joint mask (True = active, False = locked)
        self._joint_mask = self._create_joint_mask()
        
        self.log.info(f"RMPFlowController_JW initialized with locked_joints={self._locked_joints}, "
                      f"trajectory_scale={trajectory_scale}")
        
        return

    def _create_joint_mask(self) -> np.ndarray:
        """Create a boolean mask for active joints (True = can move, False = locked)."""
        mask = np.ones(self._num_joints, dtype=bool)
        for joint_idx in self._locked_joints.keys():
            if 0 <= joint_idx < self._num_joints:
                mask[joint_idx] = False
        return mask
    
    def forward(
        self,
        target_end_effector_position: np.ndarray,
        target_end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Compute joint positions to reach target end-effector pose.
        Locked joints are replaced with their configured values.
        
        Args:
            target_end_effector_position: Target position [x, y, z]
            target_end_effector_orientation: Target orientation as quaternion [w, x, y, z]
            
        Returns:
            ArticulationAction with computed joint positions
        """
        # Get the action from parent controller
        action = mg.MotionPolicyController.forward(
            self,
            target_end_effector_position=target_end_effector_position,
            target_end_effector_orientation=target_end_effector_orientation,
        )
        
        # Apply locked joint values
        if action.joint_positions is not None and len(self._locked_joints) > 0:
            joint_positions = np.array(action.joint_positions)
            
            for joint_idx, locked_value in self._locked_joints.items():
                if 0 <= joint_idx < len(joint_positions):
                    joint_positions[joint_idx] = locked_value
            
            action = ArticulationAction(
                joint_positions=joint_positions,
                joint_velocities=action.joint_velocities,
                joint_efforts=action.joint_efforts,
            )
        
        return action

    def set_locked_joints(self, locked_joints: Dict[int, float]) -> None:
        """
        Update locked joints configuration at runtime.
        
        Args:
            locked_joints: Dictionary mapping joint indices to locked values.
        """
        self._locked_joints = locked_joints
        self._joint_mask = self._create_joint_mask()
        self.log.info(f"Updated locked_joints to {self._locked_joints}")
    
    def unlock_joint(self, joint_idx: int) -> None:
        """Unlock a specific joint."""
        if joint_idx in self._locked_joints:
            del self._locked_joints[joint_idx]
            self._joint_mask = self._create_joint_mask()
            self.log.info(f"Unlocked joint {joint_idx}")
    
    def lock_joint(self, joint_idx: int, value: float) -> None:
        """Lock a specific joint to a value."""
        self._locked_joints[joint_idx] = value
        self._joint_mask = self._create_joint_mask()
        self.log.info(f"Locked joint {joint_idx} to value {value}")

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, 
            robot_orientation=self._default_orientation
        )


# ============================================================================
# Convenience presets for common configurations
# ============================================================================

# Preset: Lock wrist rotation for simpler pick-and-place
PRESET_LOCK_WRIST_ROTATION = {6: 0.0}

# Preset: Lock upper arm rotation for more predictable paths
PRESET_LOCK_UPPER_ARM = {2: 0.0}

# Preset: Lock both wrist and upper arm for very constrained motion
PRESET_MINIMAL_MOTION = {2: 0.0, 6: 0.0}

# Preset: Lock forearm rotation for planar-ish motion
PRESET_LOCK_FOREARM = {4: 0.0}

# Preset: Maximum constraint - only use shoulder, elbow, and wrist tilt
PRESET_ESSENTIAL_ONLY = {
    2: 0.0,   # Upper arm rotation
    4: 0.0,   # Forearm rotation
    6: 0.0,   # Wrist rotation
}


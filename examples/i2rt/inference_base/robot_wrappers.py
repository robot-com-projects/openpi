"""
Robot wrapper classes for bimanual control.

This module provides wrapper classes that add additional functionality
to robot interfaces, such as button and gripper support.
"""

import time
import numpy as np


class YAMLeaderRobot:
    """
    Wrapper for YAM leader robot with button and gripper support.
    
    This class extends the basic YAM robot interface to provide:
    - Joint positions with gripper state
    - Button state reading
    - PD gain updates
    """
    
    def __init__(self, robot):
        """
        Initialize wrapper.
        
        Args:
            robot: YAM robot instance from i2rt
        """
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self):
        """
        Get joint positions and button states.
        
        Returns:
            tuple: (qpos_with_gripper, button_states)
                - qpos_with_gripper: numpy array of shape [7] (6 joints + gripper)
                - button_states: numpy array of button states
        """
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray):
        """
        Command joint positions (without gripper).
        
        Args:
            joint_pos: numpy array of shape [6] with joint positions
        """
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray):
        """
        Update PD gains.
        
        Args:
            kp: numpy array of shape [6] with proportional gains
            kd: numpy array of shape [6] with derivative gains
        """
        self._robot.update_kp_kd(kp, kd)


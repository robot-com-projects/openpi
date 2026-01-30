#!/usr/bin/env python3

"""
Bimanual robot inference with Isaac Sim via ROS2.

This script:
1. Subscribes to camera images from three cameras via ROS2 (published by Isaac Sim)
2. Subscribes to joint states via ROS2
3. Runs inference using openpi.policies with OpenPiPolicyAdapter
4. Publishes joint commands via ROS2 (Isaac Sim subscribes and executes)

Example:
    python3 isaac_sim_inference.py \
        --config-name pi05_i2rt_lora \
        --checkpoint-dir /workspace/thirdparty/models/openpi/checkpoints/pi05_i2rt_lora/cup_to_plate_pick_and_place_pi05_lora_finetuning/25000
        --execution-hz 30.0 \
        --max-actions-per-inference 25

Topics:
    Subscribed:
      - /left_camera (sensor_msgs/Image): Left wrist camera
      - /right_camera (sensor_msgs/Image): Right wrist camera  
      - /top_camera (sensor_msgs/Image): Top/torso camera
      - /joint_states (sensor_msgs/JointState): Current joint positions
    
    Published:
      - /joint_commands (sensor_msgs/JointState): Target joint positions
"""

import sys
import time
import argparse
import os
import numpy as np
from typing import Dict, Optional

# Fix for cv_bridge _ARRAY_API compatibility issue
try:
    import numpy.core.multiarray as _multiarray
    if '_ARRAY_API' not in _multiarray.__dict__:
        _multiarray.__dict__['_ARRAY_API'] = None
    if not hasattr(_multiarray, '_ARRAY_API'):
        setattr(_multiarray, '_ARRAY_API', None)
except (AttributeError, ImportError, KeyError):
    pass

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
from cv_bridge import CvBridge

# Compute paths relative to this file's location
current_file_path = os.path.dirname(os.path.abspath(__file__))
robot_os_root = os.path.abspath(os.path.join(current_file_path, "..", "..", "..", "..", ".."))
bimanual_path = os.path.join(robot_os_root, "ml", "bimanual")

# Import openpi dependencies
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

# Add parent directory to path for absolute imports
sys.path.insert(0, bimanual_path)

# Import refactored modules
from inference_base import OpenPiPolicyAdapter

# ==================== CONFIGURATION ====================

# Camera topic mappings (Isaac Sim publishes these via ROS2 bridge)
CAMERA_TOPICS = {
    'torso': '/top_camera',
    'teleop_left': '/left_camera',
    'teleop_right': '/right_camera',
}

# Joint state topics
JOINT_STATE_TOPIC = '/joint_states'
JOINT_COMMAND_TOPIC = '/joint_command'

# Policy image key mapping (using slashes as expected by openpi policies)
POLICY_IMAGE_KEYS = {
    'torso': 'observation/images/torso',
    'teleop_left': 'observation/images/teleop_left',
    'teleop_right': 'observation/images/teleop_right',
}

# Joint names for the bimanual robot (16 DOF: interleaved left/right)
# Order matches Isaac Sim: left1, right1, left2, right2, ..., left6, right6, then fingers
JOINT_NAMES = [
    'left_joint1', 'right_joint1',
    'left_joint2', 'right_joint2',
    'left_joint3', 'right_joint3',
    'left_joint4', 'right_joint4',
    'left_joint5', 'right_joint5',
    'left_joint6', 'right_joint6',
    'left_left_finger', 'left_right_finger',
    'right_left_finger', 'right_right_finger'
]

# =======================================================


class ROS2InferenceNode(Node):
    """ROS2 node for inference with Isaac Sim communication"""
    
    def __init__(self, policy_adapter: OpenPiPolicyAdapter,
                 execution_hz: float = 30.0,
                 max_actions_per_inference: int = 25,
                 action_diff_threshold: float = 0.1):
        super().__init__('isaac_sim_inference_node')
        
        self.policy_adapter = policy_adapter
        self.execution_hz = execution_hz
        self.max_actions_per_inference = max_actions_per_inference
        self.action_diff_threshold = action_diff_threshold
        
        self.bridge = CvBridge()
        
        # Camera images storage
        self.images: Dict[str, Optional[np.ndarray]] = {
            'torso': None,
            'teleop_left': None,
            'teleop_right': None,
        }
        
        # Joint state storage
        self.current_joint_positions: Optional[np.ndarray] = None
        self.joint_names_received: list = []
        
        # Inference state
        self.action_buffer = []
        self.action_index = 0
        self.last_execution_time = 0.0
        
        # Create camera subscribers
        for camera_name, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image,
                topic,
                lambda msg, name=camera_name: self._image_callback(msg, name),
                10
            )
            self.get_logger().info(f"Subscribed to camera: {topic}")
        
        # Create joint state subscriber
        self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self._joint_state_callback,
            10
        )
        self.get_logger().info(f"Subscribed to joint states: {JOINT_STATE_TOPIC}")
        
        # Create joint command publisher
        self.joint_command_pub = self.create_publisher(
            JointState,
            JOINT_COMMAND_TOPIC,
            10
        )
        self.get_logger().info(f"Publishing joint commands to: {JOINT_COMMAND_TOPIC}")
        
        # Create timer for inference loop
        timer_period = 1.0 / execution_hz
        self.inference_timer = self.create_timer(timer_period, self._inference_callback)
        
        self.get_logger().info("ROS2 Inference Node initialized")
        self.get_logger().info("Waiting for camera images and joint states...")
    
    def _image_callback(self, msg: Image, camera_name: str):
        """Callback for camera image messages"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.images[camera_name] = cv_image
        except Exception as e:
            self.get_logger().error(f"Error processing image from {camera_name}: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Callback for joint state messages"""
        try:
            # Store joint names if not yet received
            if not self.joint_names_received:
                self.joint_names_received = list(msg.name)
                self.get_logger().info(f"Received joint names: {self.joint_names_received}")
            
            # Convert to numpy array with 16 DOF (12 arm joints interleaved + 4 finger joints)
            positions = np.array(msg.position, dtype=np.float32)
            
            self.current_joint_positions = positions
        except Exception as e:
            self.get_logger().error(f"Error processing joint state: {e}")
    
    def _robot_obs_to_numpy(self, image: np.ndarray) -> np.ndarray:
        """Convert robot observation image to numpy format expected by policy"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # If image is already in HWC format with uint8, return as is
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                return image
            else:
                return (image * 255).astype(np.uint8)
        
        # Otherwise assume it's in CHW format and needs conversion
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return image
        
        return image
    
    def _publish_joint_command(self, target_positions: np.ndarray):
        """Publish joint command to Isaac Sim"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = target_positions.tolist()
        msg.velocity = []
        msg.effort = []
        
        self.joint_command_pub.publish(msg)
    
    def _execute_action(self, action: np.ndarray):
        """
        Execute a single action by publishing joint commands.
        
        Args:
            action: numpy array. Common formats:
                   - [14]: [left_arm_7, right_arm_7] where last value of each might be gripper
                   - [16]: [left_arm_6, left_gripper_2, right_arm_6, right_gripper_2]
                   - [18]: [left_arm_7, left_gripper_2, right_arm_7, right_gripper_2]
        """
        if self.current_joint_positions is None:
            self.get_logger().warn("No joint state received yet, skipping action")
            return
        
        action_len = len(action)
        self.get_logger().debug(f"Action shape: {action.shape}, length: {action_len}")
        # Swap: Policy's left -> Isaac's right, Policy's right -> Isaac's left
        target_right_arm = action[:6]  # Policy's left -> Isaac's right
        right_gripper_val = action[6]  # Policy's left gripper -> Isaac's right gripper
        target_left_arm = action[7:13]  # Policy's right -> Isaac's left
        left_gripper_val = action[13]  # Policy's right gripper -> Isaac's left gripper
        
        # Get current arm joint positions (exclude gripper fingers from joint_names_received)
        # Find where gripper fingers start in joint names
        gripper_start_idx = None
        for i, name in enumerate(self.joint_names_received):
            if 'finger' in name.lower():
                gripper_start_idx = i
                break
        
        if gripper_start_idx is None:
            self.get_logger().warn("Could not find gripper fingers in joint names, assuming last 4 joints")
            gripper_start_idx = len(self.joint_names_received) - 4
        
        num_arm_joints = gripper_start_idx
        
        # De-interleave joints: Isaac Sim sends [left1, right1, left2, right2, ...]
        # Extract left and right arm positions separately
        current_left_arm = []
        current_right_arm = []
        for i in range(0, num_arm_joints, 2):
            if i < len(self.current_joint_positions):
                current_left_arm.append(self.current_joint_positions[i])  # Isaac's "left"
            if i + 1 < len(self.current_joint_positions):
                current_right_arm.append(self.current_joint_positions[i + 1])  # Isaac's "right"
        
        current_left_arm = np.array(current_left_arm, dtype=np.float32)
        current_right_arm = np.array(current_right_arm, dtype=np.float32)
        
        # Ensure target arrays are numpy arrays
        target_left_arm = np.array(target_left_arm, dtype=np.float32)
        target_right_arm = np.array(target_right_arm, dtype=np.float32)
        
        # Limit target and current positions difference to prevent large jumps
        # Ensure both arrays have the same length for the operation
        min_len_left = min(len(target_left_arm), len(current_left_arm))
        if len(target_left_arm) < len(current_left_arm):
            # Pad target_left_arm with current values
            target_left_arm = np.concatenate([target_left_arm, current_left_arm[len(target_left_arm):]])
        elif len(current_left_arm) < len(target_left_arm):
            # Truncate target_left_arm
            target_left_arm = target_left_arm[:len(current_left_arm)]
        
        diff_left = target_left_arm - current_left_arm
        diff_left = np.clip(diff_left, -self.action_diff_threshold, self.action_diff_threshold)
        target_left_arm = current_left_arm + diff_left
        
        min_len_right = min(len(target_right_arm), len(current_right_arm))
        if len(target_right_arm) < len(current_right_arm):
            # Pad target_right_arm with current values
            target_right_arm = np.concatenate([target_right_arm, current_right_arm[len(target_right_arm):]])
        elif len(current_right_arm) < len(target_right_arm):
            # Truncate target_right_arm
            target_right_arm = target_right_arm[:len(current_right_arm)]
        
        diff_right = target_right_arm - current_right_arm
        diff_right = np.clip(diff_right, -self.action_diff_threshold, self.action_diff_threshold)
        target_right_arm = current_right_arm + diff_right
        
        # Convert gripper values (0-1) to finger positions
        # Only command left finger (right finger is mimic joint)
        # Range: -0.002 (open) to 0.038 (closed)
        GRIPPER_OPEN = -0.002
        GRIPPER_CLOSED = 0.038
        
        # Map 0-1 to finger position (linear interpolation)
        # 0 = open, 1 = closed
        if isinstance(left_gripper_val, (int, float)) or (isinstance(left_gripper_val, np.ndarray) and left_gripper_val.size == 1):
            val = float(left_gripper_val) if isinstance(left_gripper_val, np.ndarray) else left_gripper_val
            left_finger_left = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * val
        else:
            # Use first value if array
            left_finger_left = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * float(left_gripper_val[0])
        
        if isinstance(right_gripper_val, (int, float)) or (isinstance(right_gripper_val, np.ndarray) and right_gripper_val.size == 1):
            val = float(right_gripper_val) if isinstance(right_gripper_val, np.ndarray) else right_gripper_val
            right_finger_left = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * val
        else:
            # Use first value if array
            right_finger_left = GRIPPER_OPEN + (GRIPPER_CLOSED - GRIPPER_OPEN) * float(right_gripper_val[0])
        
        # Combine in interleaved order for Isaac Sim: left1, right1, left2, right2, ...
        num_joints = min(len(target_left_arm), len(target_right_arm), 6)
        interleaved_arm = []
        for i in range(num_joints):
            interleaved_arm.append(target_left_arm[i])   # Isaac's "left"
            interleaved_arm.append(target_right_arm[i])  # Isaac's "right"
        
        # Add fingers: left_left, left_right (mimic), right_left, right_right (mimic)
        # Only command left finger, right finger is handled by mimic joint
        target_positions = np.array(interleaved_arm + [
            left_finger_left, 0.0,   # Isaac's "left" fingers: command left, right is mimic
            right_finger_left, 0.0   # Isaac's "right" fingers: command left, right is mimic
        ], dtype=np.float32)
        
        self._publish_joint_command(target_positions)
    
    def _inference_callback(self):
        """Timer callback for inference loop"""
        current_time = time.time()
        
        # Check if we need to run inference
        need_inference = len(self.action_buffer) == 0 or self.action_index >= self.max_actions_per_inference
        
        if need_inference:
            # Check if we have all required data
            images_ready = all(img is not None for img in self.images.values())
            joints_ready = self.current_joint_positions is not None
            
            if not images_ready:
                missing = [k for k, v in self.images.items() if v is None]
                self.get_logger().warn(f"Waiting for camera images: {missing}")
                return
            
            if not joints_ready:
                self.get_logger().warn("Waiting for joint states...")
                return
            
            self.get_logger().info(f"Running inference (action_index={self.action_index})...")
            
            # Process images for policy
            images = {}
            for robot_key, policy_key in POLICY_IMAGE_KEYS.items():
                if self.images[robot_key] is not None:
                    img = self._robot_obs_to_numpy(self.images[robot_key])
                    images[policy_key] = img
            
            # Convert joint positions from 16-element interleaved format to 14-element format
            # Policy expects: [left_arm_6, left_gripper_1, right_arm_6, right_gripper_1] = 14 elements
            # Current format: [left1, right1, left2, right2, ..., left6, right6, left_fingers_2, right_fingers_2] = 16 elements
            
            # Extract left and right arm joints (first 12 elements are interleaved arm joints)
            left_arm_joints = []
            right_arm_joints = []
            for i in range(0, 12, 2):
                if i < len(self.current_joint_positions):
                    left_arm_joints.append(self.current_joint_positions[i])  # Isaac's "left"
                if i + 1 < len(self.current_joint_positions):
                    right_arm_joints.append(self.current_joint_positions[i + 1])  # Isaac's "right"
            
            # Convert finger positions to 2 gripper values (one per arm)
            # Only read left finger (right finger is mimic joint)
            # Range: -0.002 (open) to 0.038 (closed)
            GRIPPER_OPEN = -0.002
            GRIPPER_CLOSED = 0.038
            
            if len(self.current_joint_positions) >= 16:
                left_left_finger = self.current_joint_positions[12]   # Isaac's "left_left"
                right_left_finger = self.current_joint_positions[14]   # Isaac's "right_left"
                
                # Convert finger position to normalized gripper value (0 = open, 1 = closed)
                left_gripper_val = (left_left_finger - GRIPPER_OPEN) / (GRIPPER_CLOSED - GRIPPER_OPEN)
                left_gripper_val = np.clip(left_gripper_val, 0.0, 1.0)
                
                right_gripper_val = (right_left_finger - GRIPPER_OPEN) / (GRIPPER_CLOSED - GRIPPER_OPEN)
                right_gripper_val = np.clip(right_gripper_val, 0.0, 1.0)
            else:
                left_gripper_val = 0.0
                right_gripper_val = 0.0
            
            # Combine to 14-element format for policy: [policy_left_arm_6, policy_left_gripper_1, policy_right_arm_6, policy_right_gripper_1]
            # Swap: Isaac's left -> policy's right, Isaac's right -> policy's left
            policy_state = np.concatenate([
                np.array(right_arm_joints, dtype=np.float32),  # Isaac's right -> policy's left
                np.array([right_gripper_val], dtype=np.float32),
                np.array(left_arm_joints, dtype=np.float32),   # Isaac's left -> policy's right
                np.array([left_gripper_val], dtype=np.float32)
            ])
            
            # Get joint positions (using slash format as expected by openpi policies)
            states = {
                'observation/state': policy_state
            }
            
            # Run inference
            try:
                actions_batch = self.policy_adapter.run_inference(images=images, states=states)
                self.action_buffer = actions_batch
                self.action_index = 0
                self.get_logger().info(f"Inference complete: got {actions_batch.shape[0]} actions")
            except Exception as e:
                self.get_logger().error(f"Inference error: {e}")
                return
        
        # Execute action from buffer
        if self.action_index < len(self.action_buffer):
            current_action = self.action_buffer[self.action_index]
            self._execute_action(current_action)
            self.action_index += 1
            
            if self.last_execution_time > 0:
                fps = 1.0 / (current_time - self.last_execution_time)
                self.get_logger().info(
                    f"Executed action {self.action_index}/{self.max_actions_per_inference} | fps: {fps:.1f}"
                )
            
            self.last_execution_time = current_time


def load_policy(config_name: str, checkpoint_dir: str) -> OpenPiPolicyAdapter:
    """
    Load policy using openpi.policies.
    
    Args:
        config_name: Policy config name
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        OpenPiPolicyAdapter instance
    """
    print("Loading inference model using openpi.policies...")
    print(f"  Config name: {config_name}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    
    config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    print("✓ Inference model loaded successfully")
    
    return OpenPiPolicyAdapter(policy)


def main():
    parser = argparse.ArgumentParser(description='Bimanual robot inference via ROS2 (for Isaac Sim)')
    parser.add_argument('--config-name', type=str, default='pi0_i2rt_lora',
                        help='Policy config name (default: pi0_i2rt_lora)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to checkpoint directory (required)')
    parser.add_argument('--execution-hz', type=float, default=30.0,
                        help='Action execution frequency in Hz (default: 30.0)')
    parser.add_argument('--max-actions-per-inference', type=int, default=25,
                        help='Number of actions to execute before next inference (default: 25)')
    parser.add_argument('--action-diff-threshold', type=float, default=0.2,
                        help='Maximum change per action step in radians (default: 0.1)')
    
    args = parser.parse_args()
    
    # Check required arguments
    if args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required")
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Load policy
        policy_adapter = load_policy(args.config_name, args.checkpoint_dir)
        
        # Create inference node
        node = ROS2InferenceNode(
            policy_adapter=policy_adapter,
            execution_hz=args.execution_hz,
            max_actions_per_inference=args.max_actions_per_inference,
            action_diff_threshold=args.action_diff_threshold
        )
        
        print("\n" + "="*60)
        print("Starting inference loop via ROS2...")
        print("Isaac Sim should be running on host and publishing to:")
        print(f"  - Camera topics: {list(CAMERA_TOPICS.values())}")
        print(f"  - Joint states: {JOINT_STATE_TOPIC}")
        print(f"Joint commands will be published to: {JOINT_COMMAND_TOPIC}")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Spin the node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n\nStopping inference...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()

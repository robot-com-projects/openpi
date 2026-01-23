"""
Base controller class for bimanual teleoperation with inference.

This module provides a clean, modular controller that can work with
different policy backends through the PolicyAdapter interface.
"""

import time
import numpy as np
from typing import Optional, Dict, Any

from .robot_wrappers import YAMLeaderRobot
from .utils import robot_obs_to_numpy
from .policy_adapters import PolicyAdapter
from .recording import InferenceRecorder


class BimanualControllerBase:
    """
    Base class for bimanual teleoperation with inference mode.
    
    This class handles:
    - Button event handling
    - Teleoperation mode
    - Inference mode
    - Action execution
    
    Button 0: Toggle sync/unsync (teleoperation)
    Button 1: Toggle inference mode (only when unsynced)
    """
    
    def __init__(
        self,
        policy_adapter: PolicyAdapter,
        inference_robot,
        leader_right: Optional[YAMLeaderRobot] = None,
        leader_left: Optional[YAMLeaderRobot] = None,
        follower_right=None,
        follower_left=None,
        bilateral_kp: float = 0.0,
        actions_per_inference: int = 12,
        execution_hz: float = 66.0,
        action_diff_threshold: float = 0.3,
        mode: str = "full_control",
        record_samples: bool = False,
        recording_repo_id: str = "inference_samples",
        recording_fps: int = 30,
    ):
        """
        Initialize controller.
        
        Args:
            policy_adapter: PolicyAdapter instance for running inference
            inference_robot: Robot instance for getting observations
            leader_right: Right leader robot (optional, for full_control mode)
            leader_left: Left leader robot (optional, for full_control mode)
            follower_right: Right follower client (optional, for full_control mode)
            follower_left: Left follower client (optional, for full_control mode)
            bilateral_kp: Bilateral force feedback gain
            actions_per_inference: Number of actions to execute before next inference
            execution_hz: Action execution frequency
            action_diff_threshold: Maximum change per action step in radians
            mode: Control mode ("full_control" or "observation_only")
        """
        self.policy_adapter = policy_adapter
        self.inference_robot = inference_robot
        self.leader_right = leader_right
        self.leader_left = leader_left
        self.follower_right = follower_right
        self.follower_left = follower_left
        self.bilateral_kp = bilateral_kp
        self.actions_per_inference = actions_per_inference
        self.execution_hz = execution_hz
        self.action_diff_threshold = action_diff_threshold
        self.mode = mode
        
        # State flags
        self.synchronized = False
        self.inference_mode = False
        self.button0_prev = False
        self.button1_prev = False
        
        # Store initial kp values for bilateral control
        if self.leader_right is not None and self.leader_left is not None:
            self.robot_kp_right = self.leader_right._robot._kp
            self.robot_kp_left = self.leader_left._robot._kp
        else:
            self.robot_kp_right = None
            self.robot_kp_left = None
        
        # For real robot inference-execution cycle
        self.action_buffer = []  # Stores actions from inference
        self.action_index = 0    # Tracks which action to execute next
        self.last_execution_time = 0  # For execution timing
        
        # Recording support
        self.recorder = None
        if record_samples:
            try:
                self.recorder = InferenceRecorder(
                    robot=inference_robot,
                    repo_id=recording_repo_id,
                    fps=recording_fps,
                )
                print("✓ Inference recording enabled")
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize recorder: {e}")
                self.recorder = None
    
    def slow_move_to_leader(self, leader_pos_right, leader_pos_left, duration: float = 1.0):
        """Slowly move followers to match leader positions"""
        if self.follower_right is None or self.follower_left is None:
            return
            
        current_follower_right = self.follower_right.get_joint_pos()
        current_follower_left = self.follower_left.get_joint_pos()
        
        steps = 100
        for i in range(steps):
            alpha = i / steps
            
            # Interpolate positions
            target_right = leader_pos_right * alpha + current_follower_right * (1 - alpha)
            target_left = leader_pos_left * alpha + current_follower_left * (1 - alpha)
            
            self.follower_right.command_joint_pos(target_right)
            self.follower_left.command_joint_pos(target_left)
            
            time.sleep(0.03)
    
    def handle_button_events(self, current_button):
        """Handle button press events with debouncing"""
        if self.mode == "observation_only":
            # Button handling disabled in observation-only mode
            return
            
        button0_pressed = current_button[0] > 0.5
        button1_pressed = current_button[1] > 0.5
        
        # Button 0: Toggle sync/unsync
        if button0_pressed and not self.button0_prev:
            if self.inference_mode:
                # If inference is active, just disable it and go to unsync
                print("🤖 Inference mode DISABLED (Button 0 pressed during inference)")
                self.inference_mode = False
                self.synchronized = False
                
                # Clear bilateral PD
                if self.leader_right is not None and self.leader_left is not None:
                    self.leader_right.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                    self.leader_left.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
            else:
                # Normal sync toggle
                self.synchronized = not self.synchronized
                
                if self.synchronized:
                    print("🔗 SYNC mode activated")
                    # Get current positions
                    leader_pos_right, _ = self.leader_right.get_info()
                    leader_pos_left, _ = self.leader_left.get_info()
                    
                    # Set bilateral PD
                    self.leader_right.update_kp_kd(
                        kp=self.robot_kp_right * self.bilateral_kp, 
                        kd=np.zeros(6)
                    )
                    self.leader_left.update_kp_kd(
                        kp=self.robot_kp_left * self.bilateral_kp, 
                        kd=np.zeros(6)
                    )
                    
                    # Command leaders to current position
                    self.leader_right.command_joint_pos(leader_pos_right[:6])
                    self.leader_left.command_joint_pos(leader_pos_left[:6])
                    
                    # Slowly move followers to leader
                    self.slow_move_to_leader(leader_pos_right, leader_pos_left)
                else:
                    print("🔓 UNSYNC mode - robot stopped")
                    # Clear bilateral PD
                    if self.leader_right is not None and self.leader_left is not None:
                        self.leader_right.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
                        self.leader_left.update_kp_kd(kp=np.zeros(6), kd=np.zeros(6))
        
        # Button 1: Toggle inference (only when unsynced)
        if button1_pressed and not self.button1_prev:
            if not self.synchronized:
                self.inference_mode = not self.inference_mode
                
                if self.inference_mode:
                    print("🤖 INFERENCE mode activated")
                else:
                    print("🛑 INFERENCE mode deactivated - robot stopped")
            else:
                print("⚠️  Cannot activate inference while in sync mode")
        
        # Update previous button states
        self.button0_prev = button0_pressed
        self.button1_prev = button1_pressed
    
    def run_teleoperation(self):
        """Execute one step of teleoperation"""
        if self.mode == "observation_only":
            return
            
        if self.leader_right is None or self.leader_left is None:
            return
        if self.follower_right is None or self.follower_left is None:
            return
        
        # Get leader positions
        leader_pos_right, _ = self.leader_right.get_info()
        leader_pos_left, _ = self.leader_left.get_info()
        
        # Get follower positions
        follower_pos_right = self.follower_right.get_joint_pos()
        follower_pos_left = self.follower_left.get_joint_pos()

        # Limit follower and leader positions difference
        diff_right = leader_pos_right - follower_pos_right
        diff_left = leader_pos_left - follower_pos_left
        diff_threshold = 0.1
        diff_right = np.clip(diff_right, -diff_threshold, diff_threshold)
        diff_left = np.clip(diff_left, -diff_threshold, diff_threshold)
        follower_pos_right = leader_pos_right - diff_right
        follower_pos_left = leader_pos_left - diff_left
        
        # Command followers to match leaders
        self.follower_right.command_joint_pos(leader_pos_right)
        self.follower_left.command_joint_pos(leader_pos_left)
        
        # Set bilateral force (followers push back on leaders)
        self.leader_right.command_joint_pos(follower_pos_right[:6])
        self.leader_left.command_joint_pos(follower_pos_left[:6])
    
    def execute_inference_action(self, actions: np.ndarray):
        """
        Extract joint positions from actions array and command follower arms.
        
        Args:
            actions: numpy array of shape [action_dim] where action_dim = 14 (7 per arm)
                     Format: [left.j0, left.j1, ..., left.j6, right.j0, right.j1, ..., right.j6]
        """
        # Extract left and right arm target positions (7 DOFs each)
        target_right = actions[:7]
        target_left = actions[7:14]
        
        if self.mode == "observation_only":
            # Print the actions instead of executing them
            print(f"  Action - Right arm: {target_right}")
            print(f"  Action - Left arm:  {target_left}")
            print(f"  Full action array: {actions}")
            return
        
        if self.follower_right is None or self.follower_left is None:
            return
        
        # Get current follower positions
        follower_pos_right = self.follower_right.get_joint_pos()
        follower_pos_left = self.follower_left.get_joint_pos()
        
        # Limit target and current positions difference
        diff_right = target_right - follower_pos_right
        diff_left = target_left - follower_pos_left
        diff_right = np.clip(diff_right, -self.action_diff_threshold, self.action_diff_threshold)
        diff_left = np.clip(diff_left, -self.action_diff_threshold, self.action_diff_threshold)
        target_right = follower_pos_right + diff_right
        target_left = follower_pos_left + diff_left
        
        # Command the follower arms
        self.follower_right.command_joint_pos(target_right)
        self.follower_left.command_joint_pos(target_left)
    
    def prepare_observation(self, raw_obs: Dict[str, Any]) -> tuple:
        """
        Prepare observation from raw robot observation.
        
        Args:
            raw_obs: Raw observation dictionary from robot
            
        Returns:
            tuple: (images_dict, states_dict, robot_task_string)
        """
        # Process images - map robot observation keys to policy input keys
        # This mapping may need to be adjusted based on policy backend
        image_key_mapping = {
            'torso': 'observation.images.torso',  # For pi.policy_api
            'teleop_left': 'observation.images.teleop_left',
            'teleop_right': 'observation.images.teleop_right',
        }
        
        # Try openpi format if pi format doesn't work
        openpi_image_key_mapping = {
            'torso': 'observation/images/torso',  # For openpi
            'teleop_left': 'observation/images/teleop_left',
            'teleop_right': 'observation/images/teleop_right',
        }
        
        images = {}
        policy_image_keys = self.policy_adapter.get_image_keys()
        
        # Try pi.policy_api format first
        for robot_key, policy_key in image_key_mapping.items():
            if robot_key in raw_obs and policy_key in policy_image_keys:
                img = robot_obs_to_numpy(raw_obs[robot_key])
                images[policy_key] = img
        
        # If no images found, try openpi format
        if len(images) == 0:
            for robot_key, policy_key in openpi_image_key_mapping.items():
                if robot_key in raw_obs and policy_key in policy_image_keys:
                    img = robot_obs_to_numpy(raw_obs[robot_key])
                    images[policy_key] = img
        
        # Crop head cam images if needed
        for k in list(images.keys()):
            if "head_cam" in k:
                h = images[k].shape[0]
                images[k] = images[k][h // 3 :, :, :]
        
        # Collect joint positions into a single state array
        joint_positions = []
        for j in range(7):
            key = f'right.j{j}.pos'
            if key in raw_obs:
                joint_positions.append(float(raw_obs[key]))
        for j in range(7):
            key = f'left.j{j}.pos'
            if key in raw_obs:
                joint_positions.append(float(raw_obs[key]))
        
        policy_state_keys = self.policy_adapter.get_state_keys()
        states = {}
        
        # Try different state key formats
        if 'observation.state' in policy_state_keys:
            states['observation.state'] = np.array(joint_positions, dtype=np.float32)
        elif 'observation/state' in policy_state_keys:
            states['observation/state'] = np.array(joint_positions, dtype=np.float32)
        else:
            # Use first available state key
            if len(policy_state_keys) > 0:
                states[policy_state_keys[0]] = np.array(joint_positions, dtype=np.float32)
        
        robot_task_string = self.policy_adapter.get_robot_task_string()
        
        return images, states, robot_task_string
    
    def run_inference(self):
        """Execute one step of inference"""
        if not self.inference_mode:
            return
        
        current_time = time.time()
        
        # Check if we need to run inference (buffer empty or executed required actions)
        need_inference = len(self.action_buffer) == 0 or self.action_index >= self.actions_per_inference
        
        if need_inference:
            print(f"\n{'='*60}")
            print(f"Running inference (action_index={self.action_index})...")
            
            # Get current robot observation
            raw_obs = self.inference_robot.get_observation()
            
            # Prepare observation
            images, states, robot_task_string = self.prepare_observation(raw_obs)
            
            # Run inference through adapter
            # Get RNG if available (for pi.policy_api)
            rng = None
            if hasattr(self.policy_adapter, 'rng'):
                rng = self.policy_adapter.rng
            
            actions_batch = self.policy_adapter.run_inference(
                images=images,
                states=states,
                robot_task_string=robot_task_string,
                rng=rng
            )
            
            # Store actions in buffer
            self.action_buffer = actions_batch
            self.action_index = 0
            
            print(f"Inference complete: got {actions_batch.shape[0]} actions")
            print(f"Action shape: {actions_batch.shape}")
            print(f"{'='*60}\n")
        
        # Execute actions at specified frequency
        time_since_last_execution = current_time - self.last_execution_time
        execution_period = 1.0 / self.execution_hz
        
        if time_since_last_execution >= execution_period:
            if len(self.action_buffer) == 0:
                return
                
            # Get current action from buffer
            current_action = self.action_buffer[self.action_index]
            
            # Execute the action
            self.execute_inference_action(current_action)
            
            # Record sample if recording is enabled
            if self.recorder is not None:
                try:
                    # Get current observation for recording
                    raw_obs = self.inference_robot.get_observation()
                    self.recorder.record_frame(
                        observation=raw_obs,
                        action=current_action,
                        task="inference"
                    )
                except Exception as e:
                    print(f"⚠️  Warning: Could not record frame: {e}")
            
            # Update state
            self.action_index += 1
            fps = 1.0 / time_since_last_execution if time_since_last_execution > 0 else float("inf")
            print(f"Executed action {self.action_index}/{self.actions_per_inference} | fps: {fps:.2f}")
            self.last_execution_time = current_time
    
    def run(self):
        """Run the main control loop"""
        try:
            if self.mode == "observation_only":
                # Automatically enable inference mode (no button required)
                if not self.inference_mode:
                    print("🤖 Auto-enabling INFERENCE mode (observation-only mode)")
                    self.inference_mode = True
                
                while True:
                    if self.inference_mode:
                        self.run_inference()
                    else:
                        time.sleep(0.1)
                    time.sleep(0.01)
            else:
                # Full control mode: use button controls
                while True:
                    # Get button state from right leader (buttons are shared)
                    if self.leader_right is not None:
                        _, current_button = self.leader_right.get_info()
                        
                        # Handle button events
                        self.handle_button_events(current_button)
                    
                    # Execute appropriate control mode
                    if self.inference_mode:
                        self.run_inference()
                    elif self.synchronized:
                        self.run_teleoperation()
                    # else: do nothing (unsync mode)

                    time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n\nStopping control loop...")
        finally:
            # Finalize recording if enabled
            if self.recorder is not None:
                try:
                    self.recorder.finalize()
                except Exception as e:
                    print(f"⚠️  Warning: Could not finalize recording: {e}")


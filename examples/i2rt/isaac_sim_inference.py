#!/usr/bin/env python3

"""
Bimanual robot inference in Isaac Sim with camera observations.

This script:
1. Loads the RTOP robot USD file in Isaac Sim
2. Subscribes to camera images from three cameras via ROS2
3. Runs inference using openpi.policies with OpenPiPolicyAdapter
4. Executes actions by commanding the robot in Isaac Sim

Usage:
    1. Make sure Isaac Sim is set up with ROS2 bridge enabled
    2. Ensure the USD file has cameras configured and publishing to ROS2 topics
    3. Run the script:
       python3 isaac_sim_inference.py --config-name <config_name> --checkpoint-dir <checkpoint_dir>
    
    Required arguments:
      --config-name: Policy config name (e.g., 'pi0_i2rt_lora')
      --checkpoint-dir: Path to checkpoint directory
    
    Optional arguments:
      --usd-file: Path to USD file (default: rtop_ros_cams_new_design.usd)
      --headless: Run Isaac Sim in headless mode
      --sim-already-running: Skip Isaac Sim initialization (use if simulation is already loaded)
      --execution-hz: Action execution frequency (default: 30.0)
      --max-actions-per-inference: Actions to execute before next inference (default: 25)
      --action-diff-threshold: Maximum change per action step in radians (default: 0.1)

Example:
    python3 isaac_sim_inference.py \
        --config-name pi0_i2rt_lora \
        --checkpoint-dir /home/i2rt/openpi/checkpoints/pi05_i2rt_lora/pi_lora_train/20000 \
        --execution-hz 30.0 \
        --max-actions-per-inference 25

Note:
    - Camera topic names may need adjustment based on your Isaac Sim setup
    - Check available topics with: ros2 topic list
    - Make sure cameras are publishing images before running inference
"""

import sys
import time
import argparse
import os
import numpy as np
# Fix for cv_bridge _ARRAY_API compatibility issue - must be done before cv_bridge import
# Workaround: cv_bridge expects _ARRAY_API attribute in numpy.core.multiarray
# This is a known compatibility issue between cv_bridge and newer numpy versions
# We need to set this before any cv_bridge imports
try:
    import numpy.core.multiarray as _multiarray
    # Set the attribute directly in the module's __dict__ to ensure it's available
    if '_ARRAY_API' not in _multiarray.__dict__:
        _multiarray.__dict__['_ARRAY_API'] = None
    # Also set it as an attribute
    if not hasattr(_multiarray, '_ARRAY_API'):
        setattr(_multiarray, '_ARRAY_API', None)
except (AttributeError, ImportError, KeyError):
    pass
from typing import Dict, Optional

# Isaac Sim imports
# IMPORTANT: When Isaac Sim is installed via pip, SimulationApp must be initialized
# BEFORE importing omni.isaac.core modules. We'll import SimulationApp here but
# defer importing core modules until after initialization (or if sim is already running).
# Initialize to None so they can be set later
SimulationApp = None
add_reference_to_stage = None
World = None
MultiArticulation = None
rot_utils = None

# Try to import SimulationApp only (this can be done before initialization)
try:
    try:
        from isaacsim import SimulationApp
    except ImportError:
        # Fallback: try omni.isaac.sim
        from omni.isaac.sim import SimulationApp
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    print("Warning: Isaac Sim SimulationApp not available (will retry if --sim-already-running is used)")

# Core modules will be imported after SimulationApp initialization or if sim is already running
# This is required when Isaac Sim is installed via pip
def _import_isaac_core_modules():
    """Import Isaac Sim core modules. Must be called after SimulationApp is initialized."""
    global add_reference_to_stage, World, MultiArticulation, rot_utils
    try:
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core import World
        # MultiArticulation is typically in articulations
        try:
            from omni.isaac.core.articulations import Articulation as MultiArticulation
        except ImportError:
            # Alternative location - try RigidPrim or other prim types
            try:
                from omni.isaac.core.prims import RigidPrim as MultiArticulation
            except ImportError:
                # Last resort - use XFormPrim (may need adjustment based on actual usage)
                from omni.isaac.core.prims import XFormPrim as MultiArticulation
        
        import omni.isaac.core.utils.numpy.rotations as rot_utils
        return True
    except ImportError as e:
        print(f"Warning: Failed to import Isaac Sim core modules: {e}")
        return False

# SimulationApp will be initialized in main() if needed
simulation_app = None

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Compute paths relative to this file's location
current_file_path = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels: scripts -> rtop_moveit_config -> bimanual -> i2rt -> robot-os
robot_os_root = os.path.abspath(os.path.join(current_file_path, "..", "..", "..", "..", ".."))
thirdparty_i2rt_path = os.path.join(robot_os_root, "thirdparty", "i2rt")
bimanual_path = os.path.join(robot_os_root, "ml", "bimanual")
inference_base_path = os.path.join(bimanual_path, "inference_base")

# Import openpi dependencies
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

# Add parent directory to path for absolute imports
sys.path.insert(0, bimanual_path)

# Import refactored modules
from inference_base import OpenPiPolicyAdapter

# ==================== CONFIGURATION ====================
USD_FILE_PATH = "/home/zetans/Desktop/moveit_isaac_ws/src/i2rt/bimanual/usd_bimanual/rtop_ros_cams_new_design.usd"
ROBOT_PRIM_PATH = "/World/rtop"

# Camera topic mappings (Isaac Sim publishes these via ROS2 bridge)
# Note: These topic names may need to be adjusted based on your Isaac Sim setup
# You can check available topics with: ros2 topic list
CAMERA_TOPICS = {
    'torso': '/top_camera',  # Base camera
    'teleop_left': '/left_camera',  # Left wrist
    'teleop_right': '/right_camera',  # Right wrist
}

# Alternative topic names (uncomment if the above don't work)
# CAMERA_TOPICS = {
#     'torso': '/World/rtop/base/Camera/color/image_raw',
#     'teleop_left': '/World/rtop/left_d435/wrist_camera/IntelRealsense_D435_Multibody/Camera/color/image_raw',
#     'teleop_right': '/World/rtop/right_d435/wrist_camera/IntelRealsense_D435_Multibody/Camera/color/image_raw',
# }

# Policy image key mapping
POLICY_IMAGE_KEYS = {
    'torso': 'observation.images.torso',
    'teleop_left': 'observation.images.teleop_left',
    'teleop_right': 'observation.images.teleop_right',
}

# Joint names for the bimanual robot (14 DOF: 7 per arm)
JOINT_NAMES = [
    'left.j0', 'left.j1', 'left.j2', 'left.j3', 'left.j4', 'left.j5', 'left.j6',
    'right.j0', 'right.j1', 'right.j2', 'right.j3', 'right.j4', 'right.j5', 'right.j6'
]

# =======================================================


class CameraSubscriber(Node):
    """ROS2 node to subscribe to camera images from Isaac Sim"""
    
    def __init__(self):
        super().__init__('isaac_sim_camera_subscriber')
        self.bridge = CvBridge()
        self.images = {}
        self.image_lock = {}
        
        # Create subscribers for each camera
        for camera_name, topic in CAMERA_TOPICS.items():
            self.images[camera_name] = None
            self.image_lock[camera_name] = False
            self.create_subscription(
                Image,
                topic,
                lambda msg, name=camera_name: self.image_callback(msg, name),
                10
            )
            self.get_logger().info(f"Subscribed to {topic}")
    
    def image_callback(self, msg: Image, camera_name: str):
        """Callback for camera image messages"""
        try:
            # Convert ROS2 Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            # Convert to numpy array (H, W, C) format
            self.images[camera_name] = cv_image
            self.image_lock[camera_name] = True
        except Exception as e:
            self.get_logger().error(f"Error processing image from {camera_name}: {e}")
    
    def get_images(self) -> Dict[str, np.ndarray]:
        """Get current images from all cameras"""
        return {k: v for k, v in self.images.items() if v is not None}


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


class IsaacSimBimanualInference:
    """
    Manages bimanual robot inference in Isaac Sim using openpi.
    """
    
    def __init__(self, policy_adapter: OpenPiPolicyAdapter, usd_file_path: str = USD_FILE_PATH,
                 execution_hz: float = 30.0, max_actions_per_inference: int = 25,
                 action_diff_threshold: float = 0.1, sim_already_running: bool = False):
        self.usd_file_path = usd_file_path
        self.execution_hz = execution_hz
        self.max_actions_per_inference = max_actions_per_inference
        self.action_diff_threshold = action_diff_threshold
        self.policy_adapter = policy_adapter
        self.sim_already_running = sim_already_running
        
        if sim_already_running:
            # Simulation is already running, just get references to existing objects
            print("Simulation already running - connecting to existing scene...")
            self.world = World()
            # Get robot articulation from existing scene
            self.articulation = MultiArticulation(prim_path=ROBOT_PRIM_PATH)
            # Initialize articulation if not already initialized
            try:
                self.articulation.initialize()
            except Exception as e:
                print(f"Note: Articulation may already be initialized: {e}")
            print("✓ Connected to existing Isaac Sim scene")
        else:
            # Initialize Isaac Sim from scratch
            print("Initializing Isaac Sim...")
            self.world = World()
            self.world.scene.add_default_ground_plane()
            
            # Load USD file
            print(f"Loading USD file: {usd_file_path}")
            add_reference_to_stage(usd_file_path, ROBOT_PRIM_PATH)
            
            # Get robot articulation
            self.articulation = MultiArticulation(prim_path=ROBOT_PRIM_PATH)
            self.world.reset()
            self.articulation.initialize()
            print("✓ Isaac Sim initialized")
        
        # Initialize ROS2 node for camera subscriptions
        print("Initializing ROS2 camera subscribers...")
        rclpy.init()
        self.camera_subscriber = CameraSubscriber()
        print("✓ ROS2 camera subscribers initialized")
        
        # For inference-execution cycle
        self.action_buffer = []  # Stores actions from inference
        self.action_index = 0    # Tracks which action to execute next
        self.last_execution_time = 0  # For execution timing
        self.rng = None  # RNG is handled by policy_adapter
        
        # Wait a bit for cameras to start publishing
        print("Waiting for cameras to initialize...")
        time.sleep(2.0)
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions from Isaac Sim"""
        joint_positions = self.articulation.get_joint_positions()
        # Ensure we have 14 DOF (7 per arm)
        if len(joint_positions) < 14:
            # Pad if needed
            joint_positions = np.pad(joint_positions, (0, 14 - len(joint_positions)))
        elif len(joint_positions) > 14:
            # Truncate if needed
            joint_positions = joint_positions[:14]
        return joint_positions
    
    def robot_obs_to_numpy(self, image: np.ndarray) -> np.ndarray:
        """Convert robot observation image to numpy format expected by policy"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # If image is already in HWC format with uint8, return as is
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                return image
            else:
                # If float, scale to uint8
                return (image * 255).astype(np.uint8)
        
        # Otherwise assume it's in CHW format and needs conversion
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return image
        
        return image
    
    def execute_inference_action(self, actions: np.ndarray):
        """
        Extract joint positions from actions array and command robot in Isaac Sim.
        
        Args:
            actions: numpy array of shape [action_dim] where action_dim = 14 (7 per arm)
                     Format: [left.j0, left.j1, ..., left.j6, right.j0, right.j1, ..., right.j6]
        """
        # Extract left and right arm target positions (7 DOFs each)
        target_left = actions[:7]
        target_right = actions[7:14]
        
        # Get current joint positions
        current_positions = self.get_joint_positions()
        current_left = current_positions[:7]
        current_right = current_positions[7:14]
        
        # Limit target and current positions difference to prevent large jumps
        diff_left = target_left - current_left
        diff_right = target_right - current_right
        diff_left = np.clip(diff_left, -self.action_diff_threshold, self.action_diff_threshold)
        diff_right = np.clip(diff_right, -self.action_diff_threshold, self.action_diff_threshold)
        target_left = current_left + diff_left
        target_right = current_right + diff_right
        
        # Combine and command the robot
        target_positions = np.concatenate([target_left, target_right])
        self.articulation.set_joint_positions(positions=target_positions)
    
    def run_inference(self):
        """Execute one step of inference"""
        current_time = time.time()
        
        # Spin ROS2 to get latest camera images
        rclpy.spin_once(self.camera_subscriber, timeout_sec=0.01)
        
        # Check if we need to run inference (buffer empty or executed required actions)
        need_inference = len(self.action_buffer) == 0 or self.action_index >= self.max_actions_per_inference
        
        if need_inference:
            print(f"\n{'='*60}")
            print(f"Running inference (action_index={self.action_index})...")
            
            # Get current camera images
            camera_images = self.camera_subscriber.get_images()
            
            if len(camera_images) < len(CAMERA_TOPICS):
                print(f"Warning: Only received {len(camera_images)}/{len(CAMERA_TOPICS)} camera images. Waiting...")
                return
            
            # Process images for policy
            images = {}
            for robot_key, policy_key in POLICY_IMAGE_KEYS.items():
                if robot_key in camera_images:
                    img = self.robot_obs_to_numpy(camera_images[robot_key])
                    images[policy_key] = img
            
            # Crop head cam images if needed (not applicable here, but keeping for compatibility)
            for k, v in images.items():
                if k in ["observation.images.head_cam_left", "observation.images.head_cam_right"]:
                    h = v.shape[0]
                    images[k] = v[h // 3:, :, :]
            
            # Get joint positions
            joint_positions = self.get_joint_positions()
            states = {
                'observation.state': np.array(joint_positions, dtype=np.float32)
            }
            
            # Run inference using policy adapter
            # Note: OpenPiPolicyAdapter interface may vary - adjust method name if needed
            # Common methods: predict(), run_inference(), or similar
            actions_batch = self.policy_adapter.predict(images=images, states=states)
            
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
            # Get current action from buffer
            if self.action_index < len(self.action_buffer):
                current_action = self.action_buffer[self.action_index]
                
                # Execute the action
                self.execute_inference_action(current_action)
                
                # Update state
                self.action_index += 1
                
                fps = 1.0 / (current_time - self.last_execution_time) if self.last_execution_time > 0 else 0
                print(f"Executed action {self.action_index}/{self.max_actions_per_inference} | fps: {fps:.1f}")
                self.last_execution_time = current_time
    
    def run(self):
        """Run the main control loop"""
        print("\n" + "="*60)
        print("Starting inference loop...")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        try:
            while True:
                # Step simulation (only if we're managing it)
                if not self.sim_already_running:
                    self.world.step(render=True)
                else:
                    # If sim is already running, we might still need to step for rendering
                    # but the main simulation loop is handled elsewhere
                    # Optionally step for rendering updates
                    self.world.step(render=False)  # Don't render if already being rendered
                
                # Run inference
                self.run_inference()
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n\nStopping inference loop...")
        finally:
            # Cleanup
            rclpy.shutdown()
            simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description='Bimanual robot inference in Isaac Sim')
    parser.add_argument('--config-name', type=str, default='pi0_i2rt_lora',
                        help='Policy config name (default: pi0_i2rt_lora)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to checkpoint directory (required)')
    parser.add_argument('--usd-file', type=str, default=USD_FILE_PATH,
                        help=f'Path to USD file (default: {USD_FILE_PATH})')
    parser.add_argument('--headless', action='store_true',
                        help='Run Isaac Sim in headless mode')
    parser.add_argument('--sim-already-running', action='store_true',
                        help='Skip Isaac Sim initialization (use if simulation is already loaded and running)')
    parser.add_argument('--execution-hz', type=float, default=30.0,
                        help='Action execution frequency in Hz (default: 30.0)')
    parser.add_argument('--max-actions-per-inference', type=int, default=25,
                        help='Number of actions to execute before next inference (default: 25)')
    parser.add_argument('--action-diff-threshold', type=float, default=0.1,
                        help='Maximum change per action step in radians (default: 0.1)')
    
    args = parser.parse_args()
    
    # Check required arguments
    if args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required")
    
    # Initialize SimulationApp only if simulation is not already running
    global simulation_app, ISAAC_SIM_AVAILABLE
    global SimulationApp, add_reference_to_stage, World, MultiArticulation, rot_utils
    
    if not args.sim_already_running:
        if not ISAAC_SIM_AVAILABLE:
            raise ImportError("Isaac Sim is not available. Cannot initialize simulation.")
        # Initialize SimulationApp FIRST (required before importing omni.isaac.core modules)
        simulation_app = SimulationApp({"headless": args.headless})
        print("✓ SimulationApp initialized")
        # Now import core modules after initialization
        if not _import_isaac_core_modules():
            raise ImportError("Failed to import Isaac Sim core modules after initialization")
        print("✓ Isaac Sim core modules imported")
    else:
        print("✓ Using existing SimulationApp (simulation already running)")
        # Try to import Isaac Sim modules - they should be available now that sim is running
        if not ISAAC_SIM_AVAILABLE:
            print("Initial imports failed, retrying Isaac Sim imports (simulation may be running)...")
            try:
                # Try to import SimulationApp
                try:
                    from isaacsim import SimulationApp as _SimulationApp
                except ImportError:
                    from omni.isaac.sim import SimulationApp as _SimulationApp
                SimulationApp = _SimulationApp
                ISAAC_SIM_AVAILABLE = True
            except ImportError as e:
                raise ImportError(
                    f"Isaac Sim SimulationApp is not available. Cannot connect to existing simulation.\n"
                    f"Original import error: {e}\n"
                    f"Make sure Isaac Sim is running and the Python environment has access to Isaac Sim modules."
                ) from e
        
        # Import core modules (sim is already running, so they should be available)
        if not _import_isaac_core_modules():
            raise ImportError(
                "Failed to import Isaac Sim core modules. "
                "Make sure Isaac Sim is running and the Python environment has access to Isaac Sim modules."
            )
        print("✓ Isaac Sim core modules imported")
    
    # Update headless mode if specified (only if we created the app)
    if args.headless and not args.sim_already_running and simulation_app is not None:
        simulation_app.set_setting("/app/headless", True)
    
    try:
        # Load policy
        policy_adapter = load_policy(args.config_name, args.checkpoint_dir)
        
        # Setup controller with inference
        controller = IsaacSimBimanualInference(
            policy_adapter=policy_adapter,
            usd_file_path=args.usd_file,
            execution_hz=args.execution_hz,
            max_actions_per_inference=args.max_actions_per_inference,
            action_diff_threshold=args.action_diff_threshold,
            sim_already_running=args.sim_already_running
        )
        
        # Run main control loop
        controller.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Only close simulation app if we initialized it
        if not args.sim_already_running and simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()


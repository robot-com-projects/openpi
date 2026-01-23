"""
MuJoCo simulation replay functionality for bimanual inference.

This module provides functions and classes for running inference in MuJoCo
simulation with dataset replay, visualization, and plotting.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Optional imports
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .policy_adapters import PolicyAdapter
from .utils import robot_obs_to_numpy


def load_simulation_dataset(dataset_dir: str, episode_idx: int) -> tuple:
    """
    Load dataset for simulation replay.
    
    Args:
        dataset_dir: Path to dataset directory or repo_id
        episode_idx: Episode index to load
        
    Returns:
        tuple: (dataset, from_idx, to_idx, fps)
    """
    # Try to resolve repo_id and root
    if os.path.isdir(dataset_dir):
        # Local directory
        repo_id = os.path.basename(dataset_dir)
        root = dataset_dir
    else:
        # Assume it's a repo_id
        repo_id = dataset_dir
        root = None
    
    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=[episode_idx])
    
    # Get episode metadata
    episode_meta = dataset.meta.episodes[episode_idx]
    from_idx = int(float(episode_meta["dataset_from_index"]))
    to_idx = int(float(episode_meta["dataset_to_index"]))
    fps = float(dataset.meta.fps)
    
    return dataset, from_idx, to_idx, fps


def prepare_simulation_observation(sample: Dict[str, Any], policy_adapter: PolicyAdapter) -> Dict[str, Any]:
    """
    Prepare observation from dataset sample for policy inference.
    
    Args:
        sample: Dataset sample dictionary
        policy_adapter: Policy adapter to determine expected keys
        
    Returns:
        Dictionary with images, states, and prompt ready for inference
    """
    example = {}
    
    # Process images
    image_key_mapping = {
        'torso': 'observation/images/torso',
        'teleop_left': 'observation/images/teleop_left',
        'teleop_right': 'observation/images/teleop_right',
    }
    
    policy_image_keys = policy_adapter.get_image_keys()
    
    for robot_key, policy_key in image_key_mapping.items():
        # Try different possible keys
        for key in [robot_key, f"observation.images.{robot_key}", f"observation/images/{robot_key}"]:
            if key in sample:
                img = sample[key]
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                elif hasattr(img, 'cpu'):
                    img = img.cpu().numpy()
                else:
                    img = np.array(img)
                
                # Convert to HWC uint8 format
                if img.ndim == 3 and img.shape[0] == 3:  # C, H, W
                    img = np.transpose(img, (1, 2, 0))  # H, W, C
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # Use policy key if it matches, otherwise use the found key
                if policy_key in policy_image_keys:
                    example[policy_key] = img
                elif key in policy_image_keys:
                    example[key] = img
                break
    
    # Get state
    state_key = None
    for key in ['observation/state', 'observation.state', 'qpos']:
        if key in sample:
            state_key = key
            break
    
    if state_key:
        qpos = sample[state_key]
        if hasattr(qpos, 'numpy'):
            qpos = qpos.numpy()
        elif hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()
        else:
            qpos = np.array(qpos)
        
        policy_state_keys = policy_adapter.get_state_keys()
        if 'observation/state' in policy_state_keys:
            example['observation/state'] = qpos.astype(np.float32)
        elif 'observation.state' in policy_state_keys:
            example['observation.state'] = qpos.astype(np.float32)
        else:
            # Use first available state key
            if len(policy_state_keys) > 0:
                example[policy_state_keys[0]] = qpos.astype(np.float32)
    
    # Get prompt/task
    task_key = None
    for key in ['task', 'prompt', 'instruction']:
        if key in sample:
            task_key = key
            break
    
    if task_key:
        example['prompt'] = str(sample[task_key])
    else:
        example['prompt'] = policy_adapter.get_robot_task_string()
    
    return example


def apply_action_to_mujoco(current_action: np.ndarray, mj_data) -> None:
    """
    Apply action to MuJoCo simulation.
    
    Reorders action from [left(7), right(7)] to [right(7), left(7)] for MuJoCo.
    
    Args:
        current_action: Action array of shape [14] or [action_dim]
        mj_data: MuJoCo data object
    """
    if len(current_action) == 14:
        u_apply = np.concatenate([current_action[7:], current_action[:7]])
    else:
        u_apply = current_action
    
    mj_data.ctrl[:] = u_apply


def calculate_simulation_steps(dataset_idx: int, sim_to_idx: int, sample: Dict[str, Any],
                               sample_next: Optional[Dict[str, Any]], sim_fps: float, sim_dt: float) -> int:
    """
    Calculate number of simulation steps until next frame.
    
    Args:
        dataset_idx: Current dataset index
        sim_to_idx: End index of episode
        sample: Current sample
        sample_next: Next sample (if available)
        sim_fps: Dataset FPS
        sim_dt: Simulation timestep
        
    Returns:
        Number of simulation steps to take
    """
    if dataset_idx + 1 < sim_to_idx and sample_next is not None:
        ts = float(sample.get("timestamp", 0))
        ts_next = float(sample_next.get("timestamp", ts + 1.0 / sim_fps))
    else:
        ts = float(sample.get("timestamp", 0))
        ts_next = ts + 1.0 / max(1.0, sim_fps)
    
    n_steps = max(1, int(round((ts_next - ts) / sim_dt)))
    return n_steps


def log_to_rerun(example: Dict[str, Any], current_action: np.ndarray, u_gt: Optional[np.ndarray],
                 sample: Dict[str, Any], sim_frame_idx: int, sim_fps: float, mj_renderer, mj_model, mj_data):
    """
    Log data to Rerun visualization.
    
    Args:
        example: Observation example
        current_action: Predicted action
        u_gt: Ground truth action (optional)
        sample: Dataset sample
        sim_frame_idx: Current simulation frame index
        sim_fps: Dataset FPS
        mj_renderer: MuJoCo renderer (may be None)
        mj_model: MuJoCo model
        mj_data: MuJoCo data
    """
    if not RERUN_AVAILABLE:
        return
    
    # Set Rerun time
    ts = float(sample.get("timestamp", sim_frame_idx / sim_fps))
    rr.set_time_seconds("timestamp", ts)
    
    # Log images
    for key, img in example.items():
        if isinstance(img, np.ndarray) and len(img.shape) == 3:
            rr.log(f"dataset/{key}", rr.Image(img))
    
    # Log actions
    if u_gt is not None:
        u_pred = np.asarray(current_action, dtype=np.float32).reshape(-1)
        u_gt_flat = np.asarray(u_gt, dtype=np.float32).reshape(-1)
        n = min(len(u_pred), len(u_gt_flat))
        for i in range(n):
            rr.log(f"actions/gt/joint_{i}", rr.Scalars([float(u_gt_flat[i])]))
            rr.log(f"actions/pred/joint_{i}", rr.Scalars([float(u_pred[i])]))
    
    # Render and log simulation image
    if mj_renderer is not None:
        try:
            mj_renderer.update_scene(mj_data, camera=0)
            sim_image = mj_renderer.render()
            rr.log("sim", rr.Image(sim_image))
        except Exception:
            pass


def get_ground_truth_action(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Extract ground truth action from dataset sample.
    
    Args:
        sample: Dataset sample
        
    Returns:
        Ground truth action array or None
    """
    from lerobot.utils.constants import ACTION
    if ACTION in sample:
        u_gt_val = sample[ACTION]
        if hasattr(u_gt_val, 'numpy'):
            return u_gt_val.numpy()
        elif hasattr(u_gt_val, 'cpu'):
            return u_gt_val.cpu().numpy()
        else:
            return np.array(u_gt_val)
    return None


def plot_action_comparison(predicted_actions: list, ground_truth_actions: list,
                          action_timestamps: list, episode_idx: int, plots_dir: Path):
    """
    Plot predicted vs ground truth actions and save to plots directory.
    
    Args:
        predicted_actions: List of predicted action arrays
        ground_truth_actions: List of ground truth action arrays
        action_timestamps: List of timestamps
        episode_idx: Episode index for filename
        plots_dir: Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE or len(predicted_actions) == 0:
        return
    
    predicted = np.array(predicted_actions)
    ground_truth = np.array(ground_truth_actions)
    timestamps = np.array(action_timestamps)
    
    # Normalize timestamps to start from 0
    if len(timestamps) > 0:
        timestamps = timestamps - timestamps[0]
    
    n_joints = min(predicted.shape[1], ground_truth.shape[1])
    
    # Joint names for bimanual robot (7 joints per arm)
    joint_names = [
        'j0 (shoulder_pan)', 'j1 (shoulder_lift)', 'j2 (elbow)', 
        'j3 (wrist_1)', 'j4 (wrist_2)', 'j5 (wrist_3)', 'j6 (gripper)'
    ]
    
    # Split into left and right arms (assuming 14 joints total: 7 left + 7 right)
    n_joints_per_arm = 7
    left_arm_joints = min(n_joints_per_arm, n_joints // 2)
    right_arm_joints = min(n_joints_per_arm, n_joints - left_arm_joints)
    
    # Create figure with 2 columns: left arm and right arm
    fig = plt.figure(figsize=(16, 2 * max(left_arm_joints, right_arm_joints)))
    fig.suptitle(f'Predicted vs Ground Truth Actions - Episode {episode_idx}', fontsize=16, y=0.995)
    
    # Left arm plots (left column)
    for i in range(left_arm_joints):
        ax = plt.subplot(max(left_arm_joints, right_arm_joints), 2, 2 * i + 1)
        ax.plot(timestamps, predicted[:, i], label='Predicted', alpha=0.7, linewidth=1.5, color='blue')
        ax.plot(timestamps, ground_truth[:, i], label='Ground Truth', alpha=0.7, linewidth=1.5, linestyle='--', color='orange')
        ax.set_ylabel(f'Left {joint_names[i]}\n(rad)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        if i < left_arm_joints - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    # Right arm plots (right column)
    for i in range(right_arm_joints):
        ax = plt.subplot(max(left_arm_joints, right_arm_joints), 2, 2 * i + 2)
        right_idx = left_arm_joints + i
        ax.plot(timestamps, predicted[:, right_idx], label='Predicted', alpha=0.7, linewidth=1.5, color='blue')
        ax.plot(timestamps, ground_truth[:, right_idx], label='Ground Truth', alpha=0.7, linewidth=1.5, linestyle='--', color='orange')
        ax.set_ylabel(f'Right {joint_names[i]}\n(rad)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        if i < right_arm_joints - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=10)
    
    # Add column titles
    fig.text(0.25, 0.98, 'Right Arm', ha='center', fontsize=12, weight='bold')
    fig.text(0.75, 0.98, 'Left Arm', ha='center', fontsize=12, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save plot to plots directory
    plot_filename = plots_dir / f'action_comparison_episode_{episode_idx}.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Action comparison plot saved to: {plot_filename}")
    
    # Close figure to free memory
    plt.close(fig)


class SimulationReplayController:
    """
    Controller for MuJoCo simulation replay with inference.
    
    This class manages the simulation replay loop, inference, and visualization.
    """
    
    def __init__(
        self,
        policy_adapter: PolicyAdapter,
        mjcf_path: str,
        dataset_dir: str,
        episode_idx: int = 0,
        use_rerun: bool = False,
        max_actions_per_inference: Optional[int] = None,
    ):
        """
        Initialize simulation replay controller.
        
        Args:
            policy_adapter: Policy adapter for running inference
            mjcf_path: Path to MuJoCo XML file
            dataset_dir: Path to dataset directory or repo_id
            episode_idx: Episode index to replay
            use_rerun: Whether to use Rerun for visualization
            max_actions_per_inference: Max actions to execute before next inference
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is not available. Install with: pip install mujoco")
        
        self.policy_adapter = policy_adapter
        self.mjcf_path = mjcf_path
        self.dataset_dir = dataset_dir
        self.episode_idx = episode_idx
        self.use_rerun = use_rerun and RERUN_AVAILABLE
        self.plot_actions = MATPLOTLIB_AVAILABLE
        
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_renderer = None
        self.use_renderer = False
        print(f"MuJoCo model loaded from: {mjcf_path}")
        
        # Load dataset
        self.sim_dataset, self.sim_from_idx, self.sim_to_idx, self.sim_fps = load_simulation_dataset(
            dataset_dir, episode_idx
        )
        self.sim_dt = self.mj_model.opt.timestep
        print(f"Dataset loaded: episode {episode_idx}")
        print(f"Episode range: {self.sim_from_idx} to {self.sim_to_idx}")
        print(f"Dataset FPS: {self.sim_fps}")
        
        # Simulation state
        self.sim_frame_idx = 0
        self.sim_action_buffer = []
        self.sim_action_index = 0
        
        # Set default max_actions_per_inference
        if max_actions_per_inference is None:
            try:
                action_horizon = getattr(policy_adapter.policy, 'action_horizon', None)
                if action_horizon is None:
                    if hasattr(policy_adapter.policy, 'config'):
                        action_horizon = getattr(policy_adapter.policy.config, 'action_horizon', 12)
                    else:
                        action_horizon = 12
                self.max_actions_per_inference = action_horizon
            except Exception:
                self.max_actions_per_inference = 12
        else:
            self.max_actions_per_inference = max_actions_per_inference
        
        # Store actions for plotting
        if self.plot_actions:
            self.predicted_actions = []
            self.ground_truth_actions = []
            self.action_timestamps = []
            self.plots_dir = Path("action_plots")
            self.plots_dir.mkdir(exist_ok=True)
            print(f"Action plots will be saved to: {self.plots_dir.absolute()}")
        
        # Initialize Rerun if requested
        if self.use_rerun:
            rr.init(f"bimanual_inference/episode_{episode_idx}", spawn=True)
            print("Rerun visualization initialized")
    
    def run_step(self) -> bool:
        """
        Run one step of simulation replay.
        
        Returns:
            True if simulation should continue, False if finished
        """
        if self.sim_frame_idx >= (self.sim_to_idx - self.sim_from_idx):
            print("🛑 Simulation complete - episode finished")
            return False
        
        # Get current frame from dataset
        dataset_idx = self.sim_from_idx + self.sim_frame_idx
        sample = self.sim_dataset[int(dataset_idx)]
        
        # Prepare observation
        example = prepare_simulation_observation(sample, self.policy_adapter)
        
        # Log to Rerun
        if self.use_rerun:
            u_gt = get_ground_truth_action(sample) if self.use_rerun else None
            log_to_rerun(example, None, u_gt, sample, self.sim_frame_idx, self.sim_fps,
                        self.mj_renderer, self.mj_model, self.mj_data)
        
        # Run inference if needed
        actions_limit = min(self.max_actions_per_inference, len(self.sim_action_buffer)) if len(self.sim_action_buffer) > 0 else 0
        if len(self.sim_action_buffer) == 0 or self.sim_action_index >= actions_limit:
            print(f"\n{'='*60}")
            print(f"Running inference (sim_frame_idx={self.sim_frame_idx})...")
            
            # Separate images and states
            images = {k: v for k, v in example.items() if isinstance(v, np.ndarray) and len(v.shape) == 3}
            states = {k: v for k, v in example.items() if isinstance(v, np.ndarray) and len(v.shape) == 1}
            
            result = self.policy_adapter.run_inference(
                images=images,
                states=states,
                robot_task_string=example.get('prompt')
            )
            
            # Handle different return formats
            if isinstance(result, np.ndarray):
                actions_batch = result
            elif isinstance(result, dict) and "actions" in result:
                actions_batch = result["actions"]
            else:
                actions_batch = result
            
            self.sim_action_buffer = actions_batch
            self.sim_action_index = 0
            print(f"Inference complete: got {actions_batch.shape[0]} actions")
            print(f"{'='*60}\n")
        
        # Get current action from buffer
        current_action = self.sim_action_buffer[self.sim_action_index]
        self.sim_action_index += 1
        
        # Get ground truth action for comparison
        u_gt = None
        if self.use_rerun or self.plot_actions:
            u_gt = get_ground_truth_action(sample)
        
        # Store actions for plotting
        if self.plot_actions:
            u_pred = np.asarray(current_action, dtype=np.float32).reshape(-1)
            if u_gt is not None:
                u_gt_flat = np.asarray(u_gt, dtype=np.float32).reshape(-1)
                self.ground_truth_actions.append(u_gt_flat.copy())
            else:
                u_gt_flat = np.zeros_like(u_pred)
                self.ground_truth_actions.append(u_gt_flat.copy())
            self.predicted_actions.append(u_pred.copy())
            ts = float(sample.get("timestamp", self.sim_frame_idx / self.sim_fps))
            self.action_timestamps.append(ts)
        
        # Log actions to Rerun
        if self.use_rerun and u_gt is not None:
            log_to_rerun(example, current_action, u_gt, sample, self.sim_frame_idx, self.sim_fps,
                        self.mj_renderer, self.mj_model, self.mj_data)
        
        # Apply action to MuJoCo simulation
        apply_action_to_mujoco(current_action, self.mj_data)
        
        # Calculate number of simulation steps until next frame
        sample_next = None
        if dataset_idx + 1 < self.sim_to_idx:
            sample_next = self.sim_dataset[int(dataset_idx + 1)]
        n_steps = calculate_simulation_steps(
            dataset_idx, self.sim_to_idx, sample, sample_next, self.sim_fps, self.sim_dt
        )
        
        # Step simulation
        for _ in range(n_steps):
            mujoco.mj_step(self.mj_model, self.mj_data)
        
        # Try to create renderer for Rerun if needed
        if self.use_rerun and self.mj_renderer is None:
            try:
                self.mj_renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
                self.use_renderer = True
            except Exception:
                self.use_renderer = False
        
        # Render and log to Rerun if enabled
        if self.use_rerun and self.use_renderer and self.mj_renderer is not None:
            try:
                self.mj_renderer.update_scene(self.mj_data, camera=0)
                sim_image = self.mj_renderer.render()
                rr.log("sim", rr.Image(sim_image))
            except Exception:
                pass
        
        print(f"Sim step {self.sim_frame_idx}: Action applied, {n_steps} sim steps")
        
        # Advance frame
        self.sim_frame_idx += 1
        return True
    
    def run(self):
        """Run the complete simulation replay."""
        print("🤖 Starting simulation replay...")
        while self.run_step():
            import time
            time.sleep(0.01)  # Small delay for visualization
        print("✓ Simulation replay complete")
        
        # Automatically plot actions when in simulation mode
        if self.plot_actions:
            print("Generating action comparison plot...")
            plot_action_comparison(
                self.predicted_actions,
                self.ground_truth_actions,
                self.action_timestamps,
                self.episode_idx,
                self.plots_dir
            )


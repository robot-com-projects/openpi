"""
Recording functionality for inference samples.

This module provides utilities for recording inference samples to LeRobot dataset
format, similar to how lerobot records during inference.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.utils.constants import OBS_STR, ACTION


class InferenceRecorder:
    """
    Records inference samples to LeRobot dataset format.
    
    This allows viewing cameras and observations later, similar to how
    lerobot records during inference evaluation.
    """
    
    def __init__(
        self,
        robot,
        repo_id: str = "inference_samples",
        fps: int = 30,
        use_videos: bool = True,
        batch_encoding_size: int = 1,
    ):
        """
        Initialize inference recorder.
        
        Args:
            robot: Robot instance (must have action_features and observation_features)
            repo_id: Dataset repository ID (will append timestamp)
            fps: Recording FPS
            use_videos: Whether to use video encoding
            batch_encoding_size: Batch size for video encoding
        """
        self.robot = robot
        self.fps = fps
        self.use_videos = use_videos
        self.batch_encoding_size = batch_encoding_size
        
        # Create timestamped repo_id
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        if "/" in repo_id:
            # If repo_id has format "user/repo", append timestamp
            parts = repo_id.split("/")
            self.repo_id = f"{parts[0]}/{parts[1]}_{ts}" if len(parts) > 1 else f"{repo_id}_{ts}"
        else:
            self.repo_id = f"{repo_id}_{ts}"
        
        # Initialize processors
        self.processors = make_default_processors()
        
        # Dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        dataset_features = {**action_features, **obs_features}
        
        # Create dataset
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=self.use_videos,
            image_writer_threads=4,
            batch_encoding_size=self.batch_encoding_size,
        )
        
        print(f"✓ Inference recorder initialized: {self.repo_id}")
        self.episode_started = False
    
    def start_episode(self, task: str = "inference"):
        """
        Start a new episode for recording.
        
        Args:
            task: Task description for this episode
        """
        if self.episode_started:
            # Save previous episode if one was started
            self.save_episode()
        
        self.current_task = task
        self.episode_started = True
        print(f"📹 Started recording episode: {task}")
    
    def record_frame(self, observation: Dict[str, Any], action: np.ndarray, task: Optional[str] = None):
        """
        Record a single frame during inference.
        
        Args:
            observation: Observation dictionary from robot (keys like 'torso', 'teleop_left', 'right.j0.pos', etc.)
            action: Action array that was executed (shape [14] for bimanual: [left(7), right(7)])
            task: Optional task description (uses current task if not provided)
        """
        if not self.episode_started:
            # Auto-start episode if not started
            self.start_episode(task or "inference")
        
        # Build observation frame using dataset features
        observation_frame = {}
        
        # Map observation keys to dataset feature keys
        for key, value in observation.items():
            # Check if this key matches a dataset feature
            if key in self.dataset.features:
                # Direct match (e.g., camera keys like 'torso', 'teleop_left')
                observation_frame[key] = value
            elif key.endswith('.pos'):
                # Joint position key (e.g., 'right.j0.pos')
                # Dataset features use 'observation/state' or individual keys
                # For now, we'll use the key as-is if it's in features, otherwise skip
                # The dataset will handle mapping via hw_to_dataset_features
                if key in self.dataset.features:
                    observation_frame[key] = float(value) if not isinstance(value, np.ndarray) else float(value.item())
            else:
                # Try to map camera keys
                if isinstance(value, np.ndarray) and len(value.shape) == 3:
                    # Image - check if mapped key exists in features
                    if key in self.dataset.features:
                        observation_frame[key] = value
        
        # Build action frame
        action_frame = {}
        if isinstance(action, np.ndarray):
            # If action is a simple array, map to action features
            if len(action) == 14:  # Bimanual: 7 left + 7 right
                # Map to action format expected by dataset
                # Check dataset action features to see format
                action_keys = [k for k in self.dataset.features.keys() if k.startswith('action/')]
                if action_keys:
                    # Use dataset action feature format
                    for i in range(7):
                        left_key = f"action/left.j{i}.pos"
                        if left_key in self.dataset.features:
                            action_frame[left_key] = float(action[i])
                    for i in range(7):
                        right_key = f"action/right.j{i}.pos"
                        if right_key in self.dataset.features:
                            action_frame[right_key] = float(action[i + 7])
                else:
                    # Fallback: use generic action key
                    action_frame[ACTION] = action
            else:
                # Use generic action key
                action_frame[ACTION] = action
        elif isinstance(action, dict):
            action_frame = action
        else:
            action_frame[ACTION] = np.array(action)
        
        # Combine into frame
        frame = {
            **observation_frame,
            **action_frame,
            "task": task or self.current_task,
        }
        
        # Add frame to dataset
        self.dataset.add_frame(frame)
    
    def save_episode(self):
        """Save the current episode."""
        if not self.episode_started:
            return
        
        try:
            self.dataset.save_episode()
            print(f"✓ Episode saved to {self.repo_id}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save episode: {e}")
        finally:
            self.episode_started = False
    
    def finalize(self):
        """Finalize the dataset (save any remaining data)."""
        if self.episode_started:
            self.save_episode()
        
        try:
            self.dataset.finalize()
            print(f"✓ Dataset finalized: {self.repo_id}")
        except Exception as e:
            print(f"⚠️  Warning: Could not finalize dataset: {e}")
    
    def push_to_hub(self):
        """Push dataset to HuggingFace Hub."""
        try:
            self.finalize()
            self.dataset.push_to_hub()
            print(f"✓ Dataset pushed to hub: {self.repo_id}")
        except Exception as e:
            print(f"⚠️  Warning: Could not push to hub: {e}")


"""
Policy adapter interfaces for different policy APIs.

This module provides abstract interfaces and adapters for different
policy backends (pi.policy_api vs openpi.policies).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class PolicyAdapter(ABC):
    """
    Abstract base class for policy adapters.
    
    This provides a unified interface for different policy backends.
    """
    
    @abstractmethod
    def get_image_keys(self):
        """Get list of image keys expected by the policy."""
        pass
    
    @abstractmethod
    def get_state_keys(self):
        """Get list of state keys expected by the policy."""
        pass
    
    @abstractmethod
    def get_robot_task_string(self) -> str:
        """Get the robot task string/prompt."""
        pass
    
    @abstractmethod
    def run_inference(self, images: Dict[str, np.ndarray], states: Dict[str, np.ndarray], 
                     robot_task_string: Optional[str] = None, rng: Any = None) -> np.ndarray:
        """
        Run inference on the policy.
        
        Args:
            images: Dictionary mapping image keys to numpy arrays
            states: Dictionary mapping state keys to numpy arrays
            robot_task_string: Optional task description string
            rng: Random number generator (if needed)
            
        Returns:
            numpy array of actions with shape [action_horizon, action_dim]
        """
        pass


class PiPolicyAdapter(PolicyAdapter):
    """
    Adapter for pi.policy_api backend.
    
    This adapter wraps the old policy_api interface.
    """
    
    def __init__(self, policy, rng):
        """
        Initialize adapter.
        
        Args:
            policy: Policy object from pi.policy_api
            rng: JAX random number generator
        """
        self.policy = policy
        self.rng = rng
        
        # Import here to avoid requiring pi dependencies for other adapters
        from pi.policy_api import InputData, create_input, run_policy
        
        self.InputData = InputData
        self.create_input = create_input
        self.run_policy = run_policy
    
    def get_image_keys(self):
        """Get list of image keys expected by the policy."""
        return self.policy.image_keys
    
    def get_state_keys(self):
        """Get list of state keys expected by the policy."""
        return self.policy.state_keys
    
    def get_robot_task_string(self) -> str:
        """Get the robot task string/prompt."""
        return self.policy.robot_task_string
    
    def run_inference(self, images: Dict[str, np.ndarray], states: Dict[str, np.ndarray],
                     robot_task_string: Optional[str] = None, rng: Any = None) -> np.ndarray:
        """Run inference using pi.policy_api."""
        if robot_task_string is None:
            robot_task_string = self.policy.robot_task_string
        
        input_data = self.InputData(
            images=images,
            states=states,
            robot_task_string=robot_task_string
        )
        input_steps = self.create_input(self.policy, input_data)
        
        if rng is None:
            rng = self.rng
        
        import jax
        if rng is not None:
            rng, call_rng = jax.random.split(rng)
            # Update stored RNG
            self.rng = rng
        else:
            call_rng = self.rng
            # Split and update
            self.rng, call_rng = jax.random.split(self.rng)
        
        actions_batch = self.run_policy(self.policy, rng=call_rng, input_data=input_steps)
        
        return actions_batch


class OpenPiPolicyAdapter(PolicyAdapter):
    """
    Adapter for openpi.policies backend.
    
    This adapter wraps the new openpi policy interface.
    """
    
    def __init__(self, policy):
        """
        Initialize adapter.
        
        Args:
            policy: Policy object from openpi.policies
        """
        self.policy = policy
    
    def get_image_keys(self):
        """Get list of image keys expected by the policy."""
        # OpenPi policies may have different key structures
        # This is a placeholder - adjust based on actual policy structure
        if hasattr(self.policy, 'image_keys'):
            return self.policy.image_keys
        # Default image keys for openpi format
        return ['observation/images/torso', 'observation/images/teleop_left', 'observation/images/teleop_right']
    
    def get_state_keys(self):
        """Get list of state keys expected by the policy."""
        if hasattr(self.policy, 'state_keys'):
            return self.policy.state_keys
        return ['observation/state']
    
    def get_robot_task_string(self) -> str:
        """Get the robot task string/prompt."""
        if hasattr(self.policy, '_metadata') and 'robot_task_string' in self.policy._metadata:
            return self.policy._metadata['robot_task_string']
        return "Fold the shirt"  # Default
    
    def run_inference(self, images: Dict[str, np.ndarray], states: Dict[str, np.ndarray],
                     robot_task_string: Optional[str] = None, rng: Any = None) -> np.ndarray:
        """Run inference using openpi.policies."""
        # Create example dict in the format expected by policy.infer()
        example = {}
        example.update(images)
        example.update(states)
        
        if robot_task_string is not None:
            example['prompt'] = robot_task_string
        elif 'prompt' not in example:
            example['prompt'] = self.get_robot_task_string()
        
        # Run inference
        result = self.policy.infer(example)
        actions_batch = result["actions"]
        
        return actions_batch


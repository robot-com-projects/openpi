"""
Utility functions for bimanual robot control.

This module provides helper functions for:
- CAN interface checking
- Process management (gello processes)
- Image format conversion
"""

import subprocess
import os
import numpy as np


def check_can_interface(interface: str) -> bool:
    """Check if a CAN interface exists and is available"""
    try:
        result = subprocess.run(['ip', 'link', 'show', interface],
                              capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return False
        if 'state UP' in result.stdout or 'state UNKNOWN' in result.stdout:
            return True
        else:
            print(f"Warning: CAN interface {interface} exists but is not UP")
            return False
    except Exception as e:
        print(f"Error checking CAN interface {interface}: {e}")
        return False


def check_all_can_interfaces() -> bool:
    """Check if all required CAN interfaces exist"""
    required_interfaces = [
        'can_follower_r',
        'can_leader_r',
        'can_follower_l',
        'can_leader_l'
    ]
    
    missing_interfaces = []
    for interface in required_interfaces:
        if not check_can_interface(interface):
            missing_interfaces.append(interface)
    
    if missing_interfaces:
        raise RuntimeError(f"Missing or unavailable CAN interfaces: {', '.join(missing_interfaces)}")
    
    print("✓ All CAN interfaces are available")
    return True


def launch_gello_process(can_channel: str, gripper: str, server_port: int, thirdparty_i2rt_path: str) -> subprocess.Popen:
    """
    Launch a single follower gello process.
    
    Args:
        can_channel: CAN channel name (e.g., 'can_follower_r')
        gripper: Gripper type (e.g., 'linear_4310')
        server_port: Server port number
        thirdparty_i2rt_path: Path to thirdparty/i2rt directory
        
    Returns:
        subprocess.Popen object or None if failed
    """
    python_path = "python"
    script_path = os.path.join(thirdparty_i2rt_path, "scripts", "minimum_gello.py")
    
    cmd = [python_path, script_path,
           "--can_channel", can_channel,
           "--gripper", gripper,
           "--mode", "follower",
           "--server_port", str(server_port)]
    
    print(f"Starting: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error starting process for {can_channel}: {e}")
        return None


def robot_obs_to_numpy(image) -> np.ndarray:
    """
    Convert robot observation image to numpy format expected by policy.
    
    Args:
        image: Image in various formats (numpy array, tensor, etc.)
        
    Returns:
        numpy array in HWC format with uint8 dtype
    """
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


def hf_image_to_numpy(image) -> np.ndarray:
    """
    Convert HuggingFace dataset image to numpy format.
    
    Args:
        image: Image from HuggingFace dataset
        
    Returns:
        numpy array in HWC format with uint8 dtype
    """
    if not isinstance(image, np.ndarray):
        image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image


def stack(i):
    """
    Convert tensor-like object to numpy array.
    
    Args:
        i: Input that may be a string or tensor-like object
        
    Returns:
        numpy array or original string
    """
    if isinstance(i, str):
        return i
    return i.numpy()


"""
Utility Functions for Biped Environment

This module contains utility functions for quaternion operations,
rotation matrix calculations, and axis orientation analysis.
"""

import torch
import numpy as np


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    q: quaternion in [w, x, y, z] format (Genesis format)
    Returns: 3x3 rotation matrix
    
    Can handle both torch tensors and numpy arrays.
    If input is a torch tensor, output will be a torch tensor on the same device.
    """
    # Handle torch tensors
    if isinstance(q, torch.Tensor):
        device = q.device
        dtype = q.dtype
        
        # Ensure quaternion is normalized
        q = q / torch.norm(q, dim=-1, keepdim=True)
        
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Rotation matrix from quaternion
        R = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
        ], dim=-2)
        
        return R
    
    # Handle numpy arrays (fallback for compatibility)
    else:
        # Move tensor to CPU and convert to numpy if needed
        if hasattr(q, 'cpu'):
            q = q.cpu().numpy()
        
        # Normalize quaternion
        q = q / np.linalg.norm(q)
        
        w, x, y, z = q
        
        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R


def get_axis_orientation_wrt_world_z(rotation_matrix, axis_index):
    """
    Get the orientation of a local axis relative to the world Z-axis.
    
    Args:
        rotation_matrix: 3x3 rotation matrix of the link (torch tensor or numpy array)
        axis_index: 0 for X-axis, 1 for Y-axis, 2 for Z-axis
        
    Returns:
        angle_degrees: Angle between the specified axis and world Z-axis in degrees
        dot_product: Dot product value (cosine of the angle)
    """
    # Handle torch tensors
    if isinstance(rotation_matrix, torch.Tensor):
        device = rotation_matrix.device
        dtype = rotation_matrix.dtype
        
        # World Z-axis vector
        world_z = torch.tensor([0, 0, 1], device=device, dtype=dtype)
        
        # Extract the specified axis from rotation matrix
        # Column vectors of rotation matrix represent the local axes in world coordinates
        if rotation_matrix.dim() == 3:  # Batch of rotation matrices
            local_axis = rotation_matrix[..., :, axis_index]  # X=0, Y=1, Z=2
            
            # Calculate dot product (cosine of angle between vectors)
            dot_product = torch.sum(local_axis * world_z.unsqueeze(0), dim=-1)
        else:  # Single rotation matrix
            local_axis = rotation_matrix[:, axis_index]  # X=0, Y=1, Z=2
            
            # Calculate dot product (cosine of angle between vectors)
            dot_product = torch.dot(local_axis, world_z)
        
        # Clamp dot product to valid range for arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle_radians = torch.acos(dot_product)
        angle_degrees = torch.rad2deg(angle_radians)
        
        return angle_degrees, dot_product
    
    # Handle numpy arrays (fallback for compatibility)
    else:
        # World Z-axis vector
        world_z = np.array([0, 0, 1])
        
        # Extract the specified axis from rotation matrix
        # Column vectors of rotation matrix represent the local axes in world coordinates
        local_axis = rotation_matrix[:, axis_index]  # X=0, Y=1, Z=2
        
        # Calculate dot product (cosine of angle between vectors)
        dot_product = np.dot(local_axis, world_z)
        
        # Clamp dot product to valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle_radians = np.arccos(dot_product)
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees, dot_product


def get_foot_axis_dot_products(foot_quaternions, axis_index=1):
    """
    Get the dot products of foot axes (default Y-axis) with world Z-axis for reward calculation.
    
    Args:
        foot_quaternions: Tensor of shape (num_envs, 2, 4) for [left_foot, right_foot] quaternions
        axis_index: 0 for X-axis, 1 for Y-axis, 2 for Z-axis (default: 1 for Y-axis)
        
    Returns:
        left_dot: Dot product of left foot axis with world Z-axis
        right_dot: Dot product of right foot axis with world Z-axis
    """
    # Convert quaternions to rotation matrices
    left_rot_matrix = quaternion_to_rotation_matrix(foot_quaternions[:, 0])  # (num_envs, 3, 3)
    right_rot_matrix = quaternion_to_rotation_matrix(foot_quaternions[:, 1])  # (num_envs, 3, 3)
    
    # Get dot products for the specified axis
    _, left_dot = get_axis_orientation_wrt_world_z(left_rot_matrix, axis_index)
    _, right_dot = get_axis_orientation_wrt_world_z(right_rot_matrix, axis_index)
    
    return left_dot, right_dot

"""
Reward Functions Module for Biped Environment

This module contains all reward calculation functions for the biped robot environment.
Each reward function is optimized for performance using pre-allocated buffers and
vectorized operations.

OPTIMIZATION FEATURES:
- Pre-allocated reward computation buffers
- In-place tensor operations using out= parameter
- Vectorized calculations to minimize temporary tensor creation
- Efficient exponential and trigonometric operations
"""

import torch
import numpy as np
import genesis as gs
from utils import quaternion_to_rotation_matrix, get_axis_orientation_wrt_world_z, get_foot_axis_dot_products


class RewardFunctions:
    """Handles all reward calculations for the biped environment"""
    
    def __init__(self, num_envs, num_actions, reward_cfg, device):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.reward_cfg = reward_cfg
        self.device = device
        
        # Pre-allocate reward computation buffers for performance
        self.reward_buffers = {
            'lin_vel_error': torch.zeros((self.num_envs,), device=device, dtype=gs.tc_float),
            'orientation_error': torch.zeros((self.num_envs,), device=device, dtype=gs.tc_float),
            'height_error': torch.zeros((self.num_envs,), device=device, dtype=gs.tc_float),
            'action_diff': torch.zeros((self.num_envs, self.num_actions), device=device, dtype=gs.tc_float),
            'dof_diff': torch.zeros((self.num_envs, self.num_actions), device=device, dtype=gs.tc_float),
            'foot_quaternions': torch.zeros((self.num_envs, 2, 4), device=device, dtype=gs.tc_float),
        }
    
    def reward_lin_vel_z(self, base_lin_vel):
        """Penalize vertical velocity"""
        return torch.square(base_lin_vel[:, 2])

    def reward_action_rate(self, last_actions, actions):
        """Penalize rapid action changes using pre-allocated buffer"""
        torch.sub(last_actions, actions, out=self.reward_buffers['action_diff'])
        torch.square(self.reward_buffers['action_diff'], out=self.reward_buffers['action_diff'])
        return torch.sum(self.reward_buffers['action_diff'], dim=1)

    def reward_similar_to_default(self, dof_pos, default_dof_pos):
        """Encourage staying close to default joint positions"""
        torch.sub(dof_pos, default_dof_pos, out=self.reward_buffers['dof_diff'])
        torch.abs(self.reward_buffers['dof_diff'], out=self.reward_buffers['dof_diff'])
        return torch.sum(self.reward_buffers['dof_diff'], dim=1)

    def reward_forward_velocity(self, base_lin_vel):
        """Reward forward movement towards target velocity"""
        v_target = self.reward_cfg.get("forward_velocity_target", 0.5)
        vel_error = torch.square(base_lin_vel[:, 0] - v_target)
        sigma = self.reward_cfg.get("tracking_sigma", 0.25)
        return torch.exp(-vel_error / sigma)

    def reward_tracking_lin_vel_x(self, commands, base_lin_vel):
        """Track commanded linear velocity in X direction"""
        torch.sub(commands[:, 0], base_lin_vel[:, 0], out=self.reward_buffers['lin_vel_error'])
        torch.square(self.reward_buffers['lin_vel_error'], out=self.reward_buffers['lin_vel_error'])
        return torch.exp(-self.reward_buffers['lin_vel_error'] / self.reward_cfg["tracking_sigma"])

    def reward_tracking_lin_vel_y(self, commands, base_lin_vel):
        """Track commanded linear velocity in Y direction"""
        torch.sub(commands[:, 1], base_lin_vel[:, 1], out=self.reward_buffers['lin_vel_error'])
        torch.square(self.reward_buffers['lin_vel_error'], out=self.reward_buffers['lin_vel_error'])
        return torch.exp(-self.reward_buffers['lin_vel_error'] / self.reward_cfg["tracking_sigma"])

    def reward_alive_bonus(self):
        """Provide a constant alive bonus"""
        return torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def reward_fall_penalty(self, base_euler, env_cfg):
        """Penalize falling based on roll and pitch angles"""
        fall_condition = (
            (torch.abs(base_euler[:, 0]) > env_cfg.get("fall_roll_threshold", 30.0)) |
            (torch.abs(base_euler[:, 1]) > env_cfg.get("fall_pitch_threshold", 30.0))
        )
        return torch.where(
            fall_condition,
            torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_float),
            torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        )

    def reward_torso_stability(self, base_euler):
        """Reward torso stability using orientation error"""
        orientation_error = torch.sum(torch.square(base_euler[:, :2]), dim=1)
        k_stability = self.reward_cfg.get("stability_factor", 1.0)
        return torch.exp(-k_stability * orientation_error)

    def reward_height_maintenance(self, base_pos):
        """Reward maintaining target height"""
        z_target = self.reward_cfg.get("height_target", 0.35)
        torch.sub(z_target, base_pos[:, 2], out=self.reward_buffers['height_error'])
        torch.square(self.reward_buffers['height_error'], out=self.reward_buffers['height_error'])
        return -self.reward_buffers['height_error']

    def reward_joint_movement(self, dof_vel):
        """Reward joint movement up to a threshold"""
        joint_vel_magnitude = torch.sum(torch.abs(dof_vel), dim=1)
        movement_threshold = self.reward_cfg.get("movement_threshold", 0.1)
        movement_scale = self.reward_cfg.get("movement_scale", 1.0)

        return torch.clamp(joint_vel_magnitude * movement_scale, 0.0, movement_threshold)
    

    
    
    def reward_foot_parallelism(self, foot_quaternions):
        """
        Reward for foot parallelism to ground using Y-axis orientation.
        For each foot, reward = exp(abs(left_y_dot) * k) where k is a scaling factor.
        
        Args:
            foot_quaternions: Tensor of shape (num_envs, 2, 4) for [left_foot, right_foot] quaternions
            
        Returns:
            Combined reward for both feet parallelism
        """
        # Get parallelism parameters from config
        k_factor = self.reward_cfg.get("foot_parallelism_k", 1.0)  # Scaling factor for exponential

        left_rot_matrix = quaternion_to_rotation_matrix(foot_quaternions[:, 0])  # (num_envs, 3, 3)
        right_rot_matrix = quaternion_to_rotation_matrix(foot_quaternions[:, 1])  # (num_envs, 3, 3)
        
        # Get dot products for the specified axis
        _, left_y_dot = get_axis_orientation_wrt_world_z(left_rot_matrix, axis_index = 0)
        _, right_x_dot = get_axis_orientation_wrt_world_z(right_rot_matrix, axis_index = 1)
        
        # Get dot products of foot Y-axes with world Z-axis
        # left_y_dot, _ = get_foot_axis_dot_products(foot_quaternions, axis_index=1)
        # _,right_x_dot = get_foot_axis_dot_products(foot_quaternions, axis_index=1)
        print(foot_quaternions,left_y_dot,  right_x_dot)

        # Calculate rewards: exp(abs(dot_product) * k)
        # Higher abs(dot_product) means more aligned with world Z (more parallel to ground)
        left_reward = torch.exp(-torch.abs(left_y_dot) * k_factor)
        right_reward = torch.exp(-torch.abs(right_x_dot) * k_factor)
        
        # Combine left and right foot rewards
        total_reward = (left_reward + right_reward) / 2.0
        
        return total_reward
    
    def compute_rewards(self, base_lin_vel, actions, last_actions, dof_pos, default_dof_pos, commands, base_euler, base_pos, dof_vel, episode_length_buf, dt, joint_torques, foot_quaternions=None):
        """
        Calculates and returns a dictionary of all reward components.
        This consolidated method is called by the optimized environment for performance.
        """
        rewards = {
            'lin_vel_z': self.reward_lin_vel_z(base_lin_vel),
            'action_rate': self.reward_action_rate(last_actions, actions),
            'similar_to_default': self.reward_similar_to_default(dof_pos, default_dof_pos),
            'tracking_lin_vel_x': self.reward_tracking_lin_vel_x(commands, base_lin_vel),
            'tracking_lin_vel_y': self.reward_tracking_lin_vel_y(commands, base_lin_vel),
            'alive_bonus': self.reward_alive_bonus(),
            'fall_penalty': self.reward_fall_penalty(base_euler, self.reward_cfg), # Assuming env_cfg was passed as reward_cfg
            'height_maintenance': self.reward_height_maintenance(base_pos),
            'joint_movement': self.reward_joint_movement(dof_vel),
            'foot_parallelism': self.reward_foot_parallelism(foot_quaternions) if foot_quaternions is not None else torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float),
        }
        return rewards

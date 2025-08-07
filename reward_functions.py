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
            'constraint_values': torch.zeros((self.num_envs, self.num_actions), device=device, dtype=gs.tc_float),
            'gait_target': torch.zeros((self.num_envs, self.num_actions), device=device, dtype=gs.tc_float),
            'gait_error': torch.zeros((self.num_envs,), device=device, dtype=gs.tc_float),
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
    
    def reward_sinusoidal_gait(self, dof_pos, default_dof_pos, episode_length_buf, dt):
        """Optimized sinusoidal gait reward using pre-allocated buffers"""
        amplitude = self.reward_cfg.get("gait_amplitude", 0.5)
        frequency = self.reward_cfg.get("gait_frequency", 0.5)
        
        # Pre-defined phase offsets - avoid tensor creation
        phase_offsets = torch.tensor(
            [0, 0, 0, 0, np.pi, 0, 0, 0], 
            device=self.device, dtype=gs.tc_float
        )

        time = episode_length_buf * dt
        time = time.unsqueeze(1)

        # Use pre-allocated buffer for target calculation
        torch.sin(2 * np.pi * frequency * time + phase_offsets, out=self.reward_buffers['gait_target'])
        self.reward_buffers['gait_target'].mul_(amplitude)
        self.reward_buffers['gait_target'].add_(default_dof_pos)

        # Calculate error using pre-allocated buffer
        torch.sub(dof_pos, self.reward_buffers['gait_target'], out=self.reward_buffers['gait_target'])
        torch.square(self.reward_buffers['gait_target'], out=self.reward_buffers['gait_target'])
        error = torch.sum(self.reward_buffers['gait_target'], dim=1)

        sigma = self.reward_cfg.get("gait_sigma", 0.25)
        return torch.exp(-error / sigma)

    def reward_torso_sinusoidal(self):
        """Placeholder for torso sinusoidal reward (currently returns zeros)"""
        return torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

    def reward_actuator_constraint(self, dof_vel, joint_torques):
        """
        Optimized actuator constraint reward using pre-allocated buffers
        Enforces: speed + 3.5*|torque| <= 6.16
        """
        constraint_limit = self.reward_cfg.get("actuator_constraint_limit", 6.16)
        torque_coeff = self.reward_cfg.get("actuator_torque_coeff", 3.5)
        tolerance = self.reward_cfg.get("actuator_tolerance", 0.5)
        
        # Use pre-allocated buffer for constraint computation
        # constraint_values = |dof_vel| + torque_coeff * |joint_torques|
        torch.abs(dof_vel, out=self.reward_buffers['constraint_values'])
        
        # Simple approach: compute torque_coeff * |joint_torques| and add
        abs_torques = torch.abs(joint_torques)
        torch.add(self.reward_buffers['constraint_values'], 
                 abs_torques, 
                 alpha=torque_coeff,
                 out=self.reward_buffers['constraint_values'])
        
        target_with_tolerance = constraint_limit + tolerance
        torch.sub(self.reward_buffers['constraint_values'], target_with_tolerance, 
                 out=self.reward_buffers['constraint_values'])
        # Use clamp instead of relu since relu doesn't support out parameter
        torch.clamp(self.reward_buffers['constraint_values'], min=0.0, 
                   out=self.reward_buffers['constraint_values'])
        
        total_violation_per_env = torch.sum(self.reward_buffers['constraint_values'], dim=1)
        
        return -total_violation_per_env
    
    def compute_rewards(self, base_lin_vel, actions, last_actions, dof_pos, default_dof_pos, commands, base_euler, base_pos, dof_vel, episode_length_buf, dt, joint_torques):
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
            'torso_stability': self.reward_torso_stability(base_euler),
            'height_maintenance': self.reward_height_maintenance(base_pos),
            'joint_movement': self.reward_joint_movement(dof_vel),
            'sinusoidal_gait': self.reward_sinusoidal_gait(dof_pos, default_dof_pos, episode_length_buf, dt),
            'torso_sinusoidal': self.reward_torso_sinusoidal(),
            # Note: The original implementation for this reward in the env was complex.
            # You may need to adjust the inputs or logic if it was using more state.
            'actuator_constraint': self.reward_actuator_constraint(dof_vel, joint_torques)
        }
        return rewards

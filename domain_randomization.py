"""
Domain Randomization Module for Biped Environment (Optimized)

This module contains all domain randomization functionality for the biped robot environment,
including motor strength, friction, mass, observation noise, foot contact randomization,
and motor backlash effects.

OPTIMIZATION FEATURES:
- Fully vectorized randomization operations, eliminating Python loops.
- Pre-allocated randomization buffers to minimize memory allocation overhead.
- Prioritizes batch parameter updates for efficient simulator interaction.
- Efficient noise generation with in-place operations.
- Advanced tensor indexing (torch.gather) for complex, parallel computations.
"""

import torch
import genesis as gs
import numpy as np


def gs_rand_float(lower, upper, shape, device):
    """
    Generate random float tensors in a specified range.
    
    Args:
        lower (float): The lower bound of the random range.
        upper (float): The upper bound of the random range.
        shape (tuple): The shape of the output tensor.
        device (str or torch.device): The device to create the tensor on.
        
    Returns:
        torch.Tensor: A tensor of random floats.
    """
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class DomainRandomization:
    """Handles all domain randomization aspects of the biped environment"""
    
    def __init__(self, num_envs, num_actions, env_cfg, device):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.env_cfg = env_cfg
        self.device = device
        
        # Pre-allocate noise buffers for optimization
        self.noise_buffers = {
            'dof_pos': torch.zeros((self.num_envs, self.num_actions), device=device),
            'dof_vel': torch.zeros((self.num_envs, self.num_actions), device=device),
            'lin_vel': torch.zeros((self.num_envs, 3), device=device),
            'ang_vel': torch.zeros((self.num_envs, 3), device=device),
            'base_pos': torch.zeros((self.num_envs, 3), device=device),
            'base_euler': torch.zeros((self.num_envs, 3), device=device),
            'foot_contact': torch.zeros((self.num_envs, 2), device=device),
        }
        
        # Pre-allocate domain randomization buffers for vectorized operations
        self.randomization_buffers = {
            'motor_strengths': torch.ones((self.num_envs, self.num_actions), device=device, dtype=gs.tc_float),
            'friction_values': torch.ones((self.num_envs,), device=device, dtype=gs.tc_float),
            'mass_offsets': torch.zeros((self.num_envs,), device=device, dtype=gs.tc_float),
            'backlash_values': torch.zeros((self.num_envs, self.num_actions), device=device, dtype=gs.tc_float),
        }
        
        # Different update intervals for different randomizations
        self.randomization_intervals = {
            'motor_strength': 50,
            'friction': 100,
            'mass': 200,
            'observation_noise': 1,
            'foot_contacts': 1,
            'motor_backlash': 20,
        }
        self.randomization_counters = {k: 0 for k in self.randomization_intervals}
        self.randomization_step_counter = 0
        
        # Motor Backlash Buffers
        self.motor_backlash = torch.zeros((self.num_envs, self.num_actions), device=device, dtype=gs.tc_float)
        self.motor_backlash_direction = torch.ones((self.num_envs, self.num_actions), device=device, dtype=gs.tc_float)
        self.last_motor_positions = torch.zeros((self.num_envs, self.num_actions), device=device)
        
        # Foot Contact Domain Randomization Buffers
        fc_params = self.env_cfg["domain_rand"]["foot_contact_params"]
        max_delay = fc_params["contact_delay_range"][1]
        self.contact_thresholds = torch.zeros((self.num_envs, 2), device=device, dtype=gs.tc_float)
        self.contact_noise_scale = torch.zeros((self.num_envs, 2), device=device, dtype=gs.tc_float)
        self.contact_false_positive_prob = torch.zeros((self.num_envs, 2), device=device, dtype=gs.tc_float)
        self.contact_false_negative_prob = torch.zeros((self.num_envs, 2), device=device, dtype=gs.tc_float)
        self.contact_delay_steps = torch.zeros((self.num_envs, 2), device=device, dtype=torch.long)
        self.contact_delay_buffer = torch.zeros((self.num_envs, 2, max_delay + 1), device=device, dtype=gs.tc_float)
        self.contact_delay_idx = torch.zeros((self.num_envs,), device=device, dtype=torch.long)
    
    def should_update_randomization(self, randomization_type):
        """Check if a specific randomization should be updated based on its interval"""
        interval = self.randomization_intervals.get(randomization_type, 1)
        return self.randomization_step_counter % interval == 0
    
    def update_step_counter(self):
        """Update the randomization step counter"""
        self.randomization_step_counter += 1
    
    def generate_noise_batch(self):
        """Optimized batch noise generation - eliminates per-type overhead"""
        noise_scales = self.env_cfg["domain_rand"]["noise_scales"]
        
        # Generate all noise in one go for better performance
        for noise_type, buffer in self.noise_buffers.items():
            if noise_type in noise_scales:
                # Use 'out' parameter to perform operation in-place, avoiding new allocation
                torch.randn(buffer.shape, out=buffer, device=self.device)
                buffer.mul_(noise_scales[noise_type])  # In-place scaling
    
    def randomize_motor_strength(self, robot, envs_idx, orig_kp, randomized_kp, motors_dof_idx):
        """Apply motor strength randomization with vectorized operations"""
        if not self.should_update_randomization('motor_strength'):
            return
            
        dr_cfg = self.env_cfg["domain_rand"]
        
        # Vectorized motor strength randomization - eliminates loop overhead
        self.randomization_buffers['motor_strengths'][envs_idx].uniform_(
            dr_cfg["motor_strength_range"][0],
            dr_cfg["motor_strength_range"][1]
        )
        randomized_kp[envs_idx] = orig_kp * self.randomization_buffers['motor_strengths'][envs_idx]
        
        # Batch update KP values - more efficient than individual robot.set_dofs_kp calls
        if hasattr(robot, 'set_dofs_kp_batch'):
            robot.set_dofs_kp_batch(
                randomized_kp[envs_idx],
                motors_dof_idx, 
                envs_idx=envs_idx
            )
        else:
            # Fallback to individual updates if batch method unavailable
            for i, env_idx in enumerate(envs_idx):
                robot.set_dofs_kp(
                    randomized_kp[env_idx],
                    motors_dof_idx, 
                    envs_idx=[env_idx]
                )
    
    def randomize_friction(self, robot, envs_idx):
        """Apply friction randomization with vectorized operations"""
        if not self.should_update_randomization('friction'):
            return
            
        dr_cfg = self.env_cfg["domain_rand"]
        
        try:
            # Vectorized friction randomization
            self.randomization_buffers['friction_values'][envs_idx].uniform_(
                dr_cfg["friction_range"][0],
                dr_cfg["friction_range"][1]
            )
            
            if hasattr(robot, 'set_friction_batch'):
                robot.set_friction_batch(
                    self.randomization_buffers['friction_values'][envs_idx], 
                    envs_idx=envs_idx
                )
            else:
                # Fallback to individual updates
                for i, env_idx in enumerate(envs_idx):
                    if hasattr(robot, 'set_friction'):
                        robot.set_friction(self.randomization_buffers['friction_values'][env_idx].item())
        except (AttributeError, TypeError) as e:
            if not hasattr(self, '_friction_warning_shown'):
                print(f"Warning: Friction randomization not supported: {e}")
                self._friction_warning_shown = True
    
    def randomize_mass(self, robot, envs_idx):
        """Apply mass randomization with vectorized operations"""
        if not self.should_update_randomization('mass'):
            return
            
        dr_cfg = self.env_cfg["domain_rand"]
        
        try:
            torso_link = None
            for link in robot.links:
                if link.name in ["torso", "base_link", "torso_link", "base", "servo1"]:
                    torso_link = link
                    break
            
            if torso_link is not None:
                base_mass = 1.0 # Assuming a default base mass
                # Vectorized mass offset generation
                self.randomization_buffers['mass_offsets'][envs_idx].uniform_(
                    dr_cfg["added_mass_range"][0],
                    dr_cfg["added_mass_range"][1]
                )
                
                if hasattr(robot, 'set_link_mass_batch'):
                    new_masses = base_mass + self.randomization_buffers['mass_offsets'][envs_idx]
                    robot.set_link_mass_batch(torso_link.idx, new_masses, envs_idx=envs_idx)
                else:
                    # Fallback to individual updates
                    for i, env_idx in enumerate(envs_idx):
                        new_mass = base_mass + self.randomization_buffers['mass_offsets'][env_idx].item()
                        # Chain of fallbacks for different simulator APIs
                        try:
                            if hasattr(robot, 'set_link_mass'):
                                robot.set_link_mass(torso_link.idx, new_mass)
                            elif hasattr(torso_link, 'set_mass'):
                                torso_link.set_mass(new_mass)
                            elif hasattr(torso_link, 'mass'):
                                torso_link.mass = new_mass
                            else:
                                if not hasattr(self, '_mass_api_warning_shown'):
                                    print("Warning: Mass randomization not supported - no mass modification API found")
                                    self._mass_api_warning_shown = True
                                break
                        except Exception as e:
                            if not hasattr(self, '_mass_api_warning_shown'):
                                print(f"Warning: Mass randomization not supported: {e}")
                                self._mass_api_warning_shown = True
                            break
            else:
                if not hasattr(self, '_mass_warning_shown'):
                    print("Warning: Torso link not found for mass randomization")
                    self._mass_warning_shown = True
        except (AttributeError, TypeError) as e:
            if not hasattr(self, '_mass_api_warning_shown'):
                print(f"Warning: Mass randomization not supported: {e}")
                self._mass_api_warning_shown = True
    
    def setup_motor_backlash(self, envs_idx):
        """Setup motor backlash parameters for specified environments"""
        if not self.env_cfg["domain_rand"]["add_motor_backlash"]:
            return
            
        # Vectorized backlash generation
        backlash_range = self.env_cfg["domain_rand"]["backlash_range"]
        self.randomization_buffers['backlash_values'][envs_idx].uniform_(backlash_range[0], backlash_range[1])
        self.motor_backlash[envs_idx] = self.randomization_buffers['backlash_values'][envs_idx]
        self.motor_backlash_direction[envs_idx] = 1.0
        self.last_motor_positions[envs_idx] = 0.0
    
    def apply_motor_backlash(self, actions):
        """Apply motor backlash effects to actions"""
        if not self.env_cfg["domain_rand"]["add_motor_backlash"]:
            return actions
            
        position_diff = actions - self.last_motor_positions
        
        # Determine where motor direction has changed
        direction_change = (position_diff * self.motor_backlash_direction) < 0
        
        # Apply backlash offset only where direction has changed
        backlash_offset = self.motor_backlash * self.motor_backlash_direction
        actions_with_backlash = actions.clone()
        actions_with_backlash[direction_change] += backlash_offset[direction_change]
        
        # OPTIMIZATION: Use torch.where for conditional assignment. It's cleaner and can be faster.
        # Update direction, but keep old direction if movement is negligible to prevent flutter.
        new_direction = torch.sign(position_diff)
        self.motor_backlash_direction = torch.where(
            torch.abs(position_diff) >= 1e-6, 
            new_direction, 
            self.motor_backlash_direction
        )
        
        self.last_motor_positions = actions.clone()
        
        return actions_with_backlash
    
    def setup_foot_contact_randomization(self, envs_idx):
        """Setup foot contact randomization parameters for specified environments"""
        if not self.env_cfg["domain_rand"]["randomize_foot_contacts"]:
            return
            
        fc_params = self.env_cfg["domain_rand"]["foot_contact_params"]
        
        self.contact_thresholds[envs_idx] = gs_rand_float(
            fc_params["contact_threshold_range"][0], fc_params["contact_threshold_range"][1],
            (len(envs_idx), 2), self.device
        )
        self.contact_noise_scale[envs_idx] = gs_rand_float(
            fc_params["contact_noise_range"][0], fc_params["contact_noise_range"][1],
            (len(envs_idx), 2), self.device
        )
        self.contact_false_positive_prob[envs_idx] = fc_params["false_positive_rate"]
        self.contact_false_negative_prob[envs_idx] = fc_params["false_negative_rate"]
        self.contact_delay_steps[envs_idx] = torch.randint(
            fc_params["contact_delay_range"][0], fc_params["contact_delay_range"][1] + 1,
            (len(envs_idx), 2), device=self.device
        )
        
        self.contact_delay_buffer[envs_idx] = 0.0
        self.contact_delay_idx[envs_idx] = 0
    
    def apply_foot_contact_randomization_optimized(self, raw_contacts):
        """Apply optimized foot contact randomization"""
        if not self.env_cfg["domain_rand"]["randomize_foot_contacts"]:
            return raw_contacts.clone()
            
        # Initial detection based on randomized thresholds
        contact_detected = raw_contacts > self.contact_thresholds
        
        # Apply observation noise if enabled
        if self.env_cfg["domain_rand"]["add_observation_noise"]:
            contact_noise = self.noise_buffers['foot_contact'] # Uses pre-generated noise
            randomized_contacts = raw_contacts + contact_noise
        else:
            randomized_contacts = raw_contacts.clone()
        
        # Apply false positives and negatives using vectorized boolean masks
        false_pos_rand = torch.rand_like(randomized_contacts)
        false_neg_rand = torch.rand_like(randomized_contacts)
        
        false_pos_mask = (false_pos_rand < self.contact_false_positive_prob) & ~contact_detected
        false_neg_mask = (false_neg_rand < self.contact_false_negative_prob) & contact_detected
        
        randomized_contacts[false_pos_mask] = 1.0
        randomized_contacts[false_neg_mask] = 0.0
        
        # Apply contact delays in a fully vectorized manner
        if hasattr(self, 'contact_delay_steps'):
            randomized_contacts = self._apply_contact_delays_vectorized(randomized_contacts)
        
        return torch.clamp(randomized_contacts, 0.0, 1.0)
    
    def _apply_contact_delays_vectorized(self, contacts):
        """
        Apply contact delays using a fully vectorized implementation.
        This replaces the original for-loop with efficient tensor indexing.
        """
        buffer_size = self.contact_delay_buffer.shape[2]
        if buffer_size == 0:
            return contacts

        # The current circular buffer index for writing. Use modulo for wraparound.
        current_idx = self.contact_delay_idx[0] % buffer_size
        
        # Store the latest contact state in the buffer.
        self.contact_delay_buffer[:, :, current_idx] = contacts
        
        # --- MAJOR OPTIMIZATION: Vectorized Gathering ---
        # Calculate the indices to read from the buffer for all environments at once.
        # This is the core of the vectorization, replacing the per-environment loop.
        delay_indices = (current_idx - self.contact_delay_steps) % buffer_size
        
        # `torch.gather` selects elements from the buffer along dim 2 using our calculated indices.
        # We need to unsqueeze delay_indices to match the buffer's dimension for gathering.
        delayed_contacts = torch.gather(self.contact_delay_buffer, 2, delay_indices.unsqueeze(-1)).squeeze(-1)
        
        # Create a mask to apply delays only where they are configured (delay > 0).
        delay_active_mask = (self.contact_delay_steps > 0)
        
        # Update the contacts tensor with the delayed values only where the mask is true.
        contacts[delay_active_mask] = delayed_contacts[delay_active_mask]
        
        # Increment the index for the next step.
        self.contact_delay_idx += 1
        
        return contacts
    
    def reset_randomization_buffers(self, envs_idx):
        """Reset randomization-related buffers for specified environments"""
        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = torch.tensor(envs_idx, device=self.device, dtype=torch.long)

        self.motor_backlash[envs_idx] = 0.0
        self.motor_backlash_direction[envs_idx] = 1.0
        self.last_motor_positions[envs_idx] = 0.0
        
        if self.contact_delay_buffer.numel() > 0:
            self.contact_delay_buffer[envs_idx] = 0.0
        self.contact_delay_idx[envs_idx] = 0
def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128 ],
            "critic_hidden_dims": [512, 256, 128 ],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            
            # New configurations
            "num_steps_per_env": 24,
            "save_interval": 50,
            
            # Logging
            "logger": "wandb",  # Options: 'tensorboard', 'wandb', 'neptune'
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,  # Updated value
        "save_interval": 50,  # Updated value
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 8,  # 8 DOF for biped: 4 per leg (no torso)
        # joint/link names - based on your URDF with neutral standing pose
        "default_joint_angles": {  # [rad] - neutral standing pose with ground contact
            "right_hip1": 0.0,     # hip abduction/adduction 
            "right_hip2": -0.652,   # hip flexion/extension
            "right_knee": 1.30,    # knee flexion
            "right_ankle": -0.634,  # ankle flexion
            "left_hip1": 0.0,      # hip abduction/adduction
            "left_hip2": -0.652,    # hip flexion/extension
            "left_knee": -1.30,    # knee flexion (negative for left leg)
            "left_ankle": -0.634,   # ankle flexion
        },
        "joint_names": [
            # Right leg first (as per your configuration)
            "right_hip1",
            "right_hip2",
            "right_knee", 
            "right_ankle",
            # Left leg
            "left_hip1",
            "left_hip2", 
            "left_knee",
            "left_ankle",
        ],
        # PD control parameters - start conservative and tune
        "kp": 30.0,  # Higher than quadruped due to biped instability
        "kd": 1.0,   # Higher damping for stability
        # termination conditions - tighter for biped
        "termination_if_roll_greater_than": 55,  # degree - bipeds can lean more
        "termination_if_pitch_greater_than": 55, # degree
        
        # Actuator constraint termination
        "terminate_on_actuator_violation": True,  # Enable termination on severe violations
        "actuator_violation_termination_threshold": 2.0,  # Terminate if violation > this value
        
        # Fall penalty thresholds (in degrees)
        "fall_roll_threshold": 40.0,   # Roll threshold for fall penalty (slightly less than termination)
        "fall_pitch_threshold": 40.0,  # Pitch threshold for fall penalty (slightly less than termination)
        # base pose - height adjusted for neutral configuration ground contact
        "base_init_pos": [0.0, 0.0, 0.50],  # Lower spawn height for ground contact with neutral pose
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 90.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # Conservative scaling
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        
        # Binary contact threshold for observations
        "binary_contact_threshold": 0.1,  # Force threshold for binary contact detection in observations (N)
        
        # Foot contact threshold for gait rewards
        "foot_contact_threshold": 0.1,  # Force threshold for contact detection in gait rewards (N)
        
        # Domain Randomization Configuration
        "domain_rand": {
            "randomize_friction":False,  # Disabled until Genesis API support is confirmed
            "friction_range": [0.4, 1.25],  # Range for friction coefficient

            "randomize_mass": False,  # Disabled until torso link is properly identified
            "added_mass_range": [0.0, 0.4], # kg to add or remove from torso

            "randomize_motor_strength": False,  # This is working correctly
            "motor_strength_range": [0.6, 1.2], # Scale factor for kp

            "push_robot": False,  # Disabled - external force application removed
            "push_interval_s": 7, # Push the robot every 7 seconds (disabled)
            "max_push_vel_xy": 1.0, # m/s (disabled)
            
            # Motor Backlash Configuration
            "add_motor_backlash": False ,
            "backlash_range": [0.01, 0.07],  # Backlash angle range in radians (0.5-3 degrees)
            
            # Sensor Noise Configuration
            "add_observation_noise": False ,
            "noise_scales": {
                "dof_pos": 0.02,    # Noise stddev for joint positions (rad)
                "dof_vel": 0.2,     # Noise stddev for joint velocities (rad/s)
                "lin_vel": 0.1,     # Noise stddev for base linear velocity (m/s)
                "ang_vel": 0.15,    # Noise stddev for base angular velocity (rad/s)
                "base_pos": 0.01,   # Noise stddev for base position (meters)
                "base_euler": 0.03, # Noise stddev for base orientation (rad)
                "foot_contact": 0.1, # Noise stddev for foot contact sensors
            },
            
            # Foot Contact Domain Randomization
            "randomize_foot_contacts": False,
            "foot_contact_params": {
                "contact_threshold_range": [0.01, 0.15],  # Force threshold for contact detection (N)
                "contact_noise_range": [0.0, 0.2],       # Additional noise on contact readings
                "false_positive_rate": 0.05,             # Probability of false contact detection
                "false_negative_rate": 0.05,             # Probability of missing actual contact
                "contact_delay_range": [0, 2],           # Delay in contact detection (timesteps)
            }
        }
    }
    
    obs_cfg = {
        "num_obs": 37,  # 2+2+1+2+1+3+4+4+2+2+2+2+2+8 = 37: base(8) + commands(3) + joints(16) + contacts(2) + actions(8)
        "obs_scales": {
            "lin_vel": 2.0,      # Scaling for linear velocities in observations
            "ang_vel": 0.25,     # Scaling for angular velocities in observations
            "dof_pos": 1.0,      # Scaling for joint positions
            "dof_vel": 0.05,     # Scaling for joint velocities
            "base_euler": 1.0,   # For torso pitch/roll angles
            "base_height": 1.0,  # For torso height
        },
    }
    
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.25,  # Target height for neutral crouched pose
        "feet_height_target": 0.1,  # Ground clearance during swing
        
        # New reward parameters
        "stability_factor": 1.0,  # Torso stability smoothness factor
        "height_target": 0.25,  # Height maintenance target for neutral pose
        "movement_threshold": 2.0,  # Maximum movement reward threshold
        "movement_scale": 0.1,  # Scale factor for joint movement reward
        
        
        "tracking_sigma": 0.25,
        
        
        # Foot parallelism reward parameters
        "foot_parallelism_k": 25.0,  # Scaling factor for exponential reward (higher = more sensitive)
        
        # Foot air time reward parameters
        "feet_air_time_threshold": 0.1,  # Minimum air time threshold (seconds)
        "feet_air_time_positive_threshold": 0.5,  # Maximum time threshold for positive biped reward
        "command_threshold": 0.1,  # Minimum command magnitude to give reward
        
        # Foot slide penalty parameters
        "feet_slide_contact_threshold": 1.0,  # Force threshold for contact detection (N)
        
        "reward_scales": {
            # Velocity tracking rewards (primary objectives)
            "tracking_lin_vel_x": 20.0,     # Track commanded forward velocity
            "tracking_lin_vel_y": 6.0,      # Track commanded sideways velocity
            
            # Stability and regularization rewards
            "lin_vel_z": -2.0,              # Penalize vertical motion
            "action_rate": -0.02,           # Smooth actions
            "similar_to_default": -0.05,     # Stay near neutral pose
            "alive_bonus": 0.5,             # Alive bonus per step
            "fall_penalty": -100.0,         # Large penalty for falling
            "torso_stability": 5.0,         # Torso stability reward
            "height_maintenance": -2.0,     # Height maintenance
            "joint_movement": 1.0,          # Reward for joint movement (reduced weight)

            
            # Foot parallelism reward
            "foot_parallelism": 5.0,        # Reward for foot parallelism to ground
            
            # Gait rewards
            "feet_air_time": 2.0,           # Reward for adequate air time
            "feet_air_time_positive_biped": 1.0,  # Positive reward for proper gait timing
            "feet_slide": -2.0,             # Penalty for foot sliding
        },
        
        # Enable/disable reward functions using if True/False
        "reward_enables": {
            # Velocity tracking rewards (primary objectives)
            "tracking_lin_vel_x": True,     # Track commanded forward velocity
            "tracking_lin_vel_y": True,     # Track commanded sideways velocity
            
            # Stability and regularization rewards
            "lin_vel_z": True,              # Penalize vertical motion
            "action_rate": True,            # Smooth actions
            "similar_to_default": True,     # Stay near neutral pose

            "alive_bonus": True,            # Alive bonus per step
            "fall_penalty": True,           # Large penalty for falling
            "torso_stability": False,        # Torso stability reward
            "height_maintenance": False,     # Height maintenance
            
            "joint_movement": True,         # Reward for joint movement

            
            # Foot parallelism reward
            "foot_parallelism": True,       # Enable foot parallelism reward
            
            # Gait rewards
            "feet_air_time": True,          # Enable air time reward
            "feet_air_time_positive_biped": True,  # Enable positive gait timing reward
            "feet_slide": True,             # Enable foot slide penalty
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        # Command range for forward velocity (m/s) - progressive training
        "lin_vel_x_range": [-1.5, 1.5],    # Forward/backward velocity range
        # Command range for sideways velocity (m/s)
        "lin_vel_y_range": [0.0, 0.0],    # Left/right velocity range  
        # Command range for angular velocity (rad/s) - keep zero for now
        "ang_vel_range": [0.0, 0.0],       # No turning for now, focus on linear motion
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

import argparse
import os
import pickle
import shutil
import signal
import sys
import time
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner
from training.ppo_metrics_wrapper import PPOMetricsRunner

import genesis as gs

from training.biped_env import BipedEnv

# Import W&B integration components
try:
    from training.wandb_integration import WandbBipedEnvWrapper
    from training.wandb_logger import WandbPPOLogger
    WANDB_AVAILABLE = True
    print("✅ W&B integration available")
except ImportError as e:
    WANDB_AVAILABLE = False
    print(f"⚠️ W&B integration not available: {e}")
    print("💡 Install with: pip install wandb pytz")


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",  # Use standard PPO (we patch it)
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
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
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
        },
        "runner_class_name": "OnPolicyRunner",  # We'll create PPOMetricsRunner manually
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 9,  # 9 DOF for biped: 4 per leg + 1 torso
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
            "revolute_torso": 0.0,          # torso rotation
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
            # Torso
            "revolute_torso",
        ],
        # PD control parameters - reduced for more stable operation
        "kp": 15.0,  # Reduced from 30.0 - high gains cause excessive torques
        "kd": 0.8,   # Slightly reduced damping
        # termination conditions - tighter for biped
        "termination_if_roll_greater_than": 30,  # degree - bipeds can lean more
        "termination_if_pitch_greater_than": 30, # degree
        
        # Actuator constraint termination - more lenient to avoid immediate termination
        "terminate_on_actuator_violation": False,  # Disable termination, use reward penalty only
        "actuator_violation_termination_threshold": 5.0,  # Higher threshold if enabled
        
        # Fall penalty thresholds (in degrees)
        "fall_roll_threshold": 25.0,   # Roll threshold for fall penalty (slightly less than termination)
        "fall_pitch_threshold": 25.0,  # Pitch threshold for fall penalty (slightly less than termination)
        # base pose - height adjusted for neutral configuration ground contact
        "base_init_pos": [0.0, 0.0, 0.50],  # Lower spawn height for ground contact with neutral pose
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 90.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # Conservative scaling
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        
        # Domain Randomization Configuration
        "domain_rand": {
            "randomize_friction":True,  # Disabled until Genesis API support is confirmed
            "friction_range": [0.4, 1.25],  # Range for friction coefficient

            "randomize_mass": False,  # Disabled until torso link is properly identified
            "added_mass_range": [0.0, 0.4], # kg to add or remove from torso

            "randomize_motor_strength": True,  # This is working correctly
            "motor_strength_range": [0.6, 1.2], # Scale factor for kp

            "push_robot": False,  # Disabled - external force application removed
            "push_interval_s": 7, # Push the robot every 7 seconds (disabled)
            "max_push_vel_xy": 1.0, # m/s (disabled)
            
            # Motor Backlash Configuration
            "add_motor_backlash": True ,
            "backlash_range": [0.01, 0.07],  # Backlash angle range in radians (0.5-3 degrees)
            
            # Sensor Noise Configuration
            "add_observation_noise": True ,
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
        "num_obs": 38,  # 2+2+1+2+1+3+4+4+2+2+2+2+2+9 = 38 for new observation structure with commands
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
        "forward_velocity_target": 0.5,
        "stability_factor": 1.0,  # Torso stability smoothness factor
        "height_target": 0.25,  # Height maintenance target for neutral pose
        "movement_threshold": 2.0,  # Maximum movement reward threshold
        "movement_scale": 0.1,  # Scale factor for joint movement reward
        "gait_amplitude": 0.4,   # The desired amplitude of the joint movement in radians
        "gait_frequency": 0.6,   # The desired frequency of the gait in Hz
        "gait_sigma": 0.25,      # The tolerance for the reward. Smaller values are stricter.
        
        # Torso sinusoidal motion parameters
        "torso_amplitude": 0.2,  # Smaller amplitude for torso sinusoidal motion (rad)
        "torso_frequency": 0.3,  # Different frequency from leg gait (Hz)
        "torso_phase": 1.732,      # Phase offset for torso motion
        "torso_sigma": 0.25,     # Tolerance for torso sinusoidal reward
        
        "tracking_sigma": 0.25,
        
        # Actuator constraint parameters - more lenient settings
        "actuator_constraint_limit": 8.0,   # Increased from 6.16 to be more lenient
        "actuator_torque_coeff": 3.5,       # Coefficient for torque in constraint
        "actuator_tolerance": 1.0,           # Increased tolerance before penalty starts
        "actuator_termination_threshold": 5.0,  # Higher violation level for termination
        
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
            "torso_stability": True,        # Torso stability reward
            "height_maintenance": True,     # Height maintenance
            
            # Gait and movement rewards (reduced to prioritize command following)
            "sinusoidal_gait": True,        # Leg sinusoidal gait
            "torso_sinusoidal": True,       # Torso sinusoidal motion reward
            "joint_movement": True,         # Reward for joint movement
            
            # Actuator constraint reward
            "actuator_constraint": False,    # Penalty for actuator constraint violations
        },
        
        "reward_scales": {
            # Velocity tracking rewards (primary objectives)
            "tracking_lin_vel_x": 10.0,     # Track commanded forward velocity
            "tracking_lin_vel_y": 6.0,      # Track commanded sideways velocity
            
            # Stability and regularization rewards
            "lin_vel_z": -2.0,              # Penalize vertical motion
            "action_rate": -0.02,           # Smooth actions
            "similar_to_default": -0.1,     # Stay near neutral pose
            "alive_bonus": 0.5,             # Alive bonus per step
            "fall_penalty": -100.0,         # Large penalty for falling
            "torso_stability": 5.0,         # Torso stability reward
            "height_maintenance": -2.0,     # Height maintenance
            
            # Gait and movement rewards (reduced to prioritize command following)
            "sinusoidal_gait": 2.0,         # Leg sinusoidal gait (reduced weight)
            "torso_sinusoidal": 5.0,        # Torso sinusoidal motion reward (reduced weight)
            "joint_movement": 1.0,          # Reward for joint movement (reduced weight)
            
            # Actuator constraint reward
            "actuator_constraint": -2.0,    # Reduced penalty for more exploration
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        # Command range for forward velocity (m/s) - progressive training
        "lin_vel_x_range": [-0.5, 1.0],    # Forward/backward velocity range
        # Command range for sideways velocity (m/s)
        "lin_vel_y_range": [-0.3, 0.3],    # Left/right velocity range  
        # Command range for angular velocity (rad/s) - keep zero for now
        "ang_vel_range": [0.0, 0.0],       # No turning for now, focus on linear motion
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=1500)
    parser.add_argument("--max_iterations", type=int, default=999999)  # Very large number, will run until Ctrl+C
    
    # W&B integration arguments
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="biped-ppo-genesis", help="W&B project name")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[], help="W&B tags")
    parser.add_argument("--wandb_notes", type=str, default="", help="W&B run notes")
    parser.add_argument("--wandb_offline", action="store_true", help="Run W&B in offline mode")
    
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Create environment with optional W&B integration
    if args.wandb and WANDB_AVAILABLE:
        print("🚀 Initializing biped training with W&B logging...")
        
        # Set offline mode if requested
        if args.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'
            print("📴 W&B running in offline mode")
        
        # Create W&B-enhanced environment
        env = WandbBipedEnvWrapper(
            num_envs=args.num_envs, 
            env_cfg=env_cfg, 
            obs_cfg=obs_cfg, 
            reward_cfg=reward_cfg, 
            command_cfg=command_cfg,
            log_dir=log_dir
        )
        
        # Configure W&B logging
        wandb_config = {
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "train_cfg": train_cfg,
            "num_envs": args.num_envs,
            "max_iterations": args.max_iterations,
            "experiment_name": args.exp_name
        }
        
        # Create experiment tags
        tags = ["biped-training", "ppo", "genesis-simulator"] + args.wandb_tags
        if args.exp_name:
            tags.append(args.exp_name)
        
        # Setup W&B logging
        env.setup_wandb_logging(
            project_name=args.wandb_project,
            experiment_name=f"{args.exp_name}-{int(time.time())}",
            config=wandb_config,
            tags=tags,
            notes=args.wandb_notes or f"Biped training with {args.num_envs} environments"
        )
        
        print(f"📊 W&B Dashboard: Check your project '{args.wandb_project}' for real-time metrics")
        
    else:
        if args.wandb and not WANDB_AVAILABLE:
            print("⚠️ W&B requested but not available. Running without W&B logging.")
        
        # Create standard environment
        env = BipedEnv(
            num_envs=args.num_envs, 
            env_cfg=env_cfg, 
            obs_cfg=obs_cfg, 
            reward_cfg=reward_cfg, 
            command_cfg=command_cfg,
            log_dir=log_dir
        )

    # Use our custom metrics runner
    runner = PPOMetricsRunner(env, train_cfg, log_dir, device=gs.device)
    
    # Enhanced logging for W&B integration
    if args.wandb and WANDB_AVAILABLE and hasattr(env, 'wandb_logger'):
        # Hook into the runner's training loop for W&B logging
        original_log = runner.log
        
        def enhanced_log(locs, width=80, pad=35):
            """Enhanced logging function that also logs to W&B."""
            # Call original logging
            original_log(locs, width, pad)
            
            # Extract metrics for W&B logging
            policy_loss = locs.get('policy_loss', 0.0)
            value_loss = locs.get('value_loss', 0.0)
            entropy = locs.get('entropy', 0.0)
            kl_divergence = locs.get('kl', 0.01)
            learning_rate = locs.get('learning_rate', None)
            total_timesteps = locs.get('total_timesteps', runner.tot_timesteps)
            
            # Log to W&B
            env.log_training_iteration(
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                kl_divergence=kl_divergence,
                learning_rate=learning_rate,
                total_timesteps=total_timesteps
            )
        
        # Replace the runner's log method
        runner.log = enhanced_log
        
        # Mark learning phase start for each iteration
        original_learn = runner.learn
        
        def learn_with_timing(*args, **kwargs):
            env.start_learning_phase()
            return original_learn(*args, **kwargs)
            
        runner.learn = learn_with_timing

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Saving current model...')
        
        # Finish W&B logging if active
        if args.wandb and WANDB_AVAILABLE and hasattr(env, 'finish_wandb_logging'):
            print('🏁 Finishing W&B logging session...')
            try:
                env.finish_wandb_logging()
            except Exception as e:
                print(f'⚠️ W&B cleanup warning: {e}')
        
        # The runner automatically saves periodically, so we just exit gracefully
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Starting training... Press Ctrl+C to stop and save the model.")
    print(f"Logs will be saved to: {log_dir}")
    if args.wandb and WANDB_AVAILABLE:
        print(f"📊 W&B metrics will be logged to project: {args.wandb_project}")
        print(f"🏷️  Tags: {', '.join(tags)}")
    
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        
        # Log final model checkpoint if W&B is active
        if args.wandb and WANDB_AVAILABLE and hasattr(env, 'log_model_checkpoint'):
            model_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
            if os.path.exists(model_path):
                print("💾 Logging final model checkpoint to W&B...")
                env.log_model_checkpoint(model_path)
                
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Final model save completed.')
    except Exception as e:
        print(f'\nTraining stopped due to error: {e}')
        raise
    finally:
        # Always clean up W&B logging
        if args.wandb and WANDB_AVAILABLE and hasattr(env, 'finish_wandb_logging'):
            print('🏁 Finishing W&B logging session...')
            try:
                env.finish_wandb_logging()
            except Exception as e:
                print(f'⚠️ W&B cleanup warning: {e}')


if __name__ == "__main__":
    main()

"""
Usage Examples:

# Standard training without W&B (existing behavior)
python biped_train.py -e biped-walking -B 2048

# Training with W&B logging enabled
python biped_train.py -e biped-walking -B 2048 --wandb

# Advanced W&B training with custom project and tags
python biped_train.py -e biped-walking -B 2048 --wandb \
    --wandb_project "my-biped-research" \
    --wandb_tags stable-gait walking-v2 \
    --wandb_notes "Testing improved reward structure"

# Offline W&B logging (for systems without internet)
python biped_train.py -e biped-walking -B 2048 --wandb --wandb_offline

# Training with specific max iterations and W&B
python biped_train.py -e biped-walking -B 2048 --max_iterations 200 --wandb
"""

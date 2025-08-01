import argparse
import os
import pickle
import shutil
import signal
import sys
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

import genesis as gs

from biped_env import BipedEnv


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
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 9,  # 9 DOF for biped: 4 per leg + 1 torso
        # joint/link names - based on your URDF with realistic standing pose
        "default_joint_angles": {  # [rad] - neutral standing pose
            "left_hip1": 0.0,      # hip abduction/adduction (range: ±1.732 rad = ±99°)
            "left_hip2": -0.1,     # hip flexion/extension (range: ±1.732 rad = ±99°) - slight forward lean
            "left_knee": 0.2,      # knee flexion (range: ±2.0 rad = ±114°) - slight bend
            "left_ankle": -0.1,    # ankle flexion (range: ±1.732 rad = ±99°) - slight plantarflexion
            "right_hip1": 0.0,     # hip abduction/adduction (range: ±1.732 rad = ±99°)  
            "right_hip2": -0.1,    # hip flexion/extension (range: ±1.732 rad = ±99°) - slight forward lean
            "right_knee": 0.2,     # knee flexion (range: ±2.0 rad = ±114°) - slight bend
            "right_ankle": -0.1,   # ankle flexion (range: ±1.732 rad = ±99°) - slight plantarflexion
            "torso": 0.0,          # torso rotation (range: ±1.732 rad = ±99°)
        },
        "joint_names": [
            # Left leg
            "left_hip1",
            "left_hip2", 
            "left_knee",
            "left_ankle",
            # Right leg
            "right_hip1",
            "right_hip2",
            "right_knee", 
            "right_ankle",
            # Torso
            "torso",
        ],
        # PD control parameters - start conservative and tune
        "kp": 30.0,  # Higher than quadruped due to biped instability
        "kd": 1.0,   # Higher damping for stability
        # termination conditions - tighter for biped
        "termination_if_roll_greater_than": 30,  # degree - bipeds can lean more
        "termination_if_pitch_greater_than": 30, # degree
        
        # Fall penalty thresholds (in degrees)
        "fall_roll_threshold": 25.0,   # Roll threshold for fall penalty (slightly less than termination)
        "fall_pitch_threshold": 25.0,  # Pitch threshold for fall penalty (slightly less than termination)
        # base pose - standing height for biped
        "base_init_pos": [0.0, 0.0, 0.5],  # Spawn height for biped
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,  # Conservative scaling
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    
    obs_cfg = {
        "num_obs": 35,  # 2+2+1+2+1+4+4+2+2+2+2+2+9 = 35 for new observation structure
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "base_euler": 1.0,  # For torso pitch/roll angles
            "base_height": 1.0,  # For torso height
        },
    }
    
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.35,  # Target standing height
        "feet_height_target": 0.1,  # Ground clearance during swing
        
        # New reward parameters
        "forward_velocity_target": 0.5,  # Target forward velocity (m/s)
        "velocity_sigma": 0.25,  # Velocity tracking smoothness
        "stability_factor": 1.0,  # Torso stability smoothness factor
        "height_target": 0.35,  # Height maintenance target
        
        "reward_scales": {
            # Existing rewards (keeping non-duplicates)
            "tracking_ang_vel": 0.5,    # Turning tracking
            "lin_vel_z": -2.0,          # Penalize vertical motion
            "action_rate": -0.01,       # Smooth actions
            "similar_to_default": -0.1, # Stay near neutral pose
            
            # New optimized rewards (replacing duplicates with better versions)
            "forward_velocity": 2.0,   # Forward velocity reward (replaces tracking_lin_vel)
            "alive_bonus": 1.0,         # Alive bonus per step
            "fall_penalty": -100.0,     # Large penalty for falling
            "torso_stability": 10.0,    # Torso stability reward (replaces uprightness)
            "height_maintenance": -1.0, # Height maintenance (replaces base_height)
        },
    }
    
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 0.3],   # Start with slow forward walking
        "lin_vel_y_range": [0, 0],       # No sideways initially
        "ang_vel_range": [0, 0],         # No turning initially
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=999999)  # Very large number, will run until Ctrl+C
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

    env = BipedEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Saving current model...')
        # The runner automatically saves periodically, so we just exit gracefully
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Starting training... Press Ctrl+C to stop and save the model.")
    print(f"Logs will be saved to: {log_dir}")
    
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Final model save completed.')
    except Exception as e:
        print(f'\nTraining stopped due to error: {e}')
        raise


if __name__ == "__main__":
    main()

"""
# training - runs until Ctrl+C (keyboard interrupt)
python biped_train.py -e biped-walking -B 2048

# training with specific max iterations
python biped_train.py -e biped-walking -B 2048 --max_iterations 200
"""

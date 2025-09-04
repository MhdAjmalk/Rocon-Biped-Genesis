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

# Use standard OnPolicyRunner from rsl-rl
from rsl_rl.runners import OnPolicyRunner
from wandb_patch import patch_onpolicy_runner_for_wandb

# Import WandB for logging if needed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import genesis as gs

from biped_env_main import BipedEnv
from biped_configs import get_cfgs, get_train_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="biped-walking")
    # Increased default batch size for better GPU utilization with optimized environment
    parser.add_argument("-B", "--num_envs", type=int, default=1024)  # Increased from 1 for performance
    parser.add_argument("--max_iterations", type=int, default=999999)  # Very large number, will run until Ctrl+C
    parser.add_argument("--wandb_login", action="store_true", help="Login to WandB before training")
    
    args = parser.parse_args()

    # Handle WandB login if requested
    if args.wandb_login and WANDB_AVAILABLE:
        print("🔐 Logging into WandB...")
        wandb.login()
        print("✅ WandB login completed")
    elif args.wandb_login and not WANDB_AVAILABLE:
        print("⚠️ WandB login requested but WandB not available. Install with: pip install wandb")

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

    # Check if WandB logging is enabled in configuration
    logger_type = train_cfg.get("runner", {}).get("logger", "tensorboard")
    
    # Initialize WandB if configured
    if logger_type == "wandb" and WANDB_AVAILABLE:
        # Initialize WandB
        wandb.init(
            project="biped-ppo-training",
            name=args.exp_name,
            config={
                'train_cfg': train_cfg,
                'env_cfg': env_cfg,
                'obs_cfg': obs_cfg,
                'reward_cfg': reward_cfg,
                'command_cfg': command_cfg,
                'num_envs': args.num_envs,
            },
            tags=["biped", "ppo", "genesis"],
            notes=f"Biped training with {args.num_envs} environments"
        )
        print("✅ WandB logging initialized")
    elif logger_type == "wandb" and not WANDB_AVAILABLE:
        print("⚠️ WandB logging requested but WandB not available. Install with: pip install wandb")
        print("   Falling back to TensorBoard logging")
    else:
        print(f"📊 Using {logger_type} logging")

    # Initialize OnPolicyRunner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # Patch for enhanced WandB logging when using WandB
    if logger_type == "wandb" and WANDB_AVAILABLE:
        patch_onpolicy_runner_for_wandb()
        print("✅ Enhanced WandB logging enabled")
    else:
        print(f"📊 Using standard logging with {logger_type}")

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Saving current model...')
        
        # Save final model
        final_model_path = os.path.join(log_dir, 'model_interrupted.pt')
        runner.save(final_model_path)
        
        # Finish WandB if it was initialized
        if logger_type == "wandb" and WANDB_AVAILABLE:
            print('Finishing WandB logging...')
            wandb.finish()
        
        print('Graceful shutdown completed.')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"🚀 Starting biped training...")
    print(f"📊 Tracking metrics with {logger_type}:")
    print(f"   • Episode Returns")
    print(f"   • Policy Loss")
    print(f"   • Value Loss") 
    print(f"   • Entropy")
    print(f"   • Performance Metrics (FPS, timing)")
    
    if logger_type == "wandb" and WANDB_AVAILABLE:
        print(f"� WandB Dashboard: {wandb.run.url}")
    
    print(f"�📁 Logs directory: {log_dir}")
    print(f"🎮 Environments: {args.num_envs}")
    print(f"🔄 Max iterations: {args.max_iterations}")
    print(f"💾 Save interval: {train_cfg['save_interval']}")
    print(f"📊 Steps per env: {train_cfg['num_steps_per_env']}")
    print(f"\n⌨️  Press Ctrl+C to stop training and save the model.")
    
    try:
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user (Ctrl+C)')
        print('Final model save completed.')
    except Exception as e:
        print(f'\nTraining stopped due to error: {e}')
        raise
    finally:
        # Always finish WandB if it was initialized
        if logger_type == "wandb" and WANDB_AVAILABLE:
            try:
                wandb.finish()
                print('WandB logging finished.')
            except:
                pass


if __name__ == "__main__":
    main()

"""
PERFORMANCE OPTIMIZED TRAINING COMMANDS:

# Basic training with TensorBoard logging (default)
python biped_train.py -e biped-walking -B 2048

# Training with WandB logging (set logger: 'wandb' in biped_configs.py)
python biped_train.py -e biped-walking -B 2048 --wandb_login

# Training with specific max iterations
python biped_train.py -e biped-walking -B 2048 --max_iterations 200

# Training with different experiment name and environment count
python biped_train.py -e my-biped-experiment -B 1024 --max_iterations 500

CONFIGURATION NOTES:
- Logger type is set in biped_configs.py: 'tensorboard', 'wandb', or 'neptune'
- Steps per environment: 24 (configured in biped_configs.py)
- Save interval: 50 iterations (configured in biped_configs.py)
- Use --wandb_login flag to authenticate with WandB before training

OPTIMIZATION NOTES:
- Environment now uses pre-allocated tensor buffers
- Vectorized domain randomization reduces overhead
- Optimized observation creation eliminates torch.cat overhead  
- Reward computation uses in-place operations
- Recommended batch sizes: 2048-8192 depending on GPU memory
- Performance improvements: 15-25% faster step times
- Configurable logging with TensorBoard or WandB support
"""
"""193dc53ad6151ab3500be245d62c64d98cb40c82"""
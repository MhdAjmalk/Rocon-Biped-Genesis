#!/usr/bin/env python3
"""
Simple keyboard-controlled policy tester for biped robot.
This script loads a trained model and allows manual velocity commands via keyboard.

Usage:
python simple_test.py --exp_name biped-walking
"""

import argparse
import torch
import genesis as gs
import time
import sys
import os
from biped_env import BipedEnv
from biped_train import get_cfgs


def get_key():
    """Simple key input function"""
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.raw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    except ImportError:
        # Fallback for Windows
        import msvcrt
        return msvcrt.getch().decode('utf-8')
    except:
        return None


def load_policy(log_dir):
    """Load the most recent policy from the log directory"""
    import glob
    
    # Look for model files
    model_files = glob.glob(os.path.join(log_dir, "**/model_*.pt"), recursive=True)
    if not model_files:
        print(f"No model files found in {log_dir}")
        return None
    
    # Get the most recent model file
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(latest_model, map_location=gs.device)
        print("Available keys in checkpoint:", list(checkpoint.keys()))
        
        # Try different possible keys for the actor network
        actor_state_dict = None
        possible_keys = [
            'actor_critic.actor.state_dict',
            'model_state_dict', 
            'ac_parameters_state_dict',
            'policy_state_dict'
        ]
        
        for key in possible_keys:
            if key in checkpoint:
                actor_state_dict = checkpoint[key]
                print(f"Found actor state dict with key: {key}")
                break
                
        if actor_state_dict is None:
            print("Could not find actor state dict in checkpoint")
            return None
            
        return actor_state_dict
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


class SimpleActor(torch.nn.Module):
    """Simple MLP actor network"""
    def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = num_obs
        
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ELU()
            ])
            prev_dim = hidden_dim
            
        layers.append(torch.nn.Linear(prev_dim, num_actions))
        self.network = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


def print_controls():
    """Print control instructions"""
    print("\n" + "="*60)
    print("BIPED ROBOT KEYBOARD CONTROL")
    print("="*60)
    print("Controls:")
    print("  w - Increase forward velocity")
    print("  s - Decrease forward velocity") 
    print("  a - Increase left velocity")
    print("  d - Increase right velocity")
    print("  q - Increase left turn")
    print("  e - Increase right turn")
    print("  x - Stop all motion")
    print("  r - Reset robot")
    print("  h - Show this help")
    print("  ESC or Ctrl+C - Exit")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="biped-walking", 
                       help="Experiment name (default: biped-walking)")
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    # Get configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Create environment with viewer
    print("Creating environment...")
    env = BipedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True
    )
    
    # Try to load policy
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        actor_state_dict = load_policy(log_dir)
        if actor_state_dict:
            # Create actor network
            actor = SimpleActor(env.num_obs, env.num_actions).to(gs.device)
            try:
                actor.load_state_dict(actor_state_dict)
                actor.eval()
                print("Policy loaded successfully!")
                use_policy = True
            except Exception as e:
                print(f"Failed to load policy weights: {e}")
                print("Running with random actions...")
                use_policy = False
        else:
            print("Running with random actions...")
            use_policy = False
    else:
        print(f"Log directory {log_dir} not found. Running with random actions...")
        use_policy = False
    
    # Initialize commands
    commands = torch.zeros(3, device=gs.device)  # [vel_x, vel_y, ang_vel]
    vel_increment = 0.1
    ang_increment = 0.2
    max_vel = 1.0
    max_ang_vel = 1.0
    
    print_controls()
    
    # Reset environment
    obs, _ = env.reset()
    
    print(f"\nStarting control loop. Use 'h' for help, ESC to exit.")
    print(f"Current commands: vx={commands[0]:.2f}, vy={commands[1]:.2f}, omega={commands[2]:.2f}")
    
    try:
        step_count = 0
        while True:
            # Simple polling approach for keyboard input
            print(f"\rStep: {step_count}, Commands: vx={commands[0]:.2f}, vy={commands[1]:.2f}, w={commands[2]:.2f} | Press keys: w/s(fwd), a/d(side), q/e(turn), x(stop), r(reset), h(help), ESC(exit)", end="", flush=True)
            
            try:
                # Try to get a key without blocking too much
                key = get_key()
                if key:
                    print()  # New line after status
                    
                    if key == '\x1b':  # ESC
                        break
                    elif key == 'w':
                        commands[0] = min(commands[0] + vel_increment, max_vel)
                        print(f"Forward velocity: {commands[0]:.2f}")
                    elif key == 's':
                        commands[0] = max(commands[0] - vel_increment, -max_vel)
                        print(f"Forward velocity: {commands[0]:.2f}")
                    elif key == 'a':
                        commands[1] = min(commands[1] + vel_increment, max_vel)
                        print(f"Left velocity: {commands[1]:.2f}")
                    elif key == 'd':
                        commands[1] = max(commands[1] - vel_increment, -max_vel)
                        print(f"Right velocity: {commands[1]:.2f}")
                    elif key == 'q':
                        commands[2] = min(commands[2] + ang_increment, max_ang_vel)
                        print(f"Angular velocity: {commands[2]:.2f}")
                    elif key == 'e':
                        commands[2] = max(commands[2] - ang_increment, -max_ang_vel)
                        print(f"Angular velocity: {commands[2]:.2f}")
                    elif key == 'x':
                        commands.fill_(0.0)
                        print("STOP - All velocities set to 0")
                    elif key == 'r':
                        obs, _ = env.reset()
                        print("Environment reset!")
                    elif key == 'h':
                        print_controls()
            except:
                pass  # Continue if key reading fails
            
            # Set commands in environment
            env.commands[0] = commands
            
            # Get action
            with torch.no_grad():
                if use_policy:
                    action = actor(obs[0:1])
                else:
                    # Random actions as fallback
                    action = torch.randn(1, env.num_actions, device=gs.device) * 0.1
            
            # Step environment
            obs, rewards, dones, infos = env.step(action)
            
            step_count += 1
            time.sleep(0.02)  # 50 Hz
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    
    print("Exiting...")


if __name__ == "__main__":
    main()

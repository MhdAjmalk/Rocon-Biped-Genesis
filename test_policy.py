#!/usr/bin/env python3
"""
Interactive policy testing script for the biped robot.
Use keyboard to send velocity commands and watch the robot respond.

Controls:
- W/S: Forward/Backward velocity
- A/D: Left/Right velocity  
- Q/E: Turn left/right
- SPACE: Stop all motion
- R: Reset robot
- ESC: Exit

Usage:
python test_policy.py --model_path logs/biped-walking/model_10000.pt
"""

import argparse
import torch
import genesis as gs
import numpy as np
import threading
import time
import sys
import select
import tty
import termios
from biped_env import BipedEnv
from biped_train import get_cfgs


class KeyboardController:
    def __init__(self):
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.ang_vel = 0.0
        self.reset_flag = False
        self.exit_flag = False
        
        # Velocity increments
        self.vel_increment = 0.1
        self.ang_increment = 0.2
        self.max_vel = 1.0
        self.max_ang_vel = 1.0
        
        # Store terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        
    def setup_keyboard(self):
        """Setup non-blocking keyboard input"""
        tty.setraw(sys.stdin.fileno())
        
    def restore_keyboard(self):
        """Restore terminal settings"""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        
    def get_key(self):
        """Get a single keypress without blocking"""
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None
        
    def process_input(self):
        """Process keyboard input and update velocity commands"""
        key = self.get_key()
        if key is None:
            return
            
        # Convert key to lowercase for consistency
        key = key.lower()
        
        if key == 'w':  # Forward
            self.vel_x = min(self.vel_x + self.vel_increment, self.max_vel)
            print(f"Forward velocity: {self.vel_x:.2f}")
        elif key == 's':  # Backward
            self.vel_x = max(self.vel_x - self.vel_increment, -self.max_vel)
            print(f"Forward velocity: {self.vel_x:.2f}")
        elif key == 'a':  # Left
            self.vel_y = min(self.vel_y + self.vel_increment, self.max_vel)
            print(f"Sideways velocity: {self.vel_y:.2f}")
        elif key == 'd':  # Right
            self.vel_y = max(self.vel_y - self.vel_increment, -self.max_vel)
            print(f"Sideways velocity: {self.vel_y:.2f}")
        elif key == 'q':  # Turn left
            self.ang_vel = min(self.ang_vel + self.ang_increment, self.max_ang_vel)
            print(f"Angular velocity: {self.ang_vel:.2f}")
        elif key == 'e':  # Turn right
            self.ang_vel = max(self.ang_vel - self.ang_increment, -self.max_ang_vel)
            print(f"Angular velocity: {self.ang_vel:.2f}")
        elif key == ' ':  # Stop
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.ang_vel = 0.0
            print("STOP - All velocities set to 0")
        elif key == 'r':  # Reset
            self.reset_flag = True
            print("RESET requested")
        elif key == '\x1b':  # ESC key
            self.exit_flag = True
            print("Exit requested")
            
    def get_commands(self):
        """Get current velocity commands as tensor"""
        return torch.tensor([self.vel_x, self.vel_y, self.ang_vel], dtype=torch.float32)


def load_policy(model_path, env):
    """Load the trained policy from checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location=gs.device)
        
        # Extract the actor network
        if 'model_state_dict' in checkpoint:
            actor_state_dict = checkpoint['model_state_dict']
        elif 'ac_parameters_state_dict' in checkpoint:
            actor_state_dict = checkpoint['ac_parameters_state_dict']
        else:
            # Assume the checkpoint is the state dict itself
            actor_state_dict = checkpoint
            
        # Create a simple MLP actor network (matching the training setup)
        from torch import nn
        
        class Actor(nn.Module):
            def __init__(self, num_obs, num_actions, hidden_dims=[512, 256, 128]):
                super().__init__()
                layers = []
                prev_dim = num_obs
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ELU()
                    ])
                    prev_dim = hidden_dim
                    
                layers.append(nn.Linear(prev_dim, num_actions))
                self.network = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.network(x)
        
        actor = Actor(env.num_obs, env.num_actions).to(gs.device)
        
        # Try to load the state dict
        try:
            actor.load_state_dict(actor_state_dict)
        except RuntimeError as e:
            print(f"Failed to load exact state dict: {e}")
            # Try to load only matching keys
            actor_dict = actor.state_dict()
            filtered_dict = {k: v for k, v in actor_state_dict.items() if k in actor_dict and v.shape == actor_dict[k].shape}
            actor_dict.update(filtered_dict)
            actor.load_state_dict(actor_dict)
            print(f"Loaded {len(filtered_dict)} layers from checkpoint")
            
        actor.eval()
        return actor
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Available keys in checkpoint:", list(checkpoint.keys()) if 'checkpoint' in locals() else "Could not load checkpoint")
        return None


def print_instructions():
    """Print control instructions"""
    print("\n" + "="*50)
    print("BIPED ROBOT POLICY TESTING")
    print("="*50)
    print("Controls:")
    print("  W/S  - Forward/Backward velocity")
    print("  A/D  - Left/Right velocity")
    print("  Q/E  - Turn left/right")
    print("  SPACE - Stop all motion")
    print("  R    - Reset robot")
    print("  ESC  - Exit")
    print("\nCurrent commands will be displayed as you type.")
    print("Robot will respond to your velocity commands in real-time.")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test learned biped policy with keyboard control")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained model (e.g., logs/biped-walking/model_10000.pt)")
    parser.add_argument("--num_envs", type=int, default=1, 
                       help="Number of environments (default: 1)")
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    # Get environment configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Create environment with viewer
    env = BipedEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True
    )
    
    # Load the trained policy
    print(f"Loading policy from: {args.model_path}")
    actor = load_policy(args.model_path, env)
    if actor is None:
        print("Failed to load policy. Exiting.")
        return
    
    print("Policy loaded successfully!")
    
    # Initialize keyboard controller
    keyboard = KeyboardController()
    keyboard.setup_keyboard()
    
    try:
        print_instructions()
        
        # Reset environment
        obs, _ = env.reset()
        
        # Main control loop
        step_count = 0
        with torch.no_grad():
            while not keyboard.exit_flag:
                # Process keyboard input
                keyboard.process_input()
                
                # Handle reset
                if keyboard.reset_flag:
                    obs, _ = env.reset()
                    keyboard.reset_flag = False
                    print("Environment reset!")
                    continue
                
                # Set manual commands
                commands = keyboard.get_commands()
                env.commands[0] = commands.to(gs.device)  # Set for first environment
                
                # Get action from policy
                obs_tensor = obs[0:1]  # Take first environment
                action = actor(obs_tensor)
                
                # Step environment
                obs, rewards, dones, infos = env.step(action)
                
                # Print status every 50 steps
                if step_count % 50 == 0:
                    pos = env.base_pos[0].cpu().numpy()
                    vel = env.base_lin_vel[0].cpu().numpy()
                    print(f"Step: {step_count}, Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                          f"Vel: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}], Reward: {rewards[0]:.3f}")
                
                step_count += 1
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.02)  # 50 Hz
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Restore terminal settings
        keyboard.restore_keyboard()
        print("Terminal restored. Goodbye!")


if __name__ == "__main__":
    main()

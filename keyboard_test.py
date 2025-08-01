#!/usr/bin/env python3
"""
Keyboard-controlled biped robot policy tester.
Uses threading for keyboard input to avoid blocking issues.

Usage:
python keyboard_test.py --exp_name biped-walking

Controls:
w/s - forward/backward, a/d - left/right, q/e - turn left/right
x - stop, r - reset, h - help, esc - exit
"""

import argparse
import torch
import genesis as gs
import time
import sys
import os
import threading
import queue
from biped_env import BipedEnv
from biped_train import get_cfgs


class KeyboardInput:
    def __init__(self):
        self.key_queue = queue.Queue()
        self.running = True
        
    def start(self):
        """Start keyboard input thread"""
        self.thread = threading.Thread(target=self._input_thread, daemon=True)
        self.thread.start()
        
    def _input_thread(self):
        """Thread function for keyboard input"""
        try:
            while self.running:
                key = input()  # Simple input() call
                if key:
                    self.key_queue.put(key.lower())
        except:
            pass
            
    def get_key(self):
        """Get key from queue if available"""
        try:
            return self.key_queue.get_nowait()
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop the input thread"""
        self.running = False


def load_latest_model(log_dir):
    """Load the most recent model checkpoint"""
    import glob
    
    model_files = glob.glob(os.path.join(log_dir, "**/model_*.pt"), recursive=True)
    if not model_files:
        return None
        
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")
    
    try:
        checkpoint = torch.load(latest_model, map_location=gs.device)
        
        # Try to find actor state dict
        for key in ['actor_critic.actor.state_dict', 'model_state_dict', 'ac_parameters_state_dict']:
            if key in checkpoint:
                return checkpoint[key]
                
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


class SimpleActor(torch.nn.Module):
    """Simple actor network"""
    def __init__(self, num_obs, num_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_actions)
        )
        
    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="biped-walking")
    args = parser.parse_args()
    
    # Initialize
    gs.init(logging_level="warning")
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Create environment
    print("Creating environment with viewer...")
    env = BipedEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True)
    
    # Try to load trained policy
    log_dir = f"logs/{args.exp_name}"
    actor_weights = load_latest_model(log_dir) if os.path.exists(log_dir) else None
    
    if actor_weights:
        actor = SimpleActor(env.num_obs, env.num_actions).to(gs.device)
        try:
            actor.load_state_dict(actor_weights)
            actor.eval()
            print("✓ Policy loaded successfully!")
            use_policy = True
        except Exception as e:
            print(f"✗ Failed to load policy: {e}")
            print("Using random actions...")
            use_policy = False
    else:
        print("No trained model found. Using random actions...")
        use_policy = False
    
    # Initialize keyboard input
    keyboard = KeyboardInput()
    keyboard.start()
    
    # Control variables
    commands = torch.tensor([0.0, 0.0, 0.0], device=gs.device)  # [vx, vy, omega]
    
    print("\n" + "="*60)
    print("BIPED ROBOT KEYBOARD CONTROL")
    print("="*60)
    print("Type commands and press ENTER:")
    print("  w/s - forward/backward velocity")
    print("  a/d - left/right velocity") 
    print("  q/e - turn left/right")
    print("  x   - stop all motion")
    print("  r   - reset robot")
    print("  h   - show help")
    print("  exit - quit program")
    print("="*60)
    
    # Reset environment
    obs, _ = env.reset()
    
    try:
        step_count = 0
        print(f"\nRobot ready! Current commands: vx={commands[0]:.2f}, vy={commands[1]:.2f}, omega={commands[2]:.2f}")
        print("Type a command and press ENTER (or 'exit' to quit):")
        
        while True:
            # Check for keyboard commands
            key = keyboard.get_key()
            if key:
                if key == 'exit' or key == 'quit':
                    break
                elif key == 'w':
                    commands[0] = min(commands[0] + 0.1, 1.0)
                    print(f"Forward velocity: {commands[0]:.2f}")
                elif key == 's':
                    commands[0] = max(commands[0] - 0.1, -1.0)
                    print(f"Forward velocity: {commands[0]:.2f}")
                elif key == 'a':
                    commands[1] = min(commands[1] + 0.1, 1.0)
                    print(f"Left velocity: {commands[1]:.2f}")
                elif key == 'd':
                    commands[1] = max(commands[1] - 0.1, -1.0)
                    print(f"Right velocity: {commands[1]:.2f}")
                elif key == 'q':
                    commands[2] = min(commands[2] + 0.2, 1.0)
                    print(f"Turn left: {commands[2]:.2f}")
                elif key == 'e':
                    commands[2] = max(commands[2] - 0.2, -1.0)
                    print(f"Turn right: {commands[2]:.2f}")
                elif key == 'x':
                    commands.fill_(0.0)
                    print("STOP - All commands set to zero")
                elif key == 'r':
                    obs, _ = env.reset()
                    print("Robot reset!")
                elif key == 'h':
                    print("Commands: w/s(fwd/back), a/d(left/right), q/e(turn), x(stop), r(reset), exit(quit)")
                
                print(f"Current: vx={commands[0]:.2f}, vy={commands[1]:.2f}, omega={commands[2]:.2f}")
            
            # Set commands in environment
            env.commands[0] = commands
            
            # Get action from policy or random
            with torch.no_grad():
                if use_policy:
                    action = actor(obs[0:1])
                else:
                    action = torch.randn(1, env.num_actions, device=gs.device) * 0.1
            
            # Step simulation
            obs, rewards, dones, infos = env.step(action)
            
            # Print status occasionally
            if step_count % 200 == 0:
                pos = env.base_pos[0].cpu().numpy()
                vel = env.base_lin_vel[0].cpu().numpy()
                print(f"[Step {step_count}] Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                      f"Vel: [{vel[0]:.2f}, {vel[1]:.2f}], Reward: {rewards[0]:.3f}")
            
            step_count += 1
            time.sleep(0.02)  # 50 Hz
            
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        keyboard.stop()
        print("Goodbye!")


if __name__ == "__main__":
    main()

import argparse
import os
import pickle
from importlib import metadata

import torch

# This script uses the rsl-rl-lib for loading the PPO runner and policy.
# It checks for the correct version to ensure compatibility.
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

# Genesis is the physics simulator used for the environment.
import genesis as gs

# This imports the custom environment definition for the bipedal robot.
from biped_env import BipedEnv


def main():
    """
    Main function to load a trained policy and run inference in the simulator.
    """
    # --- 1. Argument Parsing ---
    # Sets up command-line arguments to specify which trained model to load.
    parser = argparse.ArgumentParser(description="Run inference for the bipedal robot.")
    parser.add_argument(
        "-e", 
        "--exp_name", 
        type=str, 
        default="biped-walking",
        help="The name of the experiment, used to find the log directory."
    )
    parser.add_argument(
        "--ckpt", 
        type=int, 
        default=100,
        help="The checkpoint number of the model to load (e.g., 100 for 'model_100.pt')."
    )
    args = parser.parse_args()

    # --- 2. Initialization ---
    # Initialize the Genesis simulator.
    gs.init()

    # --- 3. Load Configurations and Model ---
    # Construct the path to the directory where logs and models are saved.
    log_dir = f"logs/{args.exp_name}"
    
    # Check if the specified log directory and model file exist.
    config_path = f"{log_dir}/cfgs.pkl"
    model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    print(f"Loading configurations from: {config_path}")
    print(f"Loading model checkpoint from: {model_path}")

    # Load the configuration files saved during training.
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(config_path, "rb"))
    
    # For inference, we don't need to calculate rewards. Clearing the reward scales
    # can prevent unnecessary computations.
    reward_cfg["reward_scales"] = {}

    # --- 4. Create Environment ---
    # Instantiate the bipedal environment with the loaded configurations.
    # We use num_envs=1 because we are only visualizing one robot.
    # show_viewer=True opens the simulation window.
    env = BipedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # --- 5. Load Policy ---
    # The OnPolicyRunner is used here as a convenient way to load the model
    # and get the policy, even though we are not training.
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(model_path)
    
    # Get the policy in "inference mode". This prepares the neural network
    # for efficient execution without tracking gradients.
    policy = runner.get_inference_policy(device=gs.device)

    # --- 6. Inference Loop ---
    # Reset the environment to get the first observation.
    obs, _ = env.reset()
    
    print("\nInference started. Close the simulation window to exit.")
    
    # The context `torch.no_grad()` is a performance optimization that tells PyTorch
    # not to compute gradients, making inference faster.
    with torch.no_grad():
        # Loop indefinitely to continuously control the robot.
        while True:
            # The policy takes the current observation as input and returns the
            # optimal action (motor commands) as output.
            actions = policy(obs)
            
            # The environment executes the action and returns the next state.
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# How to run this script:
# -------------------------
# Open your terminal and run the following command.
# Replace 'biped-walking' with your experiment name and '100' with your desired checkpoint.
#
# python biped_inference.py -e biped-walking --ckpt 100
#
"""

"""
Biped Policy Inference Class

This module provides a clean interface for running inference with a trained
bipedal robot policy. It loads a trained model and provides a simple interface
to get actions from observations.

Usage:
    # Initialize the inference class
    policy_inference = BipedPolicyInference(
        experiment_name="biped-walking",
        checkpoint=100,
        device="cuda"  # or "cpu"
    )
    
    # Get actions from observations
    actions = policy_inference.get_actions(observations)
    
    # Or use the callable interface
    actions = policy_inference(observations)
"""

import os
import pickle
import torch
import numpy as np
from typing import Union, Optional
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


class BipedPolicyInference:
    """
    A clean interface for running inference with a trained bipedal robot policy.
    
    This class loads a trained PPO policy and provides methods to get actions
    from observations in a stateless manner.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 checkpoint: int,
                 device: str = "cuda",
                 logs_root: str = "./logs"):
        """
        Initialize the policy inference class.
        
        Args:
            experiment_name: Name of the experiment (e.g., "biped-walking")
            checkpoint: Checkpoint number to load (e.g., 100)
            device: Device to run inference on ("cuda" or "cpu")
            logs_root: Root directory where experiment logs are stored
        """
        self.experiment_name = experiment_name
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.logs_root = logs_root
        
        # Model configuration
        self.num_obs = 38  # Based on the observation structure
        self.num_actions = 9  # 9 DOF for biped
        
        # Load the policy
        self.policy = self._load_policy()
        self.policy.eval()  # Set to evaluation mode
        
        print(f"✅ Policy loaded successfully!")
        print(f"   Experiment: {experiment_name}")
        print(f"   Checkpoint: {checkpoint}")
        print(f"   Device: {self.device}")
        print(f"   Input dim: {self.num_obs}")
        print(f"   Output dim: {self.num_actions}")
    
    def _load_policy(self):
        """Load the trained policy from the specified experiment and checkpoint."""
        # Construct the path to the experiment directory
        log_dir = os.path.join(self.logs_root, self.experiment_name)
        
        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"Experiment directory not found: {log_dir}")
        
        # Load the saved model
        model_path = os.path.join(log_dir, f"model_{self.checkpoint}.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Load the configuration
        config_path = os.path.join(log_dir, "config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        
        # Extract the training configuration
        train_cfg = config[-1]  # Last element is train_cfg
        
        # Create a dummy environment to get the policy structure
        # We only need the policy structure, not the full environment
        from training.biped_train import get_cfgs
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        
        # Create and load the policy
        runner = OnPolicyRunner(None, train_cfg, None, device=self.device)
        runner.load(log_dir, self.checkpoint)
        
        return runner.alg.actor_critic.act_inference
    
    def get_actions(self, observations: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Get actions from observations.
        
        Args:
            observations: Input observations. Can be:
                - Single observation: shape (num_obs,) or (1, num_obs)
                - Batch of observations: shape (batch_size, num_obs)
                - Numpy array or torch tensor
        
        Returns:
            torch.Tensor: Actions with shape (batch_size, num_actions)
        """
        # Convert to torch tensor if needed
        if isinstance(observations, np.ndarray):
            observations = torch.from_numpy(observations).float()
        
        # Ensure tensor is on the correct device
        observations = observations.to(self.device)
        
        # Handle single observation case
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)  # Add batch dimension
            single_obs = True
        else:
            single_obs = False
        
        # Validate input shape
        if observations.shape[-1] != self.num_obs:
            raise ValueError(f"Expected observation dimension {self.num_obs}, "
                           f"got {observations.shape[-1]}")
        
        # Run inference
        with torch.no_grad():
            actions = self.policy(observations)
        
        # Remove batch dimension if input was single observation
        if single_obs:
            actions = actions.squeeze(0)
        
        return actions
    
    def __call__(self, observations: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Callable interface for the policy.
        
        Args:
            observations: Input observations
            
        Returns:
            torch.Tensor: Actions
        """
        return self.get_actions(observations)
    
    def get_actions_numpy(self, observations: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Get actions as numpy arrays.
        
        Args:
            observations: Input observations
            
        Returns:
            np.ndarray: Actions as numpy array
        """
        actions = self.get_actions(observations)
        return actions.cpu().numpy()
    
    def set_device(self, device: str):
        """
        Change the device for inference.
        
        Args:
            device: New device ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        self.policy = self.policy.to(self.device)
        print(f"Device changed to: {self.device}")
    
    def get_info(self) -> dict:
        """
        Get information about the loaded policy.
        
        Returns:
            dict: Information about the policy
        """
        return {
            "experiment_name": self.experiment_name,
            "checkpoint": self.checkpoint,
            "device": str(self.device),
            "num_observations": self.num_obs,
            "num_actions": self.num_actions,
            "policy_type": type(self.policy).__name__
        }


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the BipedPolicyInference class.
    """
    
    # Example 1: Basic usage
    try:
        policy_inference = BipedPolicyInference(
            experiment_name="biped-walking",
            checkpoint=100,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Create dummy observation for testing
        dummy_obs = torch.randn(38)  # Single observation
        actions = policy_inference.get_actions(dummy_obs)
        print(f"Single observation test - Actions shape: {actions.shape}")
        
        # Test with batch of observations
        batch_obs = torch.randn(5, 38)  # Batch of 5 observations
        batch_actions = policy_inference.get_actions(batch_obs)
        print(f"Batch observation test - Actions shape: {batch_actions.shape}")
        
        # Test callable interface
        actions_callable = policy_inference(dummy_obs)
        print(f"Callable interface test - Actions shape: {actions_callable.shape}")
        
        # Test numpy interface
        numpy_obs = np.random.randn(38)
        numpy_actions = policy_inference.get_actions_numpy(numpy_obs)
        print(f"Numpy interface test - Actions shape: {numpy_actions.shape}")
        print(f"Actions type: {type(numpy_actions)}")
        
        # Print policy information
        info = policy_inference.get_info()
        print(f"\nPolicy Info: {info}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have a trained model in the logs directory.")
        print("Example: logs/biped-walking/model_100.pt")

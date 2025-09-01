"""
WandB Logger for PPO Training

This module provides comprehensive logging capabilities for PPO training using Weights & Biases.
"""

import wandb
import torch
import numpy as np
import os
from typing import Dict, Any, Optional, List


class WandbPPOLogger:
    """
    Comprehensive WandB logger for PPO training that tracks all key metrics.
    
    Logs:
    - Episode Returns and Lengths
    - PPO Training Metrics (Policy Loss, Value Loss, Entropy, KL Divergence)
    - Reward Components breakdown
    - Performance Metrics (FPS, timing)
    - Model checkpoints
    """
    
    def __init__(self, 
                 project_name: str = "biped-ppo-training",
                 experiment_name: str = "experiment",
                 config: Dict[str, Any] = None,
                 tags: List[str] = None,
                 notes: str = "",
                 log_frequency: int = 1):
        """
        Initialize WandB logger.
        
        Args:
            project_name: WandB project name
            experiment_name: Name for this specific run
            config: Configuration dictionary to log
            tags: List of tags for organizing runs
            notes: Description/notes for this run
            log_frequency: How often to log metrics (every N iterations)
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log_frequency = log_frequency
        
        # Initialize WandB
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            reinit=True
        )
        
        print(f"✅ W&B initialized successfully. Project: {project_name}")
        print(f"   Experiment: {experiment_name}")
        print(f"   Dashboard: {wandb.run.url}")
        
        # Tracking variables
        self.iteration_count = 0
        
    def log_full_training_step(self,
                              policy_loss: float,
                              value_loss: float, 
                              entropy: float,
                              kl_divergence: float,
                              episode_returns: torch.Tensor,
                              episode_lengths: torch.Tensor,
                              dones: torch.Tensor,
                              reward_components: Dict[str, Any],
                              fps: int,
                              collection_time: float,
                              learning_time: float,
                              total_timesteps: int,
                              learning_rate: float = None,
                              iteration_time: float = None):
        """
        Log a complete training step with all metrics.
        
        Args:
            policy_loss: Policy loss from PPO update
            value_loss: Value function loss
            entropy: Policy entropy
            kl_divergence: KL divergence between old and new policy
            episode_returns: Tensor of episode returns
            episode_lengths: Tensor of episode lengths  
            dones: Boolean tensor indicating completed episodes
            reward_components: Dictionary of individual reward components
            fps: Frames per second
            collection_time: Time spent collecting rollouts
            learning_time: Time spent on learning updates
            total_timesteps: Total timesteps collected so far
            learning_rate: Current learning rate (optional)
            iteration_time: Total iteration time (optional)
        """
        
        # Only log every log_frequency iterations
        if self.iteration_count % self.log_frequency != 0:
            self.iteration_count += 1
            return
        
        # Prepare metrics dictionary
        metrics = {}
        
        # PPO Training Metrics
        metrics["ppo/policy_loss"] = policy_loss
        metrics["ppo/value_loss"] = value_loss  
        metrics["ppo/entropy"] = entropy
        metrics["ppo/kl_divergence"] = kl_divergence
        
        if learning_rate is not None:
            metrics["ppo/learning_rate"] = learning_rate
        
        # Episode Metrics
        if len(episode_returns) > 0 and torch.any(dones):
            # Only log statistics for completed episodes
            completed_returns = episode_returns[dones]
            completed_lengths = episode_lengths[dones]
            
            if len(completed_returns) > 0:
                metrics["episode/mean_return"] = torch.mean(completed_returns).item()
                metrics["episode/max_return"] = torch.max(completed_returns).item()
                metrics["episode/min_return"] = torch.min(completed_returns).item()
                metrics["episode/mean_length"] = torch.mean(completed_lengths.float()).item()
                metrics["episode/max_length"] = torch.max(completed_lengths).item()
                metrics["episode/min_length"] = torch.min(completed_lengths).item()
                metrics["episode/completed_this_iter"] = len(completed_returns)
                metrics["episode/count"] = self.iteration_count
        
        # Performance Metrics
        metrics["performance/fps"] = fps
        metrics["performance/collection_time"] = collection_time
        metrics["performance/learning_time"] = learning_time
        metrics["performance/total_timesteps"] = total_timesteps
        
        if iteration_time is not None:
            metrics["performance/iteration_time"] = iteration_time
            if iteration_time > 0:
                metrics["performance/learning_ratio"] = learning_time / iteration_time
        
        # Reward Components
        for component_name, component_value in reward_components.items():
            if isinstance(component_value, torch.Tensor):
                metrics[f"rewards/{component_name}"] = component_value.item()
            else:
                metrics[f"rewards/{component_name}"] = float(component_value)
        
        # Log to WandB
        wandb.log(metrics, step=self.iteration_count)
        
        self.iteration_count += 1
    
    def log_custom_plots(self):
        """Log custom plots and visualizations."""
        # This can be extended to add custom plots
        # For now, we'll just log basic status
        wandb.log({"status/logger_active": 1}, step=self.iteration_count)
    
    def save_model_checkpoint(self, model_path: str, iteration: int):
        """
        Save model checkpoint to WandB artifacts.
        
        Args:
            model_path: Path to the saved model file
            iteration: Training iteration number
        """
        try:
            if os.path.exists(model_path):
                # Create artifact for model checkpoint
                artifact_name = f"model_checkpoint_iter_{iteration}"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    description=f"Model checkpoint at iteration {iteration}"
                )
                
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                
                print(f"✅ Model checkpoint saved to W&B: {model_path}")
            else:
                print(f"⚠️ Model file not found: {model_path}")
                
        except Exception as e:
            print(f"❌ Failed to save model checkpoint to W&B: {e}")
    
    def finish(self):
        """Finish the WandB run."""
        try:
            wandb.finish()
            print("✅ W&B run finished successfully")
        except Exception as e:
            print(f"⚠️ Error finishing W&B run: {e}")


# Helper function for easy initialization
def init_wandb_logger(project_name: str = "biped-ppo-training", 
                     experiment_name: str = "experiment",
                     config: Dict[str, Any] = None,
                     **kwargs) -> WandbPPOLogger:
    """
    Convenience function to initialize WandB logger.
    
    Args:
        project_name: WandB project name
        experiment_name: Experiment name
        config: Configuration to log
        **kwargs: Additional arguments for WandbPPOLogger
    
    Returns:
        Initialized WandbPPOLogger instance
    """
    return WandbPPOLogger(
        project_name=project_name,
        experiment_name=experiment_name, 
        config=config,
        **kwargs
    )

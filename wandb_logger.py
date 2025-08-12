"""
Weights & Biases (wandb) Integration for PPO Biped Training

This module provides comprehensive logging and visualization of PPO training metrics
using Weights & Biases for real-time monitoring and experiment tracking.
"""

import wandb
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import time
import os
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class WandbPPOLogger:
    """
    Comprehensive W&B logger for PPO biped robot training.
    
    Logs all key metrics including:
    - Episode returns and statistics
    - PPO algorithm metrics (policy loss, value loss, entropy, KL divergence)
    - Reward components breakdown
    - Training performance metrics
    - System performance (FPS, computation times)
    """
    
    def __init__(
        self, 
        project_name: str = "biped-ppo-training",
        experiment_name: str = None,
        config: Dict = None,
        tags: List[str] = None,
        notes: str = None,
        resume: bool = False,
        log_frequency: int = 1,
        save_model: bool = True
    ):
        """
        Initialize wandb logger.
        
        Args:
            project_name: Name of the wandb project
            experiment_name: Name of this specific experiment/run
            config: Configuration dictionary to log
            tags: List of tags for this run
            notes: Description/notes for this run
            resume: Whether to resume from previous run
            log_frequency: How often to log metrics (every N iterations)
            save_model: Whether to save models to wandb
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"biped-training-{int(time.time())}"
        self.log_frequency = log_frequency
        self.save_model = save_model
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=tags,
            notes=notes,
            resume="allow" if resume else None
        )
        
        # Metrics tracking
        self.iteration_count = 0
        self.episode_count = 0
        self.total_timesteps = 0
        self.start_time = time.time()
        
        # Episode metrics buffers for statistics
        self.episode_returns = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        
        print(f"🚀 W&B Logger initialized for project: {project_name}")
        print(f"📊 Run name: {self.experiment_name}")
        print(f"🔗 Dashboard: {wandb.run.url}")
        
    def log_ppo_metrics(
        self,
        policy_loss: float,
        value_loss: float, 
        entropy: float,
        kl_divergence: float,
        learning_rate: float = None,
        gradient_norm: float = None,
        action_noise_std: float = None
    ):
        """Log PPO algorithm-specific metrics."""
        metrics = {
            "ppo/policy_loss": policy_loss,
            "ppo/value_loss": value_loss,
            "ppo/entropy": entropy, 
            "ppo/kl_divergence": kl_divergence,
        }
        
        if learning_rate is not None:
            metrics["ppo/learning_rate"] = learning_rate
        if gradient_norm is not None:
            metrics["ppo/gradient_norm"] = gradient_norm
        if action_noise_std is not None:
            metrics["ppo/action_noise_std"] = action_noise_std
            
        wandb.log(metrics, step=self.iteration_count)
        
    def log_episode_metrics(
        self,
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor, 
        dones: torch.Tensor
    ):
        """Log episode-level metrics."""
        # Only log for completed episodes
        completed_episodes = dones.nonzero(as_tuple=False).flatten()
        
        if len(completed_episodes) > 0:
            # Extract metrics for completed episodes
            completed_returns = episode_returns[completed_episodes]
            completed_lengths = episode_lengths[completed_episodes]
            
            # Update buffers
            for ret, length in zip(completed_returns, completed_lengths):
                self.episode_returns.append(ret.item())
                self.episode_lengths.append(length.item())
                self.episode_count += 1
            
            # Log episode statistics
            metrics = {
                "episode/mean_return": torch.mean(completed_returns).item(),
                "episode/max_return": torch.max(completed_returns).item(),
                "episode/min_return": torch.min(completed_returns).item(),
                "episode/mean_length": torch.mean(completed_lengths.float()).item(),
                "episode/max_length": torch.max(completed_lengths).item(),
                "episode/min_length": torch.min(completed_lengths).item(),
                "episode/count": self.episode_count,
                "episode/completed_this_iter": len(completed_episodes)
            }
            
            # Rolling statistics (last 100 episodes)
            if len(self.episode_returns) > 10:
                recent_returns = list(self.episode_returns)[-100:]
                recent_lengths = list(self.episode_lengths)[-100:]
                
                metrics.update({
                    "episode/recent_mean_return": np.mean(recent_returns),
                    "episode/recent_std_return": np.std(recent_returns),
                    "episode/recent_mean_length": np.mean(recent_lengths),
                    "episode/recent_std_length": np.std(recent_lengths)
                })
            
            wandb.log(metrics, step=self.iteration_count)
            
    def log_reward_components(self, reward_dict: Dict[str, torch.Tensor]):
        """Log detailed breakdown of reward components."""
        metrics = {}
        
        for reward_name, reward_values in reward_dict.items():
            if isinstance(reward_values, torch.Tensor):
                mean_reward = torch.mean(reward_values).item()
                metrics[f"rewards/{reward_name}"] = mean_reward
                
                # Log distribution statistics for key rewards
                if reward_name in ["tracking_lin_vel_x", "tracking_lin_vel_y", "fall_penalty"]:
                    metrics[f"rewards/{reward_name}_std"] = torch.std(reward_values).item()
                    metrics[f"rewards/{reward_name}_max"] = torch.max(reward_values).item()
                    metrics[f"rewards/{reward_name}_min"] = torch.min(reward_values).item()
        
        wandb.log(metrics, step=self.iteration_count)
        
    def log_performance_metrics(
        self,
        fps: int,
        collection_time: float,
        learning_time: float,
        total_timesteps: int,
        iteration_time: float = None
    ):
        """Log training performance and timing metrics."""
        self.total_timesteps = total_timesteps
        
        # Calculate training progress
        elapsed_time = time.time() - self.start_time
        timesteps_per_second = total_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        metrics = {
            "performance/fps": fps,
            "performance/collection_time": collection_time,
            "performance/learning_time": learning_time,
            "performance/total_timesteps": total_timesteps,
            "performance/timesteps_per_second": timesteps_per_second,
            "performance/elapsed_time": elapsed_time,
        }
        
        if iteration_time is not None:
            metrics["performance/iteration_time"] = iteration_time
            
        wandb.log(metrics, step=self.iteration_count)
        
    def log_environment_info(
        self,
        num_environments: int,
        episode_length_s: float,
        action_scale: float,
        control_frequency: float
    ):
        """Log environment configuration information."""
        metrics = {
            "env/num_environments": num_environments,
            "env/episode_length_s": episode_length_s,
            "env/action_scale": action_scale,
            "env/control_frequency": control_frequency,
        }
        
        wandb.log(metrics, step=self.iteration_count)
        
    def log_hyperparameters(self, config: Dict):
        """Log hyperparameters and configuration."""
        wandb.config.update(config)
        
    def log_full_training_step(
        self,
        # PPO metrics
        policy_loss: float,
        value_loss: float,
        entropy: float,
        kl_divergence: float,
        
        # Episode metrics  
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor,
        dones: torch.Tensor,
        
        # Reward components
        reward_components: Dict[str, torch.Tensor],
        
        # Performance metrics
        fps: int,
        collection_time: float,
        learning_time: float,
        total_timesteps: int,
        
        # Optional metrics
        learning_rate: float = None,
        gradient_norm: float = None,
        action_noise_std: float = None,
        iteration_time: float = None
    ):
        """Log all metrics for a complete training iteration."""
        if self.iteration_count % self.log_frequency == 0:
            # Log all metric categories
            self.log_ppo_metrics(
                policy_loss, value_loss, entropy, kl_divergence,
                learning_rate, gradient_norm, action_noise_std
            )
            
            self.log_episode_metrics(episode_returns, episode_lengths, dones)
            self.log_reward_components(reward_components)
            self.log_performance_metrics(fps, collection_time, learning_time, total_timesteps, iteration_time)
        
        self.iteration_count += 1
        
    def log_custom_plots(self):
        """Create and log custom visualization plots."""
        if len(self.episode_returns) < 10:
            return
            
        # Create episode returns plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Episode returns over time
        axes[0, 0].plot(list(self.episode_returns))
        axes[0, 0].set_title('Episode Returns')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].grid(True)
        
        # Episode lengths over time
        axes[0, 1].plot(list(self.episode_lengths))
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True)
        
        # Returns distribution
        axes[1, 0].hist(list(self.episode_returns), bins=30, alpha=0.7)
        axes[1, 0].set_title('Episode Returns Distribution')
        axes[1, 0].set_xlabel('Return')
        axes[1, 0].set_ylabel('Frequency')
        
        # Lengths distribution
        axes[1, 1].hist(list(self.episode_lengths), bins=30, alpha=0.7)
        axes[1, 1].set_title('Episode Lengths Distribution')
        axes[1, 1].set_xlabel('Length')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({"custom_plots/training_overview": wandb.Image(fig)}, step=self.iteration_count)
        plt.close(fig)
        
    def save_model_checkpoint(self, model_path: str, iteration: int):
        """Save model checkpoint to wandb."""
        if self.save_model and os.path.exists(model_path):
            artifact = wandb.Artifact(
                name=f"model-{self.experiment_name}",
                type="model",
                description=f"Model checkpoint at iteration {iteration}"
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            
    def finish(self):
        """Finish the wandb run."""
        # Log final custom plots
        self.log_custom_plots()
        
        # Log final summary
        if len(self.episode_returns) > 0:
            final_metrics = {
                "final/total_episodes": len(self.episode_returns),
                "final/mean_episode_return": np.mean(list(self.episode_returns)),
                "final/best_episode_return": np.max(list(self.episode_returns)),
                "final/total_timesteps": self.total_timesteps,
                "final/total_training_time": time.time() - self.start_time,
            }
            wandb.log(final_metrics)
        
        wandb.finish()
        print("🏁 W&B logging finished!")


class WandbTrainingIntegration:
    """
    Integration class that connects the biped environment with W&B logging.
    This class acts as a bridge between the training loop and the W&B logger.
    """
    
    def __init__(self, logger: WandbPPOLogger):
        self.logger = logger
        self.last_log_time = time.time()
        
    def setup_environment_logging(self, env):
        """Set up logging hooks in the environment."""
        # Store original step method
        original_step = env.step
        
        def step_with_logging(actions):
            # Call original step
            obs, rewards, dones, infos = original_step(actions)
            
            # Extract metrics for logging
            if hasattr(env, 'episode_length_buf') and hasattr(env, 'episode_sums'):
                # Log episode metrics if episodes completed
                if dones.any():
                    # Calculate episode returns (approximate)
                    episode_returns = rewards * env.episode_length_buf.float()
                    
                    self.logger.log_episode_metrics(
                        episode_returns=episode_returns,
                        episode_lengths=env.episode_length_buf,
                        dones=dones
                    )
                    
                    # Log reward components
                    reward_components = {}
                    for reward_name, reward_sum in env.episode_sums.items():
                        if isinstance(reward_sum, torch.Tensor):
                            reward_components[reward_name] = reward_sum
                            
                    self.logger.log_reward_components(reward_components)
            
            return obs, rewards, dones, infos
        
        # Replace environment step method
        env.step = step_with_logging
        
    def log_training_iteration(
        self,
        ppo_metrics: Dict[str, float],
        performance_metrics: Dict[str, float],
        env_info: Dict = None
    ):
        """Log a complete training iteration."""
        # Extract PPO metrics
        policy_loss = ppo_metrics.get('policy_loss', 0.0)
        value_loss = ppo_metrics.get('value_loss', 0.0)
        entropy = ppo_metrics.get('entropy', 0.0)
        kl_divergence = ppo_metrics.get('kl_divergence', 0.01)
        learning_rate = ppo_metrics.get('learning_rate', None)
        
        # Extract performance metrics
        fps = performance_metrics.get('fps', 0)
        collection_time = performance_metrics.get('collection_time', 0.0)
        learning_time = performance_metrics.get('learning_time', 0.0)
        total_timesteps = performance_metrics.get('total_timesteps', 0)
        iteration_time = performance_metrics.get('iteration_time', None)
        
        # Log PPO metrics
        self.logger.log_ppo_metrics(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            kl_divergence=kl_divergence,
            learning_rate=learning_rate
        )
        
        # Log performance metrics
        self.logger.log_performance_metrics(
            fps=fps,
            collection_time=collection_time,
            learning_time=learning_time,
            total_timesteps=total_timesteps,
            iteration_time=iteration_time
        )
        
        # Log environment info if provided
        if env_info:
            self.logger.log_environment_info(
                num_environments=env_info.get('num_envs', 0),
                episode_length_s=env_info.get('episode_length_s', 0.0),
                action_scale=env_info.get('action_scale', 1.0),
                control_frequency=env_info.get('control_frequency', 50.0)
            )


def create_wandb_config(
    env_cfg: Dict,
    obs_cfg: Dict,
    reward_cfg: Dict,
    command_cfg: Dict,
    train_cfg: Dict,
    num_envs: int
) -> Dict:
    """Create a comprehensive configuration dictionary for wandb logging."""
    config = {
        # Environment configuration
        "env/num_envs": num_envs,
        "env/num_actions": env_cfg.get("num_actions", 9),
        "env/episode_length_s": env_cfg.get("episode_length_s", 20.0),
        "env/action_scale": env_cfg.get("action_scale", 0.25),
        "env/kp": env_cfg.get("kp", 30.0),
        "env/kd": env_cfg.get("kd", 0.5),
        
        # Observation configuration
        "obs/num_obs": obs_cfg.get("num_obs", 38),
        
        # Training configuration
        "train/num_steps_per_env": train_cfg.get("num_steps_per_env", 24),
        "train/save_interval": train_cfg.get("save_interval", 100),
        
        # PPO algorithm configuration
        "ppo/clip_param": train_cfg["algorithm"].get("clip_param", 0.2),
        "ppo/desired_kl": train_cfg["algorithm"].get("desired_kl", 0.01),
        "ppo/entropy_coef": train_cfg["algorithm"].get("entropy_coef", 0.01),
        "ppo/gamma": train_cfg["algorithm"].get("gamma", 0.99),
        "ppo/lam": train_cfg["algorithm"].get("lam", 0.95),
        "ppo/learning_rate": train_cfg["algorithm"].get("learning_rate", 0.001),
        "ppo/num_learning_epochs": train_cfg["algorithm"].get("num_learning_epochs", 5),
        "ppo/num_mini_batches": train_cfg["algorithm"].get("num_mini_batches", 4),
        "ppo/value_loss_coef": train_cfg["algorithm"].get("value_loss_coef", 1.0),
        
        # Policy network configuration
        "policy/actor_hidden_dims": train_cfg["policy"].get("actor_hidden_dims", [512, 256, 128]),
        "policy/critic_hidden_dims": train_cfg["policy"].get("critic_hidden_dims", [512, 256, 128]),
        "policy/activation": train_cfg["policy"].get("activation", "elu"),
        
        # Reward scaling
        "rewards/scales": reward_cfg.get("reward_scales", {}),
        
        # Command configuration
        "commands/num_commands": command_cfg.get("num_commands", 3),
    }
    
    return config

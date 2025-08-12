"""
Integration wrapper for biped environment with W&B logging.

This module provides integration between the existing biped training setup
and the W&B logging system, extending the current PPO metrics tracking.
"""

import torch
import time
from typing import Dict, Any, Optional
from training.wandb_logger import WandbPPOLogger, WandbTrainingIntegration
from training.biped_env import BipedEnv


class WandbBipedEnvWrapper(BipedEnv):
    """
    Extended biped environment with W&B logging integration.
    Inherits from the original BipedEnv and adds comprehensive W&B logging.
    """
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, log_dir=None):
        # Initialize parent environment
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer, log_dir)
        
        # Initialize W&B logger
        self.wandb_logger: Optional[WandbPPOLogger] = None
        self.wandb_integration: Optional[WandbTrainingIntegration] = None
        self.training_iteration = 0
        self.log_frequency = 1  # Log every iteration
        
        # Performance tracking
        self.step_timer = time.time()
        self.collection_start_time = None
        self.learning_start_time = None
        
        print("🔧 W&B Biped Environment Wrapper initialized")
        
    def setup_wandb_logging(
        self,
        project_name: str = "biped-ppo-genesis",
        experiment_name: str = None,
        config: Dict = None,
        tags: list = None,
        notes: str = None,
        log_frequency: int = 1
    ):
        """Initialize and configure W&B logging."""
        self.log_frequency = log_frequency
        
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = f"biped-{int(time.time())}"
            
        # Initialize W&B logger
        self.wandb_logger = WandbPPOLogger(
            project_name=project_name,
            experiment_name=experiment_name,
            config=config,
            tags=tags,
            notes=notes,
            log_frequency=log_frequency
        )
        
        # Create integration helper
        self.wandb_integration = WandbTrainingIntegration(self.wandb_logger)
        
        # Log environment configuration
        self.wandb_logger.log_environment_info(
            num_environments=self.num_envs,
            episode_length_s=self.env_cfg["episode_length_s"],
            action_scale=self.env_cfg["action_scale"],
            control_frequency=1.0 / self.dt
        )
        
        print(f"📊 W&B logging configured for project: {project_name}")
        
    def step(self, actions):
        """Enhanced step function with W&B logging."""
        # Record collection start time
        if self.collection_start_time is None:
            self.collection_start_time = time.time()
            
        # Call parent step function
        obs, rewards, dones, infos = super().step(actions)
        
        # Log episode-level metrics when episodes complete
        if self.wandb_logger and dones.any():
            try:
                completed_episodes = dones.nonzero(as_tuple=False).flatten()
                if len(completed_episodes) > 0:
                    # Calculate total episode rewards from all reward components
                    total_episode_rewards = torch.zeros_like(rewards)
                    for key, reward_sum in self.episode_sums.items():
                        if key != "fps":  # Skip FPS as it's not a reward
                            total_episode_rewards += reward_sum
                    
                    self.wandb_logger.log_episode_metrics(
                        episode_returns=total_episode_rewards,
                        episode_lengths=self.episode_length_buf,
                        dones=dones
                    )
                    
                    # Log reward components
                    self.wandb_logger.log_reward_components(self.episode_sums)
            except Exception as e:
                # Log warning but don't crash training
                print(f"⚠️ W&B episode logging warning: {e}")
            
        return obs, rewards, dones, infos
        
    def log_training_iteration(
        self,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        kl_divergence: float,
        learning_rate: float = None,
        gradient_norm: float = None,
        action_noise_std: float = None,
        total_timesteps: int = None
    ):
        """Log PPO training metrics for this iteration."""
        if not self.wandb_logger:
            return
            
        try:
            # Calculate performance metrics
            current_time = time.time()
            
            # Collection timing
            collection_time = 0.0
            if self.collection_start_time:
                collection_time = current_time - self.collection_start_time
                self.collection_start_time = None
                
            # Learning timing (if learning phase started)
            learning_time = 0.0
            if self.learning_start_time:
                learning_time = current_time - self.learning_start_time
                self.learning_start_time = None
                
            # Calculate FPS
            steps_since_last = self.num_envs * getattr(self, 'num_steps_per_env', 24)
            iteration_time = current_time - self.step_timer
            fps = int(steps_since_last / iteration_time) if iteration_time > 0 else 0
            self.step_timer = current_time
            
            # Total timesteps
            if total_timesteps is None:
                total_timesteps = self.training_iteration * steps_since_last
                
            # Log all metrics using the full training step function
            dummy_episode_returns = torch.zeros(self.num_envs)
            dummy_episode_lengths = torch.zeros(self.num_envs, dtype=torch.long)
            dummy_dones = torch.zeros(self.num_envs, dtype=torch.bool)
            
            self.wandb_logger.log_full_training_step(
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                kl_divergence=kl_divergence,
                episode_returns=dummy_episode_returns,
                episode_lengths=dummy_episode_lengths,
                dones=dummy_dones,
                reward_components=self.episode_sums,
                fps=fps,
                collection_time=collection_time,
                learning_time=learning_time,
                total_timesteps=total_timesteps,
                learning_rate=learning_rate,
                gradient_norm=gradient_norm,
                action_noise_std=action_noise_std,
                iteration_time=iteration_time
            )
            
            self.training_iteration += 1
            
            # Log custom plots periodically
            if self.training_iteration % (self.log_frequency * 10) == 0:
                self.wandb_logger.log_custom_plots()
                
        except Exception as e:
            # Log warning but don't crash training
            print(f"⚠️ W&B training iteration logging warning: {e}")
            
    def start_learning_phase(self):
        """Mark the start of the learning phase for timing."""
        self.learning_start_time = time.time()
        
    def log_model_checkpoint(self, model_path: str):
        """Log model checkpoint to W&B."""
        if self.wandb_logger:
            self.wandb_logger.save_model_checkpoint(model_path, self.training_iteration)
            
    def finish_wandb_logging(self):
        """Finish W&B logging."""
        if self.wandb_logger:
            self.wandb_logger.finish()
            print("🏁 W&B logging session finished!")


def parse_rsl_rl_output(output_text: str) -> Dict[str, float]:
    """
    Parse RSL-RL training output to extract metrics.
    
    Parses output like:
    "Computation: 5588 steps/s (collection: 4.063s, learning 0.232s)
    Value function loss: 1241016460.8000
    Surrogate loss: 0.0957
    Mean action noise std: 1.01
    Mean total reward: 230234.44"
    """
    metrics = {}
    lines = output_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Parse computation line
        if line.startswith("Computation:"):
            parts = line.split()
            # Extract FPS
            if "steps/s" in line:
                fps_part = line.split("steps/s")[0].split(":")[-1].strip()
                try:
                    metrics["fps"] = float(fps_part)
                except:
                    pass
                    
            # Extract collection and learning times
            if "collection:" in line and "learning" in line:
                try:
                    collection_part = line.split("collection:")[1].split("s")[0].strip()
                    learning_part = line.split("learning")[1].split("s")[0].strip()
                    metrics["collection_time"] = float(collection_part)
                    metrics["learning_time"] = float(learning_part)
                except:
                    pass
                    
        # Parse loss values
        elif "Value function loss:" in line:
            try:
                value = float(line.split(":")[-1].strip())
                metrics["value_loss"] = value
            except:
                pass
                
        elif "Surrogate loss:" in line:
            try:
                value = float(line.split(":")[-1].strip())
                metrics["policy_loss"] = value
            except:
                pass
                
        elif "Mean action noise std:" in line:
            try:
                value = float(line.split(":")[-1].strip())
                metrics["action_noise_std"] = value
            except:
                pass
                
        elif "Mean total reward:" in line:
            try:
                value = float(line.split(":")[-1].strip())
                metrics["mean_total_reward"] = value
            except:
                pass
                
        elif "Mean episode length:" in line:
            try:
                value = float(line.split(":")[-1].strip())
                metrics["mean_episode_length"] = value
            except:
                pass
                
        # Parse reward components
        elif "Mean episode rew_" in line and ":" in line:
            try:
                reward_name = line.split("Mean episode rew_")[1].split(":")[0].strip()
                reward_value = float(line.split(":")[-1].strip())
                metrics[f"reward_{reward_name}"] = reward_value
            except:
                pass
                
        # Parse timing information
        elif "Total timesteps:" in line:
            try:
                value = int(line.split(":")[-1].strip())
                metrics["total_timesteps"] = value
            except:
                pass
                
        elif "Iteration time:" in line:
            try:
                value = float(line.split(":")[-1].strip().rstrip("s"))
                metrics["iteration_time"] = value
            except:
                pass
                
    return metrics


def create_wandb_biped_config(cfg: Dict) -> Dict:
    """Create comprehensive wandb config from biped environment config."""
    config = {
        # Environment settings
        "env/num_envs": cfg.get("num_envs", 1),
        "env/num_actions": cfg["env"].get("num_actions", 9),
        "env/episode_length_s": cfg["env"].get("episode_length_s", 20.0),
        "env/action_scale": cfg["env"].get("action_scale", 0.25),
        "env/kp": cfg["env"].get("kp", 30.0),
        "env/kd": cfg["env"].get("kd", 0.5),
        
        # Observation settings
        "obs/num_obs": cfg["env"]["obs"].get("num_obs", 38),
        
        # Training settings
        "train/num_steps_per_env": cfg["train"].get("num_steps_per_env", 24),
        "train/max_iterations": cfg["train"].get("max_iterations", 1500),
        "train/save_interval": cfg["train"].get("save_interval", 100),
        
        # PPO Algorithm settings
        "ppo/clip_param": cfg["train"]["algorithm"].get("clip_param", 0.2),
        "ppo/desired_kl": cfg["train"]["algorithm"].get("desired_kl", 0.01),
        "ppo/entropy_coef": cfg["train"]["algorithm"].get("entropy_coef", 0.01),
        "ppo/gamma": cfg["train"]["algorithm"].get("gamma", 0.99),
        "ppo/lam": cfg["train"]["algorithm"].get("lam", 0.95),
        "ppo/learning_rate": cfg["train"]["algorithm"].get("learning_rate", 0.001),
        "ppo/num_learning_epochs": cfg["train"]["algorithm"].get("num_learning_epochs", 5),
        "ppo/num_mini_batches": cfg["train"]["algorithm"].get("num_mini_batches", 4),
        "ppo/value_loss_coef": cfg["train"]["algorithm"].get("value_loss_coef", 1.0),
        
        # Policy network
        "policy/actor_hidden_dims": cfg["train"]["policy"].get("actor_hidden_dims", [512, 256, 128]),
        "policy/critic_hidden_dims": cfg["train"]["policy"].get("critic_hidden_dims", [512, 256, 128]),
        "policy/activation": cfg["train"]["policy"].get("activation", "elu"),
        
        # Device settings
        "device/sim_device": cfg.get("sim_device", "cuda:0"),
        "device/graphics_device": cfg.get("graphics_device", 0),
        
        # Reward scales
        "rewards": cfg["env"]["reward"].get("reward_scales", {})
    }
    
    return config

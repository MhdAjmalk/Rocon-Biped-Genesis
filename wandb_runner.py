"""
Custom PPO Runner with WandB Integration

This extends the RSL-RL OnPolicyRunner to extract and log all training metrics to WandB.
"""

import os
import time
import torch
from typing import Dict, Any
from rsl_rl.runners import OnPolicyRunner
from wandb_logger import WandbPPOLogger


class WandbOnPolicyRunner(OnPolicyRunner):
    """
    Extended OnPolicyRunner that integrates WandB logging for all training metrics.
    
    Captures and logs:
    - Episode Returns
    - Policy Loss
    - Value Loss  
    - Entropy
    - KL Divergence
    - Mean total reward
    - Performance metrics (FPS, timing)
    - Reward components
    """
    
    def __init__(self, env, train_cfg, log_dir, device='cpu', wandb_config=None):
        """
        Initialize runner with WandB integration.
        
        Args:
            env: Training environment
            train_cfg: Training configuration
            log_dir: Directory for logs
            device: Training device
            wandb_config: Configuration for WandB logging
        """
        super().__init__(env, train_cfg, log_dir, device)
        
        # Initialize WandB logger
        wandb_config = wandb_config or {}
        experiment_name = wandb_config.get('experiment_name', os.path.basename(log_dir))
        
        # Combine all configs for WandB
        full_config = {
            'train_cfg': train_cfg,
            'env_cfg': getattr(env, 'env_cfg', {}),
            'obs_cfg': getattr(env, 'obs_cfg', {}),
            'reward_cfg': getattr(env, 'reward_cfg', {}),
            'command_cfg': getattr(env, 'command_cfg', {}),
        }
        
        self.wandb_logger = WandbPPOLogger(
            project_name=wandb_config.get('project_name', 'biped-ppo-training'),
            experiment_name=experiment_name,
            config=full_config,
            tags=wandb_config.get('tags', ['biped', 'ppo', 'genesis']),
            notes=wandb_config.get('notes', 'Biped robot training with PPO'),
            log_frequency=wandb_config.get('log_frequency', 1)
        )
        
        # Timing tracking
        self.step_start_time = time.time()
        self.collection_start_time = None
        self.learning_start_time = None
        
        # Set logger_type for parent class compatibility
        self.logger_type = "wandb"
        
        print("✅ WandB PPO Runner initialized successfully")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        Enhanced learning loop with comprehensive WandB logging.
        """
        # Initialize episode tracking
        obs_data = self.env.get_observations()
        obs = obs_data[0] if isinstance(obs_data, tuple) else obs_data
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        
        for it in range(self.current_learning_iteration, num_learning_iterations):
            self.current_learning_iteration = it
            
            # Start timing collection phase
            collection_start = time.time()
            
            # =================== COLLECTION PHASE ===================
            
            for i in range(self.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs)
                obs, rewards, dones, infos = self.env.step(actions)
                privileged_obs = self.env.get_privileged_observations()
                critic_obs = privileged_obs if privileged_obs is not None else obs
                self.alg.process_env_step(rewards, dones, infos)
            
            collection_time = time.time() - collection_start
            
            # =================== LEARNING PHASE ===================
            learning_start = time.time()
            
            self.alg.compute_returns(critic_obs)
            
            # Track metrics during learning
            mean_value_loss = 0.0
            mean_surrogate_loss = 0.0
            mean_entropy = 0.0
            mean_kl = 0.0
            total_updates = 0
            
            # Perform PPO updates and collect metrics
            for epoch in range(self.alg.num_learning_epochs):
                for i in range(self.alg.num_mini_batches):
                    # Get metrics before update (if available)
                    update_info = self.alg.update()
                    
                    # Accumulate metrics
                    if isinstance(update_info, dict):
                        mean_value_loss += update_info.get('value_loss', 0.0)
                        mean_surrogate_loss += update_info.get('surrogate_loss', 0.0)
                        mean_entropy += update_info.get('entropy', 0.0)
                        mean_kl += update_info.get('kl', 0.0)
                        total_updates += 1
            
            # Average metrics over all updates
            if total_updates > 0:
                mean_value_loss /= total_updates
                mean_surrogate_loss /= total_updates
                mean_entropy /= total_updates
                mean_kl /= total_updates
            
            learning_time = time.time() - learning_start
            
            # =================== WANDB LOGGING PHASE ===================
            
            # Calculate performance metrics
            total_time = time.time() - collection_start
            steps_collected = self.num_steps_per_env * self.env.num_envs
            fps = int(steps_collected / collection_time) if collection_time > 0 else 0
            total_timesteps = it * steps_collected
            
            # Extract episode returns and lengths from environment
            episode_returns = torch.zeros(self.env.num_envs, device=self.device)
            episode_lengths = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
            dones_mask = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
            
            # Get episode data from environment extras if available
            if hasattr(self.env, 'extras') and self.env.extras:
                if 'episode_returns' in self.env.extras:
                    episode_returns = self.env.extras['episode_returns']
                    episode_lengths = self.env.extras.get('episode_lengths', episode_lengths)
                    dones_mask = torch.ones(len(episode_returns), device=self.device, dtype=torch.bool)
            
            # Extract reward components
            reward_components = {}
            if hasattr(self.env, 'episode_sums'):
                for key, value in self.env.episode_sums.items():
                    if isinstance(value, torch.Tensor):
                        reward_components[key] = torch.mean(value)
                    else:
                        reward_components[key] = value
            
            # Calculate mean total reward from reward components
            total_reward = 0.0
            for key, value in reward_components.items():
                if key != 'fps':  # Exclude FPS from reward sum
                    if isinstance(value, torch.Tensor):
                        total_reward += value.item()
                    else:
                        total_reward += float(value)
            
            # Add mean total reward to components
            reward_components['mean_total_reward'] = total_reward
            
            # Extract learning rate if available
            learning_rate = getattr(self.alg, 'learning_rate', None)
            if hasattr(self.alg, 'scheduler') and hasattr(self.alg.scheduler, 'get_last_lr'):
                learning_rate = self.alg.scheduler.get_last_lr()[0]
            
            # Log all metrics to WandB
            self.wandb_logger.log_full_training_step(
                # PPO metrics - these are the exact metrics you requested
                policy_loss=mean_surrogate_loss,      # Policy Loss
                value_loss=mean_value_loss,           # Value Loss
                entropy=mean_entropy,                 # Entropy
                kl_divergence=mean_kl,               # KL Divergence
                
                # Episode metrics
                episode_returns=episode_returns,     # Episode Return
                episode_lengths=episode_lengths,
                dones=dones_mask,
                
                # Reward components including mean total reward
                reward_components=reward_components, # Mean total reward + components
                
                # Performance metrics
                fps=fps,
                collection_time=collection_time,
                learning_time=learning_time,
                total_timesteps=total_timesteps,
                
                # Optional metrics
                learning_rate=learning_rate,
                iteration_time=total_time
            )
            
            # Console logging (existing)
            if it % self.cfg["runner"]["log_interval"] == 0:
                # Add missing variables for parent class compatibility
                learn_time = learning_time
                ep_infos = []  # Empty list for episode info
                self.log(locals())
                
                # Print WandB-specific metrics for debugging
                print(f"📊 WandB Metrics Summary:")
                print(f"   Policy Loss: {mean_surrogate_loss:.6f}")
                print(f"   Value Loss: {mean_value_loss:.6f}")
                print(f"   Entropy: {mean_entropy:.6f}")
                print(f"   KL Divergence: {mean_kl:.6f}")
                print(f"   Mean Total Reward: {total_reward:.4f}")
                print(f"   FPS: {fps}")
            
            # Save model checkpoint periodically  
            save_interval = self.cfg["runner"].get("save_interval", 100)
            if it % save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
                self.wandb_logger.save_model_checkpoint(
                    os.path.join(self.log_dir, f'model_{it}.pt'), 
                    it
                )
            
            # Update learning rate if using scheduler
            if hasattr(self.alg, 'update_learning_rate'):
                self.alg.update_learning_rate(it)
            
            # Log custom plots periodically
            if it % (self.cfg["runner"]["log_interval"] * 10) == 0:
                self.wandb_logger.log_custom_plots()
        
        # Save final model
        self.save(os.path.join(self.log_dir, 'model_final.pt'))
        self.wandb_logger.save_model_checkpoint(
            os.path.join(self.log_dir, 'model_final.pt'), 
            num_learning_iterations
        )
        
        # Finish WandB logging
        self.wandb_logger.finish()

    def log(self, locs, width=80, pad=35):
        """Enhanced logging that uses only WandB, skips TensorBoard."""
        # Extract metrics for console output display
        if hasattr(self.env, 'episode_sums'):
            total_reward = 0
            for key, value in self.env.episode_sums.items():
                if isinstance(value, torch.Tensor):
                    total_reward += torch.mean(value).item()
            
            print(f"Mean total reward: {total_reward:.2f}")
        
        # Print some basic info without calling parent (avoids TensorBoard issues)
        print(f"Iteration {locs['it']}: FPS {locs['fps']} | Collection: {locs['collection_time']:.3f}s | Learning: {locs['learn_time']:.3f}s")

    def save(self, path, infos=None):
        """Save model and optionally log to WandB."""
        # Save the model directly using the algorithm's save method
        # instead of calling parent's save which expects TensorBoard writer
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'current_learning_iteration': self.current_learning_iteration,
            'infos': infos
        }, path)
        
        # Log to WandB if it's a checkpoint save
        if 'model_' in os.path.basename(path):
            if hasattr(self, 'wandb_logger') and self.wandb_logger:
                self.wandb_logger.save_model_checkpoint(path, self.current_learning_iteration)

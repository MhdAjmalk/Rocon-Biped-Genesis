"""
Simple WandB Environment Metrics Patch
======================================

This patch adds WandB logging for environment metrics by monkey-patching
the OnPolicyRunner's logging functionality.
"""

import time
import numpy as np
import torch
from collections import defaultdict
from rsl_rl.runners import OnPolicyRunner

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def patch_onpolicy_runner_for_wandb():
    """
    Patches the OnPolicyRunner to enhance WandB logging with comprehensive metrics.
    
    This patch intercepts the learn() method to add:
    - Episode reward tracking
    - PPO loss metrics (policy, value, entropy)
    - Performance metrics (FPS, timing)
    - Action noise statistics
    
    All metrics are logged to WandB automatically during training.
    """
    
    # Store original method
    original_learn = OnPolicyRunner.learn
    
    def enhanced_learn(runner, num_learning_iterations, init_at_random_ep_len=False):
        # Ensure logger_type is set to avoid save method errors
        if not hasattr(runner, 'logger_type'):
            runner.logger_type = 'wandb'
        
        # Call original learn method but intercept the logging
        import os
        import torch
        from collections import deque
        
        # Standard rsl-rl initialization (fixed for tuple returns)
        obs_data = runner.env.get_observations()
        if isinstance(obs_data, tuple):
            obs, _ = obs_data  # Unpack tuple (obs, extras)
        else:
            obs = obs_data
            
        privileged_obs = runner.env.get_privileged_observations()  # This returns tensor directly
            
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(runner.device), critic_obs.to(runner.device)
        runner.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
        cur_episode_length = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)

        tot_timesteps = 0
        tot_time = 0
        start_time = time.time()

        for it in range(runner.current_learning_iteration, num_learning_iterations):
            start = time.time()
            
            # Rollout
            # Data collection phase - use no_grad for efficiency but allow normal tensor computation
            with torch.no_grad():
                for i in range(runner.num_steps_per_env):
                    actions = runner.alg.act(obs, critic_obs)
                    # Collect additional environment data for logging
                    obs, rewards, dones, infos = runner.env.step(actions)
                    privileged_obs = infos.get("observations", {}).get("critic", None)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(runner.device), critic_obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                    runner.alg.process_env_step(rewards, dones, infos)
                    
                    if runner.log_dir is not None:
                        # Rewards
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            # Make sure critic_obs is a normal tensor for training (no need to clone, it should be normal after no_grad)
            runner.alg.compute_returns(critic_obs)

            update_result = runner.alg.update()
            
            # Handle variable return values from update method
            if isinstance(update_result, tuple):
                # Extract values based on what's returned
                if len(update_result) >= 3:
                    mean_value_loss, mean_surrogate_loss, mean_entropy = update_result[:3]
                elif len(update_result) == 2:
                    mean_value_loss, mean_surrogate_loss = update_result
                    mean_entropy = 0.0
                elif len(update_result) == 1:
                    mean_value_loss = update_result[0]
                    mean_surrogate_loss = 0.0
                    mean_entropy = 0.0
                else:
                    mean_value_loss = mean_surrogate_loss = mean_entropy = 0.0
            else:
                # Update method might return None or a single value
                mean_value_loss = mean_surrogate_loss = mean_entropy = 0.0
            stop = time.time()
            learn_time = stop - start
            
            # Update counters and timers
            runner.current_learning_iteration += 1
            tot_timesteps += runner.num_steps_per_env * runner.env.num_envs
            tot_time += stop - start_time
            # Optional: log action noise std if available
            try:
                if hasattr(runner.alg.actor_critic, 'log_std'):
                    mean_std = runner.alg.actor_critic.log_std.exp().mean()
                elif hasattr(runner.alg.actor_critic, 'std'):
                    mean_std = runner.alg.actor_critic.std.mean()
                else:
                    mean_std = torch.tensor(1.0)  # Default value
            except:
                mean_std = torch.tensor(1.0)  # Fallback

            # Enhanced logging with WandB
            # Log metrics every iteration (or use a default interval)
            log_interval = getattr(runner, 'log_interval', 1)  # Default to every iteration
            if runner.log_dir is not None and it % log_interval == 0:
                fps = int(runner.num_steps_per_env * runner.env.num_envs / (collection_time + learn_time))
                
                # Console logging (same as original)
                print("=" * 80)
                print(f"Learning iteration {it}/{num_learning_iterations}")
                print("")
                print(f"Computation: {fps} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)")
                
                if len(rewbuffer) > 0:
                    print(f"Value function loss: {mean_value_loss:.4f}")
                    print(f"Surrogate loss: {mean_surrogate_loss:.4f}")
                    print(f"Mean action noise std: {mean_std.item():.2f}")
                    print(f"Mean total reward: {np.mean(rewbuffer):.2f}")
                    print(f"Mean episode length: {np.mean(lenbuffer):.2f}")
                
                # Environment-specific metrics
                extras = runner.env.extras
                if "episode" in extras:
                    for key, value in extras["episode"].items():
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            print(f"{key}: {value:.4f}")
                
                print("-" * 80)
                print(f"Total timesteps: {tot_timesteps}")
                print(f"Iteration time: {collection_time + learn_time:.2f}s")
                print(f"Total time: {tot_time:.2f}s")
                if it > 0:
                    print(f"ETA: {(num_learning_iterations - it) * (tot_time / it):.1f}s")
                print("")
                
                # WandB logging
                try:
                    wandb_metrics = {
                        'policy/value_loss': mean_value_loss,
                        'policy/surrogate_loss': mean_surrogate_loss, 
                        'policy/entropy': mean_entropy,
                        'training/mean_std': mean_std.item(),
                        'performance/fps': fps,
                        'timing/collection_time': collection_time,
                        'timing/learning_time': learn_time,
                        'training/total_timesteps': tot_timesteps,
                    }
                    
                    if len(rewbuffer) > 0:
                        wandb_metrics.update({
                            'performance/mean_reward': np.mean(rewbuffer),
                            'performance/mean_episode_length': np.mean(lenbuffer),
                        })
                    
                    # Add environment metrics
                    if "episode" in extras:
                        for key, value in extras["episode"].items():
                            if isinstance(value, (int, float, np.integer, np.floating)):
                                if key.startswith('rew_'):
                                    # Reward components
                                    clean_key = key.replace('rew_', '')
                                    wandb_metrics[f"rewards/{clean_key}"] = value
                                else:
                                    wandb_metrics[f"episode/{key}"] = value
                    
                    # Log to WandB
                    wandb.log(wandb_metrics, step=it)
                    
                except Exception as e:
                    print(f"⚠️ WandB logging error: {e}")

            # Model saving (same as original) - with safe writer check
            if runner.log_dir is not None and it % runner.save_interval == 0:
                model_path = os.path.join(runner.log_dir, 'model_{}.pt'.format(it))
                try:
                    if runner.writer is not None:
                        runner.save(model_path)
                    else:
                        # Direct model saving without writer
                        torch.save({
                            'model_state_dict': runner.alg.actor_critic.state_dict(),
                            'optimizer_state_dict': runner.alg.optimizer.state_dict(),
                            'iter': it
                        }, model_path)
                        print(f"💾 Model saved to {model_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save model: {e}")
        
        # Final save - with safe writer check
        if runner.log_dir is not None:
            final_model_path = os.path.join(runner.log_dir, 'model_{}.pt'.format(runner.current_learning_iteration))
            try:
                if runner.writer is not None:
                    runner.save(final_model_path)
                else:
                    # Direct model saving without writer
                    torch.save({
                        'model_state_dict': runner.alg.actor_critic.state_dict(),
                        'optimizer_state_dict': runner.alg.optimizer.state_dict(),
                        'iter': runner.current_learning_iteration
                    }, final_model_path)
                    print(f"💾 Final model saved to {final_model_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save final model: {e}")
        
        return tot_timesteps, tot_time
    
    # Replace the learn method on the OnPolicyRunner class
    OnPolicyRunner.learn = enhanced_learn
    print("✅ OnPolicyRunner patched for WandB logging")

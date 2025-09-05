"""
Environment wrapper for rsl-rl 3.0+ compatibility.

This wrapper ensures that the BipedEnv works correctly with the new rsl-rl version
that expects TensorDict observations and specific reset/step behavior.
"""

import torch
from tensordict import TensorDict
from biped_env_main import BipedEnv


class RSLRLCompatibleWrapper:
    """Wrapper to make BipedEnv compatible with rsl-rl 3.0+"""
    
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.env = BipedEnv(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer)
        self.num_envs = num_envs
        self.num_actions = self.env.num_actions
        self.device = self.env.device
        
        # Initialize and get the first observations to understand the structure
        obs, extras = self.env.reset()
        self._current_obs = obs
        self._current_extras = extras
    
    def reset(self):
        """Reset environment and return only observations as TensorDict"""
        obs, extras = self.env.reset()
        self._current_obs = obs
        self._current_extras = extras
        return obs
    
    def step(self, actions):
        """Step environment and return (obs, rewards, dones, extras)"""
        obs, rewards, dones, extras = self.env.step(actions)
        self._current_obs = obs
        self._current_extras = extras
        return obs, rewards, dones, extras
    
    def get_observations(self):
        """Get current observations as TensorDict"""
        return self._current_obs
    
    def close(self):
        """Close the environment"""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def __getattr__(self, name):
        """Delegate any other attribute access to the wrapped environment"""
        return getattr(self.env, name)

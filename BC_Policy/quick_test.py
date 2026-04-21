"""
Quick test script for BC policy - runs a few episodes and shows results.

Usage:
    python quick_test.py
"""

import numpy as np
import torch
import torch.nn as nn
from pick_egg import PickEgg, DamageablePickEgg
from robosuite.controllers import load_controller_config


class BCPolicy(nn.Module):
    """Simple MLP policy - adjust if your architecture differs."""
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.network(obs)
    
    def get_action(self, obs):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs)
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            return self.forward(obs).cpu().numpy().squeeze()


def quick_test(policy_path="best_policy.pt", num_episodes=5, render=True):
    """Run a quick test of the policy."""
    
    print("Creating environment...")
    
    # Load default controller config from robosuite
    controller_config = load_controller_config(default_controller="OSC_POSE")
    
    env = PickEgg(
        robots="PandaMobile",
        controller_configs=controller_config,
        has_renderer=render,
        reward_shaping=True,
        horizon=500,
    )
    
    # Get dimensions
    obs = env.reset()
    obs_dim = obs.shape[0]
    action_dim = env.action_spec[0].shape[0]
    
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Loading policy from {policy_path}...")
    
    # Load policy
    policy = BCPolicy(obs_dim, action_dim)
    checkpoint = torch.load(policy_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint.get('policy') or checkpoint
    else:
        state_dict = checkpoint
    
    policy.load_state_dict(state_dict)
    policy.eval()
    
    print(f"\nRunning {num_episodes} test episodes...\n")
    
    successes = 0
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        
        while not done and step < 500:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            step += 1
            
            if render:
                env.render()
        
        success = env._check_success()
        successes += success
        
        print(f"Episode {ep+1}: {'SUCCESS' if success else 'FAILED'} ({step} steps)")
    
    print(f"\nSuccess Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    env.close()


if __name__ == "__main__":
    quick_test()
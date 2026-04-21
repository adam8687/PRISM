"""
Evaluate a trained BC policy on the PickEgg environment.

Usage:
    python evaluate_policy.py --policy best_policy.pt --num_episodes 100
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm
from robosuite.controllers import load_controller_config

# Import the custom environment
from pick_egg_env import PickEgg, DamageablePickEgg


# ═══════════════════════════════════════════════════════════════════════
# Policy Architecture
# ═══════════════════════════════════════════════════════════════════════

class BCPolicy(nn.Module):
    """
    Simple MLP policy for behavioral cloning.
    Adjust architecture if your policy differs.
    """
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs):
        return self.network(obs)
    
    def get_action(self, obs, deterministic=True):
        """Get action from observation."""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs)
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            
            action = self.forward(obs)
            return action.cpu().numpy().squeeze()


# ═══════════════════════════════════════════════════════════════════════
# Evaluation Function
# ═══════════════════════════════════════════════════════════════════════

def evaluate_policy(
    policy,
    env,
    num_episodes=100,
    max_steps=500,
    render=False,
    save_videos=False,
    video_dir="evaluation_videos",
    device="cpu"
):
    """
    Evaluate policy over multiple episodes.
    
    Args:
        policy: The trained policy
        env: The environment instance
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render during evaluation
        save_videos: Whether to save videos of episodes
        video_dir: Directory to save videos
        device: Device to run policy on
    
    Returns:
        Dictionary with evaluation metrics
    """
    policy.to(device)
    policy.eval()
    
    if save_videos:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
    
    # Metrics storage
    metrics = defaultdict(list)
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        step = 0
        episode_reward = 0
        
        # Episode-specific tracking
        if hasattr(env, 'get_damage_info'):
            episode_damage = 0
        
        while not done and step < max_steps:
            # Get action from policy
            action = policy.get_action(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Track damage if using DamageablePickEgg
            if hasattr(env, 'get_damage_info'):
                damage_info = env.get_damage_info()
                if damage_info:
                    episode_damage += damage_info.get('total_damage', 0)
            
            if render:
                env.render()
        
        # Check success
        success = info.get('success', env._check_success())
        
        # Store metrics
        metrics['success'].append(float(success))
        metrics['episode_length'].append(step)
        metrics['episode_reward'].append(episode_reward)
        
        if hasattr(env, 'get_damage_info'):
            metrics['total_damage'].append(episode_damage)
    
    # Compute summary statistics
    summary = {
        'num_episodes': num_episodes,
        'success_rate': np.mean(metrics['success']),
        'avg_episode_length': np.mean(metrics['episode_length']),
        'std_episode_length': np.std(metrics['episode_length']),
        'avg_reward': np.mean(metrics['episode_reward']),
        'std_reward': np.std(metrics['episode_reward']),
    }
    
    if 'total_damage' in metrics:
        summary['avg_damage'] = np.mean(metrics['total_damage'])
        summary['std_damage'] = np.std(metrics['total_damage'])
    
    return summary, metrics


# ═══════════════════════════════════════════════════════════════════════
# Main Evaluation Script
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy on PickEgg")
    parser.add_argument("--policy", type=str, default="best_policy.pt",
                        help="Path to policy checkpoint")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--use_damage", action="store_true",
                        help="Use DamageablePickEgg environment")
    parser.add_argument("--render", action="store_true",
                        help="Render during evaluation")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save videos of episodes")
    parser.add_argument("--video_dir", type=str, default="evaluation_videos",
                        help="Directory to save videos")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Path to save evaluation results")
    
    # Environment configuration
    parser.add_argument("--robots", type=str, default="PandaMobile",
                        help="Robot type")
    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        help="Controller type")
    parser.add_argument("--camera", type=str, default="robot0_agentview_center",
                        help="Camera name for rendering")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("BC Policy Evaluation on PickEgg Environment")
    print("=" * 70)
    
    # Create environment
    print(f"\n[1/4] Creating environment (damage tracking: {args.use_damage})...")
    env_class = DamageablePickEgg if args.use_damage else PickEgg
    
    # Load default controller config from robosuite
    controller_config = load_controller_config(default_controller=args.controller)
    
    env_config = {
        "robots": args.robots,
        "controller_configs": controller_config,
        "has_renderer": args.render,
        "has_offscreen_renderer": args.save_videos,
        "camera_names": args.camera,
        "reward_shaping": True,
        "horizon": args.max_steps,
    }
    
    env = env_class(**env_config)
    
    # Get observation and action dimensions from environment
    obs = env.reset()
    obs_dim = obs.shape[0] if isinstance(obs, np.ndarray) else len(obs)
    action_dim = env.action_spec[0].shape[0]
    
    print(f"   - Observation dim: {obs_dim}")
    print(f"   - Action dim: {action_dim}")
    
    # Load policy
    print(f"\n[2/4] Loading policy from {args.policy}...")
    checkpoint = torch.load(args.policy, map_location=args.device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'policy' in checkpoint:
            state_dict = checkpoint['policy']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Try to infer architecture from state dict
    try:
        policy = BCPolicy(obs_dim, action_dim)
        policy.load_state_dict(state_dict)
        print("   ✓ Policy loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading policy with default architecture: {e}")
        print("   Please adjust the BCPolicy architecture to match your trained model")
        return
    
    # Run evaluation
    print(f"\n[3/4] Running evaluation ({args.num_episodes} episodes)...")
    summary, metrics = evaluate_policy(
        policy=policy,
        env=env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_videos=args.save_videos,
        video_dir=args.video_dir,
        device=args.device
    )
    
    # Print results
    print("\n[4/4] Evaluation Results:")
    print("=" * 70)
    print(f"Success Rate:        {summary['success_rate']:.2%}")
    print(f"Avg Episode Length:  {summary['avg_episode_length']:.1f} ± {summary['std_episode_length']:.1f}")
    print(f"Avg Reward:          {summary['avg_reward']:.3f} ± {summary['std_reward']:.3f}")
    
    if 'avg_damage' in summary:
        print(f"Avg Damage:          {summary['avg_damage']:.3f} ± {summary['std_damage']:.3f}")
    
    print("=" * 70)
    
    # Save results
    results = {
        'summary': summary,
        'raw_metrics': {k: [float(v) for v in vals] for k, vals in metrics.items()},
        'config': vars(args)
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    env.close()


if __name__ == "__main__":
    main()
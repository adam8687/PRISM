"""
Visualize evaluation results from the BC policy.

Usage:
    python visualize_results.py --results evaluation_results.json
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_evaluation_results(results_path, output_dir="plots"):
    """Create visualization plots from evaluation results."""
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    summary = results['summary']
    metrics = results['raw_metrics']
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Success Rate Bar
    ax1 = plt.subplot(2, 3, 1)
    success_rate = summary['success_rate']
    ax1.bar(['Success Rate'], [success_rate * 100], color='green' if success_rate > 0.8 else 'orange')
    ax1.set_ylim([0, 100])
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title(f'Success Rate: {success_rate:.2%}')
    ax1.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.legend()
    
    # 2. Episode Length Distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(metrics['episode_length'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(summary['avg_episode_length'], color='red', linestyle='--', 
                label=f'Mean: {summary["avg_episode_length"]:.1f}')
    ax2.set_xlabel('Episode Length (steps)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Episode Length Distribution')
    ax2.legend()
    
    # 3. Reward Distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(metrics['episode_reward'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    ax3.axvline(summary['avg_reward'], color='red', linestyle='--',
                label=f'Mean: {summary["avg_reward"]:.3f}')
    ax3.set_xlabel('Episode Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reward Distribution')
    ax3.legend()
    
    # 4. Success vs Episode Length
    ax4 = plt.subplot(2, 3, 4)
    success_array = np.array(metrics['success'])
    lengths_array = np.array(metrics['episode_length'])
    
    successful = lengths_array[success_array == 1]
    failed = lengths_array[success_array == 0]
    
    if len(successful) > 0:
        ax4.hist(successful, bins=20, alpha=0.7, label='Successful', color='green')
    if len(failed) > 0:
        ax4.hist(failed, bins=20, alpha=0.7, label='Failed', color='red')
    
    ax4.set_xlabel('Episode Length (steps)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Episode Length by Outcome')
    ax4.legend()
    
    # 5. Success Over Episodes (rolling average)
    ax5 = plt.subplot(2, 3, 5)
    window = min(10, len(metrics['success']) // 10)
    if window > 1:
        rolling_success = np.convolve(metrics['success'], 
                                     np.ones(window)/window, mode='valid')
        ax5.plot(rolling_success * 100, linewidth=2)
        ax5.fill_between(range(len(rolling_success)), 0, rolling_success * 100, 
                        alpha=0.3)
    else:
        ax5.plot([s * 100 for s in metrics['success']], linewidth=2)
    
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title(f'Success Rate Over Time (window={window})')
    ax5.set_ylim([0, 100])
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)
    
    # 6. Damage Distribution (if available)
    ax6 = plt.subplot(2, 3, 6)
    if 'total_damage' in metrics:
        ax6.hist(metrics['total_damage'], bins=30, color='orange', 
                edgecolor='black', alpha=0.7)
        ax6.axvline(summary['avg_damage'], color='red', linestyle='--',
                   label=f'Mean: {summary["avg_damage"]:.3f}')
        ax6.set_xlabel('Total Damage')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Damage Distribution')
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No damage tracking\navailable', 
                ha='center', va='center', fontsize=12)
        ax6.set_title('Damage Tracking (N/A)')
        ax6.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'evaluation_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    
    # Create summary statistics figure
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    summary_text = f"""
    EVALUATION SUMMARY
    {'='*50}
    
    Number of Episodes:     {summary['num_episodes']}
    Success Rate:          {summary['success_rate']:.2%}
    
    Episode Length:
        Mean:              {summary['avg_episode_length']:.1f} steps
        Std Dev:           {summary['std_episode_length']:.1f} steps
        Min:               {min(metrics['episode_length'])} steps
        Max:               {max(metrics['episode_length'])} steps
    
    Reward:
        Mean:              {summary['avg_reward']:.3f}
        Std Dev:           {summary['std_reward']:.3f}
        Min:               {min(metrics['episode_reward']):.3f}
        Max:               {max(metrics['episode_reward']):.3f}
    """
    
    if 'avg_damage' in summary:
        summary_text += f"""
    Damage:
        Mean:              {summary['avg_damage']:.3f}
        Std Dev:           {summary['std_damage']:.3f}
        Min:               {min(metrics['total_damage']):.3f}
        Max:               {max(metrics['total_damage']):.3f}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    output_path2 = Path(output_dir) / 'evaluation_stats.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved stats to {output_path2}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results", type=str, default="evaluation_results.json",
                        help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.results}...")
    plot_evaluation_results(args.results, args.output_dir)
    print("✓ Visualization complete!")


if __name__ == "__main__":
    main()
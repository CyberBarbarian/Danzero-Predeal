#!/usr/bin/env python3
"""
Analyze training results and generate performance reports
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

def analyze_training_stats(stats_file):
    """Analyze training statistics and generate plots."""
    
    print("="*80)
    print("📊 DanZero Training Results Analysis")
    print("="*80)
    print()
    
    # Load stats
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    iterations = stats['iterations']
    if not iterations:
        print("❌ No training data found!")
        return
    
    # Extract metrics
    iter_nums = [i['iteration'] for i in iterations]
    iter_times = [i['duration'] for i in iterations]
    rewards = [i['episode_reward_mean'] for i in iterations]
    episode_lengths = [i['episode_len_mean'] for i in iterations]
    timesteps = [i['timesteps_total'] for i in iterations]
    
    # Calculate statistics
    total_iters = len(iterations)
    total_duration = stats.get('total_duration', 0)
    avg_iter_time = np.mean(iter_times)
    best_reward = max(rewards)
    final_reward = rewards[-1]
    
    # Print summary
    print(f"Experiment: {stats['experiment_name']}")
    print(f"Start time: {stats['start_time']}")
    if 'end_time' in stats:
        print(f"End time: {stats['end_time']}")
    print()
    
    print("Configuration:")
    config = stats['config']
    print(f"  Workers: {config['num_workers']}")
    print(f"  Envs per worker: {config['num_envs_per_worker']}")
    print(f"  Total parallel envs: {config['total_envs']}")
    print(f"  GPUs: {config['num_gpus']}")
    print()
    
    print("Training Performance:")
    print(f"  Total iterations: {total_iters}")
    print(f"  Total duration: {total_duration / 3600:.2f} hours")
    print(f"  Average iteration time: {avg_iter_time:.2f}s")
    print(f"  Throughput: {1/avg_iter_time:.2f} iter/s")
    print(f"  Total timesteps: {timesteps[-1]:,}")
    print()
    
    print("Reward Statistics:")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Final reward: {final_reward:.2f}")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Std reward: {np.std(rewards):.2f}")
    print()
    
    print("Episode Statistics:")
    print(f"  Mean episode length: {np.mean(episode_lengths):.1f}")
    print(f"  Final episode length: {episode_lengths[-1]:.1f}")
    print()
    
    # Generate plots
    output_dir = Path(stats_file).parent
    print(f"📈 Generating plots in {output_dir}...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'DanZero Training Results - {stats["experiment_name"]}', fontsize=16)
    
    # Plot 1: Reward over iterations
    ax1 = axes[0, 0]
    ax1.plot(iter_nums, rewards, alpha=0.6, linewidth=0.5, label='Raw')
    # Smooth with moving average
    window = min(50, len(rewards) // 10)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smooth_iters = iter_nums[window-1:]
        ax1.plot(smooth_iters, smoothed, linewidth=2, label=f'Smoothed (window={window})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Training Reward Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode length over iterations
    ax2 = axes[0, 1]
    ax2.plot(iter_nums, episode_lengths, alpha=0.6, linewidth=0.5, label='Raw')
    if window > 1:
        smoothed_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(smooth_iters, smoothed_len, linewidth=2, label=f'Smoothed (window={window})')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mean Episode Length')
    ax2.set_title('Episode Length Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Iteration time over iterations
    ax3 = axes[1, 0]
    ax3.plot(iter_nums, iter_times, alpha=0.6, linewidth=0.5, label='Raw')
    if window > 1:
        smoothed_time = np.convolve(iter_times, np.ones(window)/window, mode='valid')
        ax3.plot(smooth_iters, smoothed_time, linewidth=2, label=f'Smoothed (window={window})')
    ax3.axhline(y=avg_iter_time, color='r', linestyle='--', label=f'Mean: {avg_iter_time:.2f}s')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Iteration Time (s)')
    ax3.set_title('Training Speed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Timesteps over iterations
    ax4 = axes[1, 1]
    ax4.plot(iter_nums, np.array(timesteps) / 1e6, linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Total Timesteps (millions)')
    ax4.set_title('Data Collection Progress')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / 'training_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {plot_file}")
    
    # Create reward distribution plot
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(rewards), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax.axvline(x=best_reward, color='g', linestyle='--', linewidth=2, label=f'Best: {best_reward:.2f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    dist_file = output_dir / 'reward_distribution.png'
    plt.savefig(dist_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {dist_file}")
    
    print()
    print("="*80)
    print("✅ Analysis complete!")
    print("="*80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <training_stats.json>")
        sys.exit(1)
    
    stats_file = sys.argv[1]
    if not Path(stats_file).exists():
        print(f"❌ File not found: {stats_file}")
        sys.exit(1)
    
    analyze_training_stats(stats_file)


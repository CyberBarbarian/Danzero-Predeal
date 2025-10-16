#!/usr/bin/env python3
"""
Visualize monitoring data from train_with_monitoring.sh output.

Usage:
    python visualize_monitoring.py <output_directory>
    
Example:
    python visualize_monitoring.py testing/outputs/monitoring_20251009_193045
"""

import sys
import json
import os
from pathlib import Path
import numpy as np

def load_cpu_data(cpu_log):
    """Load CPU utilization data."""
    timestamps = []
    cpu_usage = []
    load_1m = []
    load_5m = []
    load_15m = []
    
    with open(cpu_log, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 5:
            timestamps.append(float(parts[0]))
            cpu_usage.append(float(parts[1]))
            load_1m.append(float(parts[2]))
            load_5m.append(float(parts[3]))
            load_15m.append(float(parts[4]))
    
    # Convert timestamps to relative seconds
    if timestamps:
        start_time = timestamps[0]
        timestamps = [t - start_time for t in timestamps]
    
    return {
        'timestamps': timestamps,
        'cpu_usage': cpu_usage,
        'load_1m': load_1m,
        'load_5m': load_5m,
        'load_15m': load_15m,
    }

def load_gpu_data(gpu_log):
    """Load GPU utilization data."""
    gpu0_data = {'timestamps': [], 'util': [], 'mem_util': [], 'mem_used': [], 'temp': [], 'power': []}
    gpu1_data = {'timestamps': [], 'util': [], 'mem_util': [], 'mem_used': [], 'temp': [], 'power': []}
    
    with open(gpu_log, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    start_time = None
    
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            timestamp = float(parts[0])
            if start_time is None:
                start_time = timestamp
            
            gpu_id = int(parts[1])
            gpu_util = float(parts[2])
            mem_util = float(parts[3])
            mem_used = float(parts[4])
            mem_total = float(parts[5])
            temp = float(parts[6])
            power = float(parts[7])
            
            rel_time = timestamp - start_time
            
            if gpu_id == 0:
                gpu0_data['timestamps'].append(rel_time)
                gpu0_data['util'].append(gpu_util)
                gpu0_data['mem_util'].append(mem_util)
                gpu0_data['mem_used'].append(mem_used)
                gpu0_data['temp'].append(temp)
                gpu0_data['power'].append(power)
            elif gpu_id == 1:
                gpu1_data['timestamps'].append(rel_time)
                gpu1_data['util'].append(gpu_util)
                gpu1_data['mem_util'].append(mem_util)
                gpu1_data['mem_used'].append(mem_used)
                gpu1_data['temp'].append(temp)
                gpu1_data['power'].append(power)
    
    return gpu0_data, gpu1_data

def load_training_data(metrics_file):
    """Load training metrics."""
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    return data

def print_statistics(output_dir):
    """Print summary statistics."""
    cpu_log = os.path.join(output_dir, 'cpu_utilization.log')
    gpu_log = os.path.join(output_dir, 'gpu_utilization.log')
    metrics_file = os.path.join(output_dir, 'training_metrics.json')
    
    print("="*80)
    print("📊 MONITORING DATA STATISTICS")
    print("="*80)
    print()
    
    # CPU statistics
    if os.path.exists(cpu_log):
        cpu_data = load_cpu_data(cpu_log)
        
        print("CPU Utilization:")
        print(f"  Samples: {len(cpu_data['cpu_usage'])}")
        print(f"  Duration: {cpu_data['timestamps'][-1] if cpu_data['timestamps'] else 0:.1f}s")
        
        if cpu_data['cpu_usage']:
            print(f"  Mean: {np.mean(cpu_data['cpu_usage']):.1f}%")
            print(f"  Max: {np.max(cpu_data['cpu_usage']):.1f}%")
            print(f"  Min: {np.min(cpu_data['cpu_usage']):.1f}%")
            print(f"  Std: {np.std(cpu_data['cpu_usage']):.1f}%")
        print()
    
    # GPU statistics
    if os.path.exists(gpu_log):
        gpu0_data, gpu1_data = load_gpu_data(gpu_log)
        
        print("GPU 0 Utilization:")
        print(f"  Samples: {len(gpu0_data['util'])}")
        if gpu0_data['util']:
            print(f"  Mean: {np.mean(gpu0_data['util']):.1f}%")
            print(f"  Max: {np.max(gpu0_data['util']):.1f}%")
            print(f"  Min: {np.min(gpu0_data['util']):.1f}%")
            print(f"  Std: {np.std(gpu0_data['util']):.1f}%")
            print(f"  Avg Memory Used: {np.mean(gpu0_data['mem_used']):.1f} MB")
            print(f"  Avg Temperature: {np.mean(gpu0_data['temp']):.1f}°C")
            print(f"  Avg Power: {np.mean(gpu0_data['power']):.1f}W")
        print()
        
        print("GPU 1 Utilization:")
        print(f"  Samples: {len(gpu1_data['util'])}")
        if gpu1_data['util']:
            print(f"  Mean: {np.mean(gpu1_data['util']):.1f}%")
            print(f"  Max: {np.max(gpu1_data['util']):.1f}%")
            print(f"  Min: {np.min(gpu1_data['util']):.1f}%")
            print(f"  Std: {np.std(gpu1_data['util']):.1f}%")
            print(f"  Avg Memory Used: {np.mean(gpu1_data['mem_used']):.1f} MB")
            print(f"  Avg Temperature: {np.mean(gpu1_data['temp']):.1f}°C")
            print(f"  Avg Power: {np.mean(gpu1_data['power']):.1f}W")
        print()
    
    # Training statistics
    if os.path.exists(metrics_file):
        training_data = load_training_data(metrics_file)
        iterations = training_data.get('iterations', [])
        
        if iterations:
            iter_times = [it['duration'] for it in iterations]
            
            print("Training Performance:")
            print(f"  Iterations: {len(iterations)}")
            print(f"  Avg iteration time: {np.mean(iter_times):.2f}s")
            print(f"  Min iteration time: {np.min(iter_times):.2f}s")
            print(f"  Max iteration time: {np.max(iter_times):.2f}s")
            print(f"  Total time: {training_data.get('total_train_time', 0):.1f}s")
            print(f"  Throughput: {len(iterations) / training_data.get('total_train_time', 1):.3f} iter/s")
            print()

def generate_plots(output_dir):
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not available, skipping plot generation")
        return
    
    cpu_log = os.path.join(output_dir, 'cpu_utilization.log')
    gpu_log = os.path.join(output_dir, 'gpu_utilization.log')
    metrics_file = os.path.join(output_dir, 'training_metrics.json')
    
    # Load data
    cpu_data = load_cpu_data(cpu_log) if os.path.exists(cpu_log) else None
    gpu0_data, gpu1_data = load_gpu_data(gpu_log) if os.path.exists(gpu_log) else (None, None)
    training_data = load_training_data(metrics_file) if os.path.exists(metrics_file) else None
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('DanZero Training Monitoring - 300 Iterations', fontsize=16)
    
    # CPU Utilization
    ax = axes[0, 0]
    if cpu_data and cpu_data['cpu_usage']:
        ax.plot(cpu_data['timestamps'], cpu_data['cpu_usage'], 'b-', linewidth=1, alpha=0.7)
        mean_cpu = np.mean(cpu_data['cpu_usage'])
        ax.axhline(y=mean_cpu, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_cpu:.1f}%')
        ax.legend()
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('CPU Utilization (%)')
    ax.set_title('CPU Utilization Over Time')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # GPU 0 Utilization
    ax = axes[0, 1]
    if gpu0_data and gpu0_data['util']:
        ax.plot(gpu0_data['timestamps'], gpu0_data['util'], 'g-', linewidth=1, alpha=0.7)
        mean_gpu = np.mean(gpu0_data['util'])
        ax.axhline(y=mean_gpu, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_gpu:.1f}%')
        ax.legend()
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU 0 Compute Utilization')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # GPU 1 Utilization
    ax = axes[0, 2]
    if gpu1_data and gpu1_data['util']:
        ax.plot(gpu1_data['timestamps'], gpu1_data['util'], 'g-', linewidth=1, alpha=0.7)
        mean_gpu = np.mean(gpu1_data['util'])
        ax.axhline(y=mean_gpu, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_gpu:.1f}%')
        ax.legend()
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU 1 Compute Utilization')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # GPU 0 Memory
    ax = axes[1, 0]
    if gpu0_data and gpu0_data['mem_used']:
        ax.plot(gpu0_data['timestamps'], [m/1024 for m in gpu0_data['mem_used']], 'purple', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('GPU 0 Memory Usage')
    ax.grid(True, alpha=0.3)
    
    # GPU 1 Memory
    ax = axes[1, 1]
    if gpu1_data and gpu1_data['mem_used']:
        ax.plot(gpu1_data['timestamps'], [m/1024 for m in gpu1_data['mem_used']], 'purple', linewidth=1, alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('GPU 1 Memory Usage')
    ax.grid(True, alpha=0.3)
    
    # Temperature
    ax = axes[1, 2]
    if gpu0_data and gpu0_data['temp']:
        ax.plot(gpu0_data['timestamps'], gpu0_data['temp'], 'orange', linewidth=1, alpha=0.7, label='GPU 0')
    if gpu1_data and gpu1_data['temp']:
        ax.plot(gpu1_data['timestamps'], gpu1_data['temp'], 'red', linewidth=1, alpha=0.7, label='GPU 1')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('GPU Temperature')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Iteration Time
    ax = axes[2, 0]
    if training_data and training_data.get('iterations'):
        iterations = [it['iteration'] for it in training_data['iterations']]
        iter_times = [it['duration'] for it in training_data['iterations']]
        ax.plot(iterations, iter_times, 'b-', linewidth=1, marker='o', markersize=2, alpha=0.7)
        mean_time = np.mean(iter_times)
        ax.axhline(y=mean_time, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.2f}s')
        ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Iteration Time')
    ax.grid(True, alpha=0.3)
    
    # Episodes Progress
    ax = axes[2, 1]
    if training_data and training_data.get('iterations'):
        iterations = [it['iteration'] for it in training_data['iterations']]
        episodes = [it['num_episodes'] for it in training_data['iterations']]
        ax.plot(iterations, episodes, 'cyan', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Episodes')
    ax.set_title('Episode Completion Progress')
    ax.grid(True, alpha=0.3)
    
    # Reward Progress
    ax = axes[2, 2]
    if training_data and training_data.get('iterations'):
        iterations = [it['iteration'] for it in training_data['iterations']]
        rewards = [it['episode_reward_mean'] for it in training_data['iterations']]
        ax.plot(iterations, rewards, 'red', linewidth=2, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Reward Progression')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'monitoring_results.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Plots saved to: {plot_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_monitoring.py <output_directory>")
        print()
        print("Example:")
        print("  python visualize_monitoring.py testing/outputs/monitoring_20251009_193045")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"❌ Directory not found: {output_dir}")
        sys.exit(1)
    
    print("="*80)
    print("📊 ANALYZING MONITORING DATA")
    print("="*80)
    print(f"Directory: {output_dir}")
    print()
    
    # Print statistics
    print_statistics(output_dir)
    
    # Generate plots
    print("="*80)
    print("📈 Generating visualization plots...")
    print("="*80)
    generate_plots(output_dir)
    
    print()
    print("="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved in: {output_dir}")
    print()

if __name__ == '__main__':
    main()


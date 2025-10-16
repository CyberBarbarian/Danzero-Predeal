#!/usr/bin/env python3
"""
CPU-optimized configuration test with detailed resource profiling.
Records CPU and GPU utilization curves for analysis.

This script:
- Runs training for configurable iterations (default: 20)
- Records CPU utilization every second
- Records GPU utilization every second
- Saves detailed metrics to JSON and generates plots
- Output: testing/outputs/profiling_[timestamp]/
"""

import ray
import time
import sys
import os
import json
import threading
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from guandan.rllib.trainers import create_dmc_trainer


class ResourceMonitor:
    """Monitor CPU and GPU utilization in background thread."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitoring = False
        self.monitor_thread = None
        
        self.cpu_samples = []
        self.gpu_samples = []
        self.timestamps = []
        
        self.start_time = None
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            timestamp = time.time() - self.start_time
            
            # Get CPU utilization
            try:
                cpu_result = subprocess.run(
                    ['top', '-bn1'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                # Parse CPU usage from top output
                for line in cpu_result.stdout.split('\n'):
                    if 'Cpu(s)' in line:
                        # Extract idle percentage and calculate usage
                        parts = line.split(',')
                        for part in parts:
                            if 'id' in part:  # idle
                                idle = float(part.split()[0])
                                cpu_usage = 100.0 - idle
                                break
                        else:
                            cpu_usage = 0.0
                        break
                else:
                    cpu_usage = 0.0
            except Exception as e:
                print(f"Warning: CPU monitoring error: {e}")
                cpu_usage = 0.0
            
            # Get GPU utilization
            try:
                gpu_result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                gpu_metrics = []
                for line in gpu_result.stdout.strip().split('\n'):
                    if line:
                        parts = [float(x.strip()) for x in line.split(',')]
                        gpu_metrics.append({
                            'gpu_util': parts[0],
                            'mem_util': parts[1],
                            'mem_used': parts[2],
                            'mem_total': parts[3]
                        })
                
            except Exception as e:
                print(f"Warning: GPU monitoring error: {e}")
                gpu_metrics = []
            
            # Record samples
            self.timestamps.append(timestamp)
            self.cpu_samples.append(cpu_usage)
            self.gpu_samples.append(gpu_metrics)
            
            # Sleep for 1 second
            time.sleep(1.0)
    
    def start(self):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ Resource monitoring started")
    
    def stop(self):
        """Stop monitoring and save data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Save raw data
        data = {
            'timestamps': self.timestamps,
            'cpu_utilization': self.cpu_samples,
            'gpu_samples': self.gpu_samples,
            'duration': self.timestamps[-1] if self.timestamps else 0,
        }
        
        metrics_file = self.output_dir / 'resource_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Resource metrics saved to: {metrics_file}")
        return data
    
    def generate_plots(self, training_metrics):
        """Generate visualization plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle('DanZero Training Profiling - CPU Optimized Configuration', fontsize=16)
            
            # CPU Utilization
            ax = axes[0, 0]
            ax.plot(self.timestamps, self.cpu_samples, 'b-', linewidth=2)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('CPU Utilization (%)')
            ax.set_title('CPU Utilization Over Time')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # Add statistics
            if self.cpu_samples:
                mean_cpu = np.mean(self.cpu_samples)
                max_cpu = np.max(self.cpu_samples)
                ax.axhline(y=mean_cpu, color='r', linestyle='--', label=f'Mean: {mean_cpu:.1f}%')
                ax.legend()
            
            # GPU Utilization (GPU 0)
            ax = axes[0, 1]
            if self.gpu_samples and len(self.gpu_samples) > 0:
                gpu0_util = [s[0]['gpu_util'] if len(s) > 0 else 0 for s in self.gpu_samples]
                ax.plot(self.timestamps, gpu0_util, 'g-', linewidth=2)
                
                if gpu0_util:
                    mean_gpu = np.mean(gpu0_util)
                    ax.axhline(y=mean_gpu, color='r', linestyle='--', label=f'Mean: {mean_gpu:.1f}%')
                    ax.legend()
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('GPU Utilization (%)')
            ax.set_title('GPU 0 Utilization Over Time')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            # GPU Memory Usage
            ax = axes[1, 0]
            if self.gpu_samples and len(self.gpu_samples) > 0:
                gpu0_mem = [s[0]['mem_used']/1024 if len(s) > 0 else 0 for s in self.gpu_samples]
                gpu0_total = self.gpu_samples[0][0]['mem_total']/1024 if len(self.gpu_samples) > 0 and len(self.gpu_samples[0]) > 0 else 80
                
                ax.plot(self.timestamps, gpu0_mem, 'purple', linewidth=2)
                ax.axhline(y=gpu0_total, color='r', linestyle='--', label=f'Total: {gpu0_total:.0f} GB')
                ax.legend()
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('GPU Memory (GB)')
            ax.set_title('GPU 0 Memory Usage')
            ax.grid(True, alpha=0.3)
            
            # Training Iteration Time
            ax = axes[1, 1]
            if training_metrics and 'iterations' in training_metrics:
                iterations = [m['iteration'] for m in training_metrics['iterations']]
                iter_times = [m['duration'] for m in training_metrics['iterations']]
                
                ax.plot(iterations, iter_times, 'orange', linewidth=2, marker='o')
                
                if iter_times:
                    mean_time = np.mean(iter_times)
                    ax.axhline(y=mean_time, color='r', linestyle='--', label=f'Mean: {mean_time:.2f}s')
                    ax.legend()
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Iteration Time')
            ax.grid(True, alpha=0.3)
            
            # Episodes Progress
            ax = axes[2, 0]
            if training_metrics and 'iterations' in training_metrics:
                iterations = [m['iteration'] for m in training_metrics['iterations']]
                episodes = [m['num_episodes'] for m in training_metrics['iterations']]
                
                ax.plot(iterations, episodes, 'cyan', linewidth=2, marker='o')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Total Episodes')
            ax.set_title('Episode Completion Progress')
            ax.grid(True, alpha=0.3)
            
            # Reward Progress
            ax = axes[2, 1]
            if training_metrics and 'iterations' in training_metrics:
                iterations = [m['iteration'] for m in training_metrics['iterations']]
                rewards = [m['episode_reward_mean'] for m in training_metrics['iterations']]
                
                ax.plot(iterations, rewards, 'red', linewidth=2, marker='o')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean Episode Reward')
            ax.set_title('Reward Progression')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_file = self.output_dir / 'profiling_results.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Plots saved to: {plot_file}")
            
        except ImportError:
            print("⚠️  matplotlib not available, skipping plot generation")
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")


def main():
    # Configuration
    NUM_TEST_ITERATIONS = 20  # Default: 20 iterations
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent.parent / 'outputs' / f'profiling_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🧪 TRAINING PROFILING TEST - CPU & GPU Monitoring")
    print("="*80)
    print()
    print(f"📁 Output directory: {output_dir}")
    print(f"🔢 Test iterations: {NUM_TEST_ITERATIONS}")
    print()
    
    # Initialize Ray
    print("🔧 Initializing Ray...")
    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        num_gpus=2,
        num_cpus=192,
        object_store_memory=30 * 1024 * 1024 * 1024,  # 30 GB
    )
    
    print(f"✅ Ray initialized")
    print(f"   CPUs: 192")
    print(f"   GPUs: 2")
    print()
    
    # Create trainer
    print("="*80)
    print("🎯 Creating DMC Algorithm (CPU-optimized)...")
    print("="*80)
    print()
    print("Configuration:")
    print("  Workers: 150")
    print("  Envs per worker: 4")
    print("  Total environments: 600")
    print("  GPU allocation: 1.0 for learner, 0.0 per worker")
    print()
    
    init_start = time.time()
    
    algo = create_dmc_trainer(
        env_config={
            "observation_mode": "comprehensive",
            "use_internal_adapters": False,
            "max_steps": 1000,  # Safe limit - games naturally complete around 700 steps
        },
        num_workers=150,
        num_envs_per_worker=4,
    num_gpus=1.0,
    num_gpus_per_worker=0.0,
    lr=1e-3,
    batch_size=600,  # Match 600 parallel environments for proper worker utilization
    epsilon_start=0.2,
    epsilon_end=0.05,
    epsilon_decay_steps=10000,
    )
    
    init_time = time.time() - init_start
    print(f"✅ Algorithm initialized in {init_time:.2f}s")
    print()
    
    # Start resource monitoring
    monitor = ResourceMonitor(output_dir)
    monitor.start()
    
    # Training metrics
    training_data = {
        'config': {
            'num_workers': 150,
            'num_envs_per_worker': 4,
            'total_envs': 600,
            'num_gpus': 1.0,
            'num_test_iterations': NUM_TEST_ITERATIONS,
        },
        'init_time': init_time,
        'iterations': []
    }
    
    print("="*80)
    print(f"🏃 Running {NUM_TEST_ITERATIONS} Training Iterations with Profiling...")
    print("="*80)
    print()
    
    try:
        train_start = time.time()
        
        for iteration in range(1, NUM_TEST_ITERATIONS + 1):
            iter_start = time.time()
            
            result = algo.train()
            
            iter_time = time.time() - iter_start
            
            # Extract metrics
            env_runners = result.get('env_runners', {})
            episodes_this_iter = env_runners.get('episodes_this_iter', 0)
            episode_reward_mean = env_runners.get('episode_reward_mean', 0.0)
            episode_len_mean = env_runners.get('episode_len_mean', 0.0)
            num_episodes = env_runners.get('num_episodes', 0)
            timesteps_total = result.get('num_env_steps_sampled', 0)
            timesteps_this_iter = result.get('num_env_steps_sampled_this_iter', 0)
            
            # Handle NaN
            import math
            if math.isnan(episode_reward_mean):
                episode_reward_mean = 0.0
            if math.isnan(episode_len_mean):
                episode_len_mean = 0.0
            
            # Store metrics
            iter_data = {
                'iteration': iteration,
                'duration': iter_time,
                'episodes_this_iter': episodes_this_iter,
                'num_episodes': num_episodes,
                'timesteps_total': timesteps_total,
                'timesteps_this_iter': timesteps_this_iter,
                'episode_reward_mean': episode_reward_mean,
                'episode_len_mean': episode_len_mean,
            }
            training_data['iterations'].append(iter_data)
            
            # Progress output
            print(f"[Iter {iteration:3d}/{NUM_TEST_ITERATIONS}] "
                  f"{iter_time:5.2f}s | "
                  f"Episodes: {num_episodes:4d} (+{episodes_this_iter:2d}) | "
                  f"Steps: {timesteps_total:9d} | "
                  f"Reward: {episode_reward_mean:7.2f}")
            
            # Detailed output every 5 iterations
            if iteration % 5 == 0:
                elapsed = time.time() - train_start
                print(f"    Elapsed: {elapsed:.1f}s | "
                      f"Avg iter time: {elapsed/iteration:.2f}s | "
                      f"Episode length: {episode_len_mean:.1f}")
        
        total_train_time = time.time() - train_start
        training_data['total_train_time'] = total_train_time
        
        print()
        print("="*80)
        print("✅ TRAINING TEST COMPLETE")
        print("="*80)
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Training interrupted by user")
    except Exception as e:
        print()
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop monitoring
        print()
        print("🛑 Stopping resource monitoring...")
        resource_data = monitor.stop()
        
        # Save training metrics
        metrics_file = output_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"✅ Training metrics saved to: {metrics_file}")
        
        # Generate plots
        print()
        print("📊 Generating analysis plots...")
        monitor.generate_plots(training_data)
        
        # Generate summary report
        print()
        print("="*80)
        print("📊 PROFILING SUMMARY")
        print("="*80)
        print()
        
        if resource_data['cpu_utilization']:
            cpu_mean = np.mean(resource_data['cpu_utilization'])
            cpu_max = np.max(resource_data['cpu_utilization'])
            cpu_min = np.min(resource_data['cpu_utilization'])
            print(f"CPU Utilization:")
            print(f"  Mean: {cpu_mean:.1f}%")
            print(f"  Max:  {cpu_max:.1f}%")
            print(f"  Min:  {cpu_min:.1f}%")
            print()
        
        if resource_data['gpu_samples']:
            gpu0_utils = [s[0]['gpu_util'] if len(s) > 0 else 0 for s in resource_data['gpu_samples']]
            if gpu0_utils:
                gpu_mean = np.mean(gpu0_utils)
                gpu_max = np.max(gpu0_utils)
                gpu_min = np.min(gpu0_utils)
                print(f"GPU 0 Utilization:")
                print(f"  Mean: {gpu_mean:.1f}%")
                print(f"  Max:  {gpu_max:.1f}%")
                print(f"  Min:  {gpu_min:.1f}%")
                print()
        
        if training_data['iterations']:
            iter_times = [m['duration'] for m in training_data['iterations']]
            print(f"Training Performance:")
            print(f"  Iterations: {len(iter_times)}")
            print(f"  Avg iteration time: {np.mean(iter_times):.2f}s")
            print(f"  Total training time: {training_data.get('total_train_time', 0):.1f}s")
            print(f"  Throughput: {len(iter_times)/training_data.get('total_train_time', 1):.3f} iter/s")
            print()
        
        print("="*80)
        print(f"📁 All results saved to: {output_dir}")
        print("="*80)
        print()
        print("Files generated:")
        print(f"  • {output_dir}/resource_metrics.json - Raw CPU/GPU data")
        print(f"  • {output_dir}/training_metrics.json - Training iteration data")
        print(f"  • {output_dir}/profiling_results.png - Visualization plots")
        print()
        
        # Cleanup
        algo.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()


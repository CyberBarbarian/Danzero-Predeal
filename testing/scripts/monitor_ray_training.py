#!/usr/bin/env python3
"""
Simple training monitor that reads from Ray's automatic logging.
Monitors training progress by tailing Ray's result.json file.
"""

import os
import sys
import time
import json
import psutil
import subprocess
import glob
from datetime import datetime
from pathlib import Path

def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_info.append({
                        'utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'temperature': int(parts[3])
                    })
            return gpu_info
    except Exception as e:
        return []
    return []

def get_system_metrics():
    """Get system resource utilization"""
    try:
        cpu_util = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_util,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'gpu_info': get_gpu_utilization()
        }
    except Exception as e:
        print(f"Warning: Could not get system metrics: {e}")
        return {}

def find_latest_ray_result_dir():
    """Find the most recent Ray results directory"""
    ray_results_dir = os.path.expanduser("~/ray_results")
    if not os.path.exists(ray_results_dir):
        return None
    
    # Look for DMC training directories
    pattern = os.path.join(ray_results_dir, "DMC_guandan_ma_*")
    dirs = glob.glob(pattern)
    
    if not dirs:
        return None
    
    # Get the most recent one
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir

def read_ray_results(result_file):
    """Read and parse Ray's result.json file"""
    results = []
    try:
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: Error reading results: {e}")
    
    return results

def monitor_training(ray_result_dir, output_dir, check_interval=10):
    """Monitor training by reading Ray's result.json file"""
    
    print("=" * 80)
    print("📊 Ray Training Monitor")
    print("=" * 80)
    print(f"📁 Monitoring: {ray_result_dir}")
    print(f"💾 Saving results to: {output_dir}")
    print(f"⏱️  Check interval: {check_interval}s")
    print("=" * 80)
    print()
    
    result_file = os.path.join(ray_result_dir, "result.json")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    last_training_iteration = -1
    start_time = time.time()
    all_metrics = []
    
    try:
        while True:
            # Read all results from the file
            results = read_ray_results(result_file)
            
            if results:
                latest_result = results[-1]
                training_iteration = latest_result.get('training_iteration', 0)
                
                # Check if we have new data
                if training_iteration > last_training_iteration:
                    last_training_iteration = training_iteration
                    
                    # Get system metrics
                    sys_metrics = get_system_metrics()
                    
                    # Extract key metrics
                    timesteps_total = latest_result.get('num_env_steps_sampled_lifetime', 0)
                    episodes_total = latest_result.get('num_episodes_lifetime', 0)
                    time_total_s = latest_result.get('time_total_s', 0)
                    
                    # Try to get learner metrics if available
                    learner_results = latest_result.get('learner_results', {})
                    loss = None
                    if isinstance(learner_results, dict):
                        # Try to get loss from any agent
                        for key in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
                            if key in learner_results:
                                agent_data = learner_results[key]
                                if isinstance(agent_data, dict) and 'loss' in agent_data:
                                    loss_val = agent_data['loss']
                                    # Handle both direct values and nested structures
                                    if isinstance(loss_val, dict):
                                        loss = loss_val.get('value', loss_val.get('mean', 0))
                                    else:
                                        loss = float(loss_val) if loss_val is not None else None
                                    break
                    
                    # Print progress
                    print(f"\n📈 Iteration {training_iteration} ({time.time() - start_time:.1f}s elapsed)")
                    print(f"  Steps: {timesteps_total:,} | Episodes: {episodes_total:,}")
                    if loss is not None:
                        print(f"  Loss: {loss:.6f}")
                    print(f"  Time: {time_total_s:.1f}s")
                    
                    # Print system metrics
                    if sys_metrics:
                        print(f"  💻 CPU: {sys_metrics.get('cpu_percent', 0):.1f}%")
                        print(f"  🧠 Memory: {sys_metrics.get('memory_percent', 0):.1f}% "
                              f"({sys_metrics.get('memory_used_gb', 0):.1f}GB/{sys_metrics.get('memory_total_gb', 0):.1f}GB)")
                        
                        gpu_info = sys_metrics.get('gpu_info', [])
                        if gpu_info:
                            for i, gpu in enumerate(gpu_info):
                                print(f"  🎮 GPU {i}: {gpu['utilization']}% | "
                                      f"Mem: {gpu['memory_used']}MB/{gpu['memory_total']}MB | "
                                      f"Temp: {gpu['temperature']}°C")
                    
                    # Store metrics
                    metrics_entry = {
                        'iteration': training_iteration,
                        'timestamp': datetime.now().isoformat(),
                        'timesteps_total': timesteps_total,
                        'episodes_total': episodes_total,
                        'time_total_s': time_total_s,
                        'loss': loss,
                        'system_metrics': sys_metrics
                    }
                    all_metrics.append(metrics_entry)
                    
                    # Save metrics periodically (every 5 iterations)
                    if training_iteration % 5 == 0:
                        metrics_file = os.path.join(output_dir, f"metrics_iter_{training_iteration}.json")
                        with open(metrics_file, 'w') as f:
                            json.dump(all_metrics, f, indent=2)
                        print(f"  💾 Metrics saved to {metrics_file}")
            
            # Check if training is still running
            if not os.path.exists(result_file):
                print("\n⚠️  Result file disappeared. Training may have ended.")
                break
            
            # Wait before next check
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
    
    # Save final metrics
    if all_metrics:
        final_metrics_file = os.path.join(output_dir, "final_metrics.json")
        with open(final_metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\n📊 Final metrics saved to {final_metrics_file}")
        print(f"📈 Total iterations monitored: {last_training_iteration}")
        
        if all_metrics:
            last = all_metrics[-1]
            print(f"📊 Final stats:")
            print(f"  Steps: {last.get('timesteps_total', 0):,}")
            print(f"  Episodes: {last.get('episodes_total', 0):,}")
            if last.get('loss') is not None:
                print(f"  Loss: {last.get('loss'):.6f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Ray RLlib training")
    parser.add_argument('--ray-dir', type=str, help='Path to Ray results directory (auto-detected if not provided)')
    parser.add_argument('--output-dir', type=str, default='./monitoring_results', help='Output directory for metrics')
    parser.add_argument('--interval', type=int, default=10, help='Check interval in seconds')
    
    args = parser.parse_args()
    
    # Find Ray results directory
    if args.ray_dir:
        ray_dir = args.ray_dir
    else:
        print("🔍 Searching for latest Ray training run...")
        ray_dir = find_latest_ray_result_dir()
        if not ray_dir:
            print("❌ No Ray training directories found in ~/ray_results/")
            print("   Start your training first, then run this monitor.")
            sys.exit(1)
        print(f"✅ Found: {ray_dir}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Start monitoring
    monitor_training(ray_dir, output_dir, args.interval)

if __name__ == "__main__":
    main()



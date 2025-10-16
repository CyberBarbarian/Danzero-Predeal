#!/bin/bash

################################################################################
# DanZero Training with Resource Monitoring
# Runs 300 iterations with continuous CPU and GPU utilization tracking
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🧪 DanZero Training Test - 300 Iterations with Monitoring"
echo "================================================================================"
echo ""

# --- Configuration ---
NUM_ITERATIONS=300
NUM_WORKERS=150
NUM_ENVS_PER_WORKER=4
MONITORING_INTERVAL=1  # Monitor every 1 second

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./testing/outputs/monitoring_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/training.log"
CPU_LOG="${OUTPUT_DIR}/cpu_utilization.log"
GPU_LOG="${OUTPUT_DIR}/gpu_utilization.log"
METRICS_FILE="${OUTPUT_DIR}/training_metrics.json"

echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📁 Training log: ${LOG_FILE}"
echo "📁 CPU log: ${CPU_LOG}"
echo "📁 GPU log: ${GPU_LOG}"
echo "📁 Metrics: ${METRICS_FILE}"
echo ""

# --- Activate virtual environment ---
echo "🔧 Activating virtual environment..."
source /mnt/project_modelware/lizikang/Danvenv/bin/activate

# --- Check GPU availability ---
echo ""
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# --- H100 Optimizations ---
echo "⚡ Enabling H100 optimizations..."
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NVIDIA_TF32_OVERRIDE=1

echo "  ✅ GPU optimizations enabled"
echo ""

# --- Start monitoring processes ---
echo "📊 Starting resource monitoring..."

# CPU monitoring function
monitor_cpu() {
    local output_file=$1
    echo "timestamp,cpu_usage_percent,load_avg_1m,load_avg_5m,load_avg_15m" > "${output_file}"
    
    while true; do
        timestamp=$(date +%s)
        
        # Get CPU usage from top
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        
        # Get load averages
        load_avg=$(cat /proc/loadavg | awk '{print $1","$2","$3}')
        
        echo "${timestamp},${cpu_usage},${load_avg}" >> "${output_file}"
        
        sleep ${MONITORING_INTERVAL}
    done
}

# GPU monitoring function
monitor_gpu() {
    local output_file=$1
    # Header
    echo "timestamp,gpu_id,gpu_util,mem_util,mem_used_mb,mem_total_mb,temp_c,power_w" > "${output_file}"
    
    while true; do
        timestamp=$(date +%s)
        
        # Query both GPUs
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
                   --format=csv,noheader,nounits | while read line; do
            echo "${timestamp},${line}" >> "${output_file}"
        done
        
        sleep ${MONITORING_INTERVAL}
    done
}

# Start monitoring in background
monitor_cpu "${CPU_LOG}" &
CPU_MONITOR_PID=$!
echo "  ✅ CPU monitoring started (PID: ${CPU_MONITOR_PID})"

monitor_gpu "${GPU_LOG}" &
GPU_MONITOR_PID=$!
echo "  ✅ GPU monitoring started (PID: ${GPU_MONITOR_PID})"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping monitoring processes..."
    kill ${CPU_MONITOR_PID} 2>/dev/null || true
    kill ${GPU_MONITOR_PID} 2>/dev/null || true
    echo "✅ Cleanup complete"
    
    # Generate summary
    echo ""
    echo "📊 Generating summary report..."
    python3 << 'PYTHON_SUMMARY'
import json
import os
from datetime import datetime

output_dir = os.environ.get('OUTPUT_DIR')
cpu_log = f"{output_dir}/cpu_utilization.log"
gpu_log = f"{output_dir}/gpu_utilization.log"
metrics_file = f"{output_dir}/training_metrics.json"

print("="*80)
print("📊 MONITORING SUMMARY")
print("="*80)
print()

# CPU statistics
try:
    with open(cpu_log, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    cpu_values = [float(line.split(',')[1]) for line in lines if len(line.split(',')) > 1]
    
    if cpu_values:
        print(f"CPU Utilization:")
        print(f"  Samples: {len(cpu_values)}")
        print(f"  Mean: {sum(cpu_values)/len(cpu_values):.1f}%")
        print(f"  Max: {max(cpu_values):.1f}%")
        print(f"  Min: {min(cpu_values):.1f}%")
        print()
except Exception as e:
    print(f"⚠️  Could not parse CPU log: {e}")
    print()

# GPU statistics
try:
    with open(gpu_log, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    gpu0_values = []
    gpu1_values = []
    
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            gpu_id = int(parts[1])
            gpu_util = float(parts[2])
            if gpu_id == 0:
                gpu0_values.append(gpu_util)
            elif gpu_id == 1:
                gpu1_values.append(gpu_util)
    
    if gpu0_values:
        print(f"GPU 0 Utilization:")
        print(f"  Samples: {len(gpu0_values)}")
        print(f"  Mean: {sum(gpu0_values)/len(gpu0_values):.1f}%")
        print(f"  Max: {max(gpu0_values):.1f}%")
        print(f"  Min: {min(gpu0_values):.1f}%")
        print()
    
    if gpu1_values:
        print(f"GPU 1 Utilization:")
        print(f"  Samples: {len(gpu1_values)}")
        print(f"  Mean: {sum(gpu1_values)/len(gpu1_values):.1f}%")
        print(f"  Max: {max(gpu1_values):.1f}%")
        print(f"  Min: {min(gpu1_values):.1f}%")
        print()
except Exception as e:
    print(f"⚠️  Could not parse GPU log: {e}")
    print()

# Training metrics
try:
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        iterations = data.get('iterations', [])
        if iterations:
            iter_times = [it['duration'] for it in iterations]
            print(f"Training Performance:")
            print(f"  Iterations completed: {len(iterations)}")
            print(f"  Avg iteration time: {sum(iter_times)/len(iter_times):.2f}s")
            print(f"  Total training time: {data.get('total_train_time', 0):.1f}s")
            print()
except Exception as e:
    print(f"⚠️  Could not parse training metrics: {e}")
    print()

print("="*80)
print(f"📁 All data saved to: {output_dir}")
print("="*80)
PYTHON_SUMMARY
}
trap cleanup EXIT INT TERM

echo ""
echo "================================================================================"
echo "🎯 Training Configuration"
echo "================================================================================"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Workers: ${NUM_WORKERS}"
echo "Environments per worker: ${NUM_ENVS_PER_WORKER}"
echo "Total parallel environments: $((NUM_WORKERS * NUM_ENVS_PER_WORKER))"
echo "Monitoring interval: ${MONITORING_INTERVAL} second(s)"
echo "================================================================================"
echo ""

# Export configuration for Python script
export OUTPUT_DIR
export NUM_ITERATIONS
export NUM_WORKERS
export NUM_ENVS_PER_WORKER

# --- Run training with monitoring ---
echo "🚀 Starting training..."
echo ""

python3 << 'PYTHON_TRAINING'
import ray
import time
import torch
import os
import json
import numpy as np
from datetime import datetime

from guandan.rllib.trainers import create_dmc_trainer

# Get configuration
output_dir = os.environ.get('OUTPUT_DIR')
num_iterations = int(os.environ.get('NUM_ITERATIONS', 300))
num_workers = int(os.environ.get('NUM_WORKERS', 150))
num_envs_per_worker = int(os.environ.get('NUM_ENVS_PER_WORKER', 4))

print("="*80)
print("🔧 Initializing Ray...")
print("="*80)

# Initialize Ray
ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    num_gpus=2,
    num_cpus=192,
    object_store_memory=30 * 1024 * 1024 * 1024,
)

print(f"✅ Ray initialized")
print(f"   CPUs: 192")
print(f"   GPUs: 2")
print()

# Enable H100 optimizations
if torch.cuda.is_available():
    print("⚡ Enabling H100 Tensor Core optimizations...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("   ✅ TF32 enabled")
    print("   ✅ cuDNN benchmarking enabled")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU{i}: {props.name} ({props.total_memory / (1024**3):.0f}GB)")
print()

print("="*80)
print("🎯 Creating DMC Algorithm...")
print("="*80)

start_time = time.time()

# Create trainer
algo = create_dmc_trainer(
    env_config={
        "observation_mode": "comprehensive",
        "use_internal_adapters": False,
        "max_steps": 1200,  # Safe limit - games naturally complete around 700 steps, buffer for long games
    },
    num_workers=num_workers,
    num_envs_per_worker=num_envs_per_worker,
    num_gpus=1.0,
    num_gpus_per_worker=0.0,
    lr=1e-3,
    batch_size=600,  # Match 600 parallel environments for proper worker utilization
    epsilon_start=0.2,
    epsilon_end=0.05,
    epsilon_decay_steps=10000,
)

init_time = time.time() - start_time
print(f"✅ Algorithm initialized in {init_time:.2f}s")
print()

# Training metrics
training_data = {
    'config': {
        'num_workers': num_workers,
        'num_envs_per_worker': num_envs_per_worker,
        'total_envs': num_workers * num_envs_per_worker,
        'num_iterations': num_iterations,
    },
    'start_time': datetime.now().isoformat(),
    'init_time': init_time,
    'iterations': []
}

print("="*80)
print(f"🏃 Training Loop - {num_iterations} iterations")
print("="*80)
print()

iteration_times = []

try:
    train_start = time.time()
    
    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()
        
        # Training step
        result = algo.train()
        
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
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
            'timestamp': time.time(),
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
        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.time() - train_start
            recent_times = iteration_times[-min(100, len(iteration_times)):]
            avg_iter_time = np.mean(recent_times)
            remaining = (num_iterations - iteration) * avg_iter_time
            eta = remaining / 3600
            
            print(f"[Iter {iteration:4d}/{num_iterations}] "
                  f"{iter_time:5.2f}s | "
                  f"Episodes: {num_episodes:5d} (+{episodes_this_iter:2d}) | "
                  f"Steps: {timesteps_total:10d} | "
                  f"Reward: {episode_reward_mean:7.2f} | "
                  f"ETA: {eta:.2f}h")
        
        # Detailed progress every 50 iterations
        if iteration % 50 == 0:
            recent_times = iteration_times[-min(100, len(iteration_times)):]
            avg_iter_time = np.mean(recent_times)
            elapsed = time.time() - train_start
            
            print()
            print(f"{'='*80}")
            print(f"📊 Progress Update - Iteration {iteration}/{num_iterations}")
            print(f"{'='*80}")
            print(f"  Time elapsed: {elapsed / 3600:.2f}h")
            print(f"  Avg iteration time: {avg_iter_time:.2f}s")
            print(f"  Total episodes: {num_episodes}")
            print(f"  Total timesteps: {timesteps_total}")
            print(f"  Mean reward: {episode_reward_mean:.2f}")
            print(f"  Progress: {iteration/num_iterations*100:.1f}%")
            print(f"{'='*80}")
            print()
        
        # Save metrics periodically
        if iteration % 50 == 0:
            metrics_file = os.path.join(output_dir, 'training_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(training_data, f, indent=2)
    
    total_train_time = time.time() - train_start
    training_data['total_train_time'] = total_train_time
    training_data['end_time'] = datetime.now().isoformat()
    
    print()
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"  Total iterations: {num_iterations}")
    print(f"  Total time: {total_train_time / 3600:.2f}h")
    print(f"  Avg iteration time: {np.mean(iteration_times):.2f}s")
    print(f"  Throughput: {num_iterations / total_train_time:.3f} iter/s")
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
    # Save final metrics
    metrics_file = os.path.join(output_dir, 'training_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print()
    print(f"✅ Training metrics saved to: {metrics_file}")
    
    # Cleanup
    algo.stop()
    ray.shutdown()

PYTHON_TRAINING

echo ""
echo "================================================================================"
echo "✅ Training and Monitoring Complete!"
echo "================================================================================"
echo ""
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  • ${CPU_LOG} - CPU utilization time series"
echo "  • ${GPU_LOG} - GPU utilization time series"
echo "  • ${METRICS_FILE} - Training performance metrics"
echo "  • ${LOG_FILE} - Full training log"
echo ""
echo "📊 To visualize the results:"
echo "   python testing/scripts/visualize_monitoring.py ${OUTPUT_DIR}"
echo ""
echo "================================================================================"


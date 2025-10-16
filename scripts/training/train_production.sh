#!/bin/bash

################################################################################
# DanZero Production Training Script
# Optimized for 2× H100 80GB GPUs + 200 CPU cores
# Configuration: 200 workers, 200 parallel environments (restored)
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "🚀 DanZero Production Training - 2× H100 Configuration"
echo "================================================================================"
echo ""

# --- Configuration ---
EXPERIMENT_NAME="danzero_production_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="./logs/${EXPERIMENT_NAME}"
RESULTS_DIR="./results/${EXPERIMENT_NAME}"

# Training hyperparameters
# NUM_WORKERS=200              # Restored to 200 workers
NUM_WORKERS=100
NUM_ENVS_PER_WORKER=1        # 200 total environments (1 per worker)
NUM_GPUS=1.0                 # 1 GPU for learner
NUM_GPUS_PER_WORKER=0.0      # CPU-only workers for scalability (set >0 to force GPU inference)
# TOTAL_ITERATIONS=2000        # 1000 iterations for training
TOTAL_ITERATIONS=12
CHECKPOINT_FREQ=3          # Save every 100 iterations
EVALUATION_FREQ=10          # Evaluate every 500 iterations

# GPU role configuration
# By default (auto), if >=2 GPUs and NUM_GPUS_PER_WORKER==0, reserve 1 GPU for inference and others for learner.
# For single-GPU testing, default is to use ALL GPUs for learner (no inference split).
USE_INFERENCE_GPU="auto"     # Options: "auto" | "true" | "false"

# Ray configuration
RAY_NUM_CPUS=200             # Restored to 200 CPU cores
RAY_NUM_GPUS=2               # Both H100s
RAY_OBJECT_STORE_MEMORY=$((30 * 1024 * 1024 * 1024))  # 30 GB

# Create directories
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "📁 Experiment: ${EXPERIMENT_NAME}"
echo "📁 Checkpoints: ${CHECKPOINT_DIR}"
echo "📁 Logs: ${LOG_DIR}"
echo "📁 Results: ${RESULTS_DIR}"
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

# TF32 precision for H100 Tensor Cores
export NVIDIA_TF32_OVERRIDE=1

echo "  ✅ TF32 Tensor Core optimization enabled"
echo "  ✅ cuDNN v8 API enabled"
echo "  ✅ CUDA memory allocation optimized"
echo ""

# --- System resource monitoring setup ---
echo "📊 Starting resource monitoring..."

# Start nvidia-smi monitoring in background
nvidia-smi dmon -s pucvmet -d 10 -o TD > "${LOG_DIR}/gpu_monitor.log" 2>&1 &
GPU_MONITOR_PID=$!
echo "  ✅ GPU monitoring started (PID: ${GPU_MONITOR_PID})"

# Start CPU/memory monitoring
top -b -d 10 | grep --line-buffered "Cpu\|Mem" > "${LOG_DIR}/cpu_monitor.log" 2>&1 &
CPU_MONITOR_PID=$!
echo "  ✅ CPU monitoring started (PID: ${CPU_MONITOR_PID})"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Cleaning up monitoring processes..."
    kill ${GPU_MONITOR_PID} 2>/dev/null || true
    kill ${CPU_MONITOR_PID} 2>/dev/null || true
    echo "✅ Cleanup complete"
}
trap cleanup EXIT INT TERM

echo ""
echo "================================================================================"
echo "🎯 Training Configuration"
echo "================================================================================"
echo "Workers: ${NUM_WORKERS}"
echo "Environments per worker: ${NUM_ENVS_PER_WORKER}"
echo "Total parallel environments: $((NUM_WORKERS * NUM_ENVS_PER_WORKER))"
echo "GPU allocation: ${NUM_GPUS} for learner, ${NUM_GPUS_PER_WORKER} per worker"
echo ""
echo "Training iterations: ${TOTAL_ITERATIONS}"
echo "Checkpoint frequency: ${CHECKPOINT_FREQ} iterations"
echo "Evaluation frequency: ${EVALUATION_FREQ} iterations"
echo ""
echo "Model: 1.3M parameters (paper specification)"
echo "  • Architecture: 5 layers × 512 hidden units"
echo "  • Activation: Tanh"
echo "  • Learning rate: 1e-3"
echo "  • Batch size: 64"
echo "================================================================================"
echo ""

# --- Export configuration for Python script ---
export EXPERIMENT_NAME
export CHECKPOINT_DIR
export LOG_DIR
export RESULTS_DIR
export NUM_WORKERS
export NUM_ENVS_PER_WORKER
export NUM_GPUS
export NUM_GPUS_PER_WORKER
export TOTAL_ITERATIONS
export CHECKPOINT_FREQ
export EVALUATION_FREQ
export RAY_NUM_CPUS
export RAY_NUM_GPUS
export RAY_OBJECT_STORE_MEMORY
export USE_INFERENCE_GPU

# --- Training script ---
echo "🚀 Starting training..."
echo ""

python3 << 'PYTHON_SCRIPT'
import ray
import time
import torch
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from guandan.rllib.trainers import create_dmc_trainer
from guandan.training.logger import Logger

# Get configuration from environment
experiment_name = os.environ.get('EXPERIMENT_NAME')
checkpoint_dir = os.environ.get('CHECKPOINT_DIR')
log_dir = os.environ.get('LOG_DIR')
results_dir = os.environ.get('RESULTS_DIR')

# Make checkpoint_dir absolute to avoid pyarrow URI issues
checkpoint_dir = os.path.abspath(checkpoint_dir)
results_dir = os.path.abspath(results_dir)

num_workers = int(os.environ.get('NUM_WORKERS', 120))
num_envs_per_worker = int(os.environ.get('NUM_ENVS_PER_WORKER', 3))
num_gpus = float(os.environ.get('NUM_GPUS', 1.0))
num_gpus_per_worker = float(os.environ.get('NUM_GPUS_PER_WORKER', 0.0))
use_inference_gpu_env = os.environ.get('USE_INFERENCE_GPU', 'auto').lower()
if use_inference_gpu_env in ('true', '1', 'yes'):
    use_inference_gpu = True
elif use_inference_gpu_env in ('false', '0', 'no'):
    use_inference_gpu = False
else:
    use_inference_gpu = None  # auto

total_iterations = int(os.environ.get('TOTAL_ITERATIONS', 10000))
checkpoint_freq = int(os.environ.get('CHECKPOINT_FREQ', 100))
evaluation_freq = int(os.environ.get('EVALUATION_FREQ', 500))

ray_num_cpus = int(os.environ.get('RAY_NUM_CPUS', 192))
ray_num_gpus = int(os.environ.get('RAY_NUM_GPUS', 2))
ray_object_store = int(os.environ.get('RAY_OBJECT_STORE_MEMORY'))

print("="*80)
print("🔧 Initializing Ray...")
print("="*80)

# Initialize Ray
ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    num_gpus=ray_num_gpus,
    num_cpus=ray_num_cpus,
    object_store_memory=ray_object_store,
    _system_config={
        "max_direct_call_object_size": 100 * 1024 * 1024,  # 100 MB
        "task_retry_delay_ms": 1000,
    }
)

print(f"✅ Ray initialized")
print(f"   CPUs: {ray_num_cpus}")
print(f"   GPUs: {ray_num_gpus}")
print(f"   Object store: {ray_object_store / (1024**3):.1f} GB")
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

# Create trainer with optimal configuration
algo = create_dmc_trainer(
    env_config={
        "observation_mode": "comprehensive",
        "use_internal_adapters": False,
        "max_steps": 1500,  # Conservative limit - games naturally complete around 700 steps
    },
    num_workers=num_workers,
    num_envs_per_worker=num_envs_per_worker,
    num_gpus=num_gpus,
    num_gpus_per_worker=num_gpus_per_worker,
    use_inference_gpu=use_inference_gpu,
    lr=1e-3,
    batch_size=200,  # Match 200 parallel environments for proper worker utilization
    epsilon_start=0.2,
    epsilon_end=0.05,
    epsilon_decay_steps=10000,
)

init_time = time.time() - start_time
print(f"✅ Algorithm initialized in {init_time:.2f}s")
print()

# Training metrics and logger
training_stats = {
    'experiment_name': experiment_name,
    'start_time': datetime.now().isoformat(),
    'config': {
        'num_workers': num_workers,
        'num_envs_per_worker': num_envs_per_worker,
        'total_envs': num_workers * num_envs_per_worker,
        'num_gpus': num_gpus,
        'num_gpus_per_worker': num_gpus_per_worker,
    },
    'iterations': []
}
run_logger = Logger(log_dir=results_dir, xpid="")
run_logger.save_meta({'experiment': experiment_name, 'config': training_stats['config']})
# Bootstrap a scalar so TensorBoard shows immediately
run_logger.log({'bootstrap/started': 1})

print("="*80)
print(f"🏃 Training Loop - {total_iterations} iterations")
print("="*80)
print()

best_reward = float('-inf')
iteration_times = []

# Track previous values for calculating deltas
prev_episodes = 0
prev_timesteps = 0

# Track win counts for win rate calculation
total_team_1_wins = 0
total_team_2_wins = 0

try:
    for iteration in range(1, total_iterations + 1):
        iter_start = time.time()
        
        # Training step
        result = algo.train()
        
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
        # Extract metrics from correct locations
        # RLlib puts these in env_runners section
        env_runners = result.get('env_runners', {})
        episode_reward_mean = env_runners.get('episode_return_mean', 0.0)
        episode_len_mean = env_runners.get('episode_len_mean', 0.0)
        num_episodes = env_runners.get('num_episodes', 0)
        
        # Extract per-agent rewards (self-play tracking)
        agent_rewards = env_runners.get('agent_episode_returns_mean', {})
        agent_0_reward = agent_rewards.get('agent_0', 0.0)
        agent_1_reward = agent_rewards.get('agent_1', 0.0)
        agent_2_reward = agent_rewards.get('agent_2', 0.0)
        agent_3_reward = agent_rewards.get('agent_3', 0.0)
        
        # Calculate team rewards (zero-sum game: team1=agents 0&2, team2=agents 1&3)
        team_1_reward = agent_0_reward + agent_2_reward
        team_2_reward = agent_1_reward + agent_3_reward
        reward_sum = agent_0_reward + agent_1_reward + agent_2_reward + agent_3_reward
        
        # ===== MUST HAVE METRICS =====
        
        # 1. WIN RATE TRACKING
        # Count wins this iteration (positive reward = win)
        if team_1_reward > 0:
            total_team_1_wins += 1
        if team_2_reward > 0:
            total_team_2_wins += 1
        
        team_1_win_rate = total_team_1_wins / num_episodes if num_episodes > 0 else 0
        team_2_win_rate = total_team_2_wins / num_episodes if num_episodes > 0 else 0
        
        # 2. EPISODE STATISTICS (from env_runners)
        episode_len_min = env_runners.get('episode_len_min', 0)
        episode_len_max = env_runners.get('episode_len_max', 0)
        episode_duration_sec = env_runners.get('episode_duration_sec_mean', 0.0)
        
        # 3. LEARNING METRICS (from learners section)
        learners = result.get('learners', {})
        agent_0_learner = learners.get('agent_0', {})
        agent_1_learner = learners.get('agent_1', {})
        agent_2_learner = learners.get('agent_2', {})
        agent_3_learner = learners.get('agent_3', {})
        
        agent_0_q_mean = agent_0_learner.get('q_values_mean', 0.0)
        agent_0_loss = agent_0_learner.get('total_loss', 0.0)
        agent_0_grad_norm = agent_0_learner.get('gradients_adam_global_norm', 0.0)
        
        agent_1_q_mean = agent_1_learner.get('q_values_mean', 0.0)
        agent_1_loss = agent_1_learner.get('total_loss', 0.0)
        agent_1_grad_norm = agent_1_learner.get('gradients_adam_global_norm', 0.0)
        
        agent_2_q_mean = agent_2_learner.get('q_values_mean', 0.0)
        agent_2_loss = agent_2_learner.get('total_loss', 0.0)
        agent_2_grad_norm = agent_2_learner.get('gradients_adam_global_norm', 0.0)
        
        agent_3_q_mean = agent_3_learner.get('q_values_mean', 0.0)
        agent_3_loss = agent_3_learner.get('total_loss', 0.0)
        agent_3_grad_norm = agent_3_learner.get('gradients_adam_global_norm', 0.0)
        
        # Average across all agents for summary metrics
        avg_q_value = (agent_0_q_mean + agent_1_q_mean + agent_2_q_mean + agent_3_q_mean) / 4
        avg_loss = (agent_0_loss + agent_1_loss + agent_2_loss + agent_3_loss) / 4
        avg_grad_norm = (agent_0_grad_norm + agent_1_grad_norm + agent_2_grad_norm + agent_3_grad_norm) / 4
        
        # 4. SYMMETRY VERIFICATION
        team_performance_gap = abs(team_1_reward - team_2_reward)
        agent_reward_std = np.std([agent_0_reward, agent_1_reward, agent_2_reward, agent_3_reward])
        
        # Get timesteps from env_runners section
        timesteps_total = env_runners.get('num_env_steps_sampled', 0)
        
        # Calculate deltas
        episodes_this_iter = num_episodes - prev_episodes
        timesteps_this_iter = timesteps_total - prev_timesteps
        
        # 5. EFFICIENCY METRICS
        samples_per_second = timesteps_this_iter / iter_time if iter_time > 0 else 0
        episodes_per_hour = (episodes_this_iter / iter_time) * 3600 if iter_time > 0 else 0
        
        # Update previous values
        prev_episodes = num_episodes
        prev_timesteps = timesteps_total
        
        # Handle nan values
        import math
        if math.isnan(episode_reward_mean):
            episode_reward_mean = 0.0
        if math.isnan(episode_len_mean):
            episode_len_mean = 0.0
        
        # Store iteration stats (including self-play rewards + MUST HAVE metrics)
        iter_stats = {
            'iteration': iteration,
            'timestamp': time.time(),
            'duration': iter_time,
            'episodes_this_iter': episodes_this_iter,
            'num_episodes': num_episodes,
            'timesteps_total': timesteps_total,
            'timesteps_this_iter': timesteps_this_iter,
            'episode_reward_mean': episode_reward_mean,
            'episode_len_mean': episode_len_mean,
            'agent_0_reward': agent_0_reward,
            'agent_1_reward': agent_1_reward,
            'agent_2_reward': agent_2_reward,
            'agent_3_reward': agent_3_reward,
            'team_1_reward': team_1_reward,
            'team_2_reward': team_2_reward,
            'reward_sum': reward_sum,
            # MUST HAVE additions
            'team_1_win_rate': team_1_win_rate,
            'team_2_win_rate': team_2_win_rate,
            'episode_len_min': episode_len_min,
            'episode_len_max': episode_len_max,
            'episode_duration_sec': episode_duration_sec,
            'avg_q_value': avg_q_value,
            'avg_loss': avg_loss,
            'avg_grad_norm': avg_grad_norm,
            'team_performance_gap': team_performance_gap,
            'agent_reward_std': agent_reward_std,
            'samples_per_second': samples_per_second,
            'episodes_per_hour': episodes_per_hour,
        }
        training_stats['iterations'].append(iter_stats)
        # Log to TensorBoard (including self-play reward tracking + MUST HAVE metrics)
        run_logger.log({
            'iteration/duration_s': iter_time,
            'iteration/episodes_this_iter': episodes_this_iter,
            'progress/num_episodes': num_episodes,
            'progress/timesteps_total': timesteps_total,
            'reward/episode_reward_mean': episode_reward_mean,
            'episode/len_mean': episode_len_mean,
            # Per-agent rewards
            'rewards/agent_0': agent_0_reward,
            'rewards/agent_1': agent_1_reward,
            'rewards/agent_2': agent_2_reward,
            'rewards/agent_3': agent_3_reward,
            # Team rewards (zero-sum)
            'rewards/team_1': team_1_reward,
            'rewards/team_2': team_2_reward,
            # Verification (should be ~0 for zero-sum)
            'rewards/sum_all_agents': reward_sum,
            
            # ===== MUST HAVE METRICS =====
            # 1. Win rates
            'win_rates/team_1': team_1_win_rate,
            'win_rates/team_2': team_2_win_rate,
            # 2. Episode statistics
            'episode/len_min': episode_len_min,
            'episode/len_max': episode_len_max,
            'episode/duration_sec': episode_duration_sec,
            # 3. Learning metrics (per agent)
            'learning/agent_0_q_mean': agent_0_q_mean,
            'learning/agent_0_loss': agent_0_loss,
            'learning/agent_0_grad_norm': agent_0_grad_norm,
            'learning/agent_1_q_mean': agent_1_q_mean,
            'learning/agent_1_loss': agent_1_loss,
            'learning/agent_1_grad_norm': agent_1_grad_norm,
            'learning/agent_2_q_mean': agent_2_q_mean,
            'learning/agent_2_loss': agent_2_loss,
            'learning/agent_2_grad_norm': agent_2_grad_norm,
            'learning/agent_3_q_mean': agent_3_q_mean,
            'learning/agent_3_loss': agent_3_loss,
            'learning/agent_3_grad_norm': agent_3_grad_norm,
            # Learning summaries
            'learning/avg_q_value': avg_q_value,
            'learning/avg_loss': avg_loss,
            'learning/avg_grad_norm': avg_grad_norm,
            # 4. Symmetry verification
            'symmetry/team_performance_gap': team_performance_gap,
            'symmetry/agent_reward_std': agent_reward_std,
            # 5. Efficiency metrics
            'efficiency/samples_per_second': samples_per_second,
            'efficiency/episodes_per_hour': episodes_per_hour,
        })
        
        # Calculate running averages
        recent_times = iteration_times[-100:]  # Last 100 iterations
        avg_iter_time = np.mean(recent_times)
        
        # Progress reporting
        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            remaining = (total_iterations - iteration) * avg_iter_time
            eta = remaining / 3600  # hours
            
            print(f"[Iter {iteration:5d}/{total_iterations}] "
                  f"{iter_time:.2f}s | "
                  f"Eps: {int(num_episodes):4d} (+{int(episodes_this_iter):3d}) | "
                  f"T1: {team_1_reward:6.2f} T2: {team_2_reward:6.2f} | "
                  f"Steps: {int(timesteps_total):9d} | "
                  f"ETA: {eta:.1f}h")
        
        # Detailed progress every 50 iterations
        if iteration % 50 == 0:
            print()
            print(f"{'='*80}")
            print(f"📊 Progress Update - Iteration {iteration}/{total_iterations}")
            print(f"{'='*80}")
            print(f"  Time elapsed: {(time.time() - start_time) / 3600:.2f}h")
            print(f"  Avg iteration time: {avg_iter_time:.2f}s ({1/avg_iter_time:.2f} iter/s)")
            print(f"  Total episodes: {num_episodes}")
            print(f"  Episodes this iter: {episodes_this_iter}")
            print(f"  Total timesteps: {timesteps_total}")
            print(f"  Timesteps this iter: {timesteps_this_iter}")
            print(f"  Mean reward: {episode_reward_mean:.2f}")
            print(f"  Team 1 reward: {team_1_reward:.2f} | Team 2 reward: {team_2_reward:.2f}")
            print(f"  Agent rewards: [0]{agent_0_reward:.2f} [1]{agent_1_reward:.2f} [2]{agent_2_reward:.2f} [3]{agent_3_reward:.2f}")
            print(f"  Reward sum (zero-sum check): {reward_sum:.4f}")
            print(f"  Win rates: Team1={team_1_win_rate:.1%} Team2={team_2_win_rate:.1%}")
            print(f"  Symmetry: Gap={team_performance_gap:.3f} AgentStd={agent_reward_std:.3f}")
            print(f"  Learning: Q={avg_q_value:.3f} Loss={avg_loss:.4f} Grad={avg_grad_norm:.4f}")
            print(f"  Efficiency: {samples_per_second:.0f} samples/s, {episodes_per_hour:.0f} eps/h")
            print(f"  Mean episode length: {episode_len_mean:.1f}")
            print(f"  Estimated time remaining: {remaining / 3600:.1f}h")
            print(f"{'='*80}")
            print()
        
        # Checkpoint - Save each checkpoint separately with iteration number
        if iteration % checkpoint_freq == 0:
            # Create separate directory for this checkpoint
            iter_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{iteration:06d}")
            os.makedirs(iter_checkpoint_dir, exist_ok=True)
            checkpoint_path = algo.save(iter_checkpoint_dir)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Track best reward for monitoring (but don't save separate best model yet)
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                print(f"🏆 New best reward achieved: {episode_reward_mean:.2f} (at iteration {iteration})")
        
        # Evaluation
        if iteration % evaluation_freq == 0:
            print()
            print(f"{'='*80}")
            print(f"🎯 Evaluation at iteration {iteration}")
            print(f"{'='*80}")
            # Could add dedicated evaluation here if needed
            print(f"  Current mean reward: {episode_reward_mean:.2f}")
            print(f"  Best reward so far: {best_reward:.2f}")
            print(f"{'='*80}")
            print()
        
        # Save training stats periodically
        if iteration % 50 == 0:
            stats_file = os.path.join(results_dir, 'training_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(training_stats, f, indent=2)

except KeyboardInterrupt:
    print()
    print("="*80)
    print("⚠️  Training interrupted by user")
    print("="*80)
except Exception as e:
    print()
    print("="*80)
    print(f"❌ Training error: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
finally:
    # Final save - Save final checkpoint separately
    print()
    print("="*80)
    print("💾 Saving final checkpoint...")
    final_iter = len(training_stats['iterations'])
    final_checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint_{final_iter:06d}_final")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    final_checkpoint = algo.save(final_checkpoint_dir)
    print(f"✅ Final checkpoint saved: {final_checkpoint}")
    
    # Save final statistics
    training_stats['end_time'] = datetime.now().isoformat()
    training_stats['total_duration'] = time.time() - start_time
    training_stats['best_reward'] = best_reward
    
    stats_file = os.path.join(results_dir, 'training_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(training_stats, f, indent=2)
    print(f"✅ Training stats saved: {stats_file}")
    
    # Summary
    print()
    print("="*80)
    print("📊 Training Summary")
    print("="*80)
    print(f"  Total iterations: {len(training_stats['iterations'])}")
    print(f"  Total duration: {training_stats['total_duration'] / 3600:.2f}h")
    if iteration_times:
        print(f"  Avg iteration time: {np.mean(iteration_times):.2f}s")
    print(f"  Best reward: {best_reward:.2f}")
    print(f"  Final checkpoint: {final_checkpoint}")
    print("="*80)
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    run_logger.close()
    print()
    print("✅ Training complete!")

PYTHON_SCRIPT

# --- Training complete ---
echo ""
echo "================================================================================"
echo "✅ Training Complete!"
echo "================================================================================"
echo ""
echo "📁 Results saved to: ${RESULTS_DIR}"
echo "📁 Checkpoints saved to: ${CHECKPOINT_DIR}"
echo "📁 Logs saved to: ${LOG_DIR}"
echo ""
echo "📊 To analyze results:"
echo "   python testing/scripts/analyze_results.py ${RESULTS_DIR}/training_stats.json"
echo ""
echo "🔄 To evaluate checkpoint:"
echo "   python testing/scripts/evaluate_checkpoint.py ${CHECKPOINT_DIR} 10"
echo ""
echo "================================================================================"


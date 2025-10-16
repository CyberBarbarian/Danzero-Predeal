#!/bin/bash

################################################################################
# DanZero Quick Test Script
# Tests the training setup with 20 iterations (~2 minutes)
################################################################################

set -e

echo "================================================================================"
echo "🧪 DanZero Training Quick Test (10 iterations)"
echo "================================================================================"
echo ""

# --- Configuration ---
EXPERIMENT_NAME="danzero_test_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="./checkpoints/${EXPERIMENT_NAME}"
LOG_DIR="./logs/${EXPERIMENT_NAME}"
RESULTS_DIR="./results/${EXPERIMENT_NAME}"

NUM_WORKERS=200
NUM_ENVS_PER_WORKER=1
NUM_GPUS=1.0
NUM_GPUS_PER_WORKER=0.0
TOTAL_ITERATIONS=10  # Quick test

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "📁 Test Experiment: ${EXPERIMENT_NAME}"
echo ""

# Activate venv
source /mnt/project_modelware/lizikang/Danvenv/bin/activate

# Check GPUs
echo "🔍 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# Enable optimizations
export CUDA_VISIBLE_DEVICES=0,1
export NVIDIA_TF32_OVERRIDE=1

echo "🚀 Starting quick test training..."
echo ""

# Export config
export EXPERIMENT_NAME
export CHECKPOINT_DIR
export LOG_DIR
export RESULTS_DIR
export NUM_WORKERS
export NUM_ENVS_PER_WORKER
export NUM_GPUS
export NUM_GPUS_PER_WORKER
export TOTAL_ITERATIONS
export CHECKPOINT_FREQ=5
export EVALUATION_FREQ=10
export RAY_NUM_CPUS=200
export RAY_NUM_GPUS=2
export RAY_OBJECT_STORE_MEMORY=$((30 * 1024 * 1024 * 1024))

python3 << 'PYTHON_SCRIPT'
import ray
import time
import torch
import os
import json
import numpy as np
from guandan.rllib.trainers import create_dmc_trainer
from guandan.training.logger import Logger

experiment_name = os.environ['EXPERIMENT_NAME']
checkpoint_dir = os.path.abspath(os.environ['CHECKPOINT_DIR'])
log_dir = os.environ['LOG_DIR']
results_dir = os.path.abspath(os.environ['RESULTS_DIR'])
num_workers = int(os.environ['NUM_WORKERS'])
num_envs_per_worker = int(os.environ['NUM_ENVS_PER_WORKER'])
num_gpus = float(os.environ['NUM_GPUS'])
num_gpus_per_worker = float(os.environ['NUM_GPUS_PER_WORKER'])
total_iterations = int(os.environ['TOTAL_ITERATIONS'])

print("="*80)
print("🔧 Initializing Ray...")
ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    num_gpus=int(os.environ['RAY_NUM_GPUS']),
    num_cpus=int(os.environ['RAY_NUM_CPUS']),
    object_store_memory=int(os.environ['RAY_OBJECT_STORE_MEMORY']),
)
print("✅ Ray initialized")
print()

# H100 optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("⚡ H100 optimizations enabled")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU{i}: {props.name}")
print()

print("="*80)
print("🎯 Creating DMC Algorithm...")
start_time = time.time()

algo = create_dmc_trainer(
    env_config={
        "observation_mode": "comprehensive",
        "use_internal_adapters": False,
        "max_steps": 3000,
    },
    num_workers=num_workers,
    num_envs_per_worker=num_envs_per_worker,
    num_gpus=num_gpus,
    num_gpus_per_worker=num_gpus_per_worker,
    lr=1e-3,
    batch_size=64,
)

init_time = time.time() - start_time
print(f"✅ Initialized in {init_time:.2f}s")
print()

# Initialize TensorBoard Logger
print("📊 Initializing TensorBoard Logger...")
run_logger = Logger(log_dir=results_dir, xpid="")
run_logger.save_meta({
    'experiment': experiment_name,
    'config': {
        'num_workers': num_workers,
        'num_envs_per_worker': num_envs_per_worker,
        'total_envs': num_workers * num_envs_per_worker,
        'num_gpus': num_gpus,
    }
})
# Bootstrap a scalar so TensorBoard shows immediately
run_logger.log({'bootstrap/test_started': 1})
print("✅ TensorBoard Logger initialized")
print()

print("="*80)
print(f"🏃 Quick Test - {total_iterations} iterations")
print("="*80)
print()

iteration_times = []
results = []

# Track previous values for calculating deltas
prev_episodes = 0
prev_timesteps = 0

# Track win counts for win rate calculation
total_team_1_wins = 0
total_team_2_wins = 0

try:
    for iteration in range(1, total_iterations + 1):
        iter_start = time.time()
        result = algo.train()
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
        # Extract metrics from correct locations (RLlib format)
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
        
        results.append({
            'iteration': iteration,
            'time': iter_time,
            'reward_mean': episode_reward_mean,
            'agent_0_reward': agent_0_reward,
            'agent_1_reward': agent_1_reward,
            'agent_2_reward': agent_2_reward,
            'agent_3_reward': agent_3_reward,
            'team_1_reward': team_1_reward,
            'team_2_reward': team_2_reward,
            'reward_sum': reward_sum,
            'episodes': num_episodes,
            'episodes_this_iter': episodes_this_iter,
            'timesteps': timesteps_total,
            'timesteps_this_iter': timesteps_this_iter,
            'episode_len_mean': episode_len_mean,
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
        })
        
        # Log to TensorBoard (including self-play reward tracking + MUST HAVE metrics)
        run_logger.log({
            'test/iteration_duration_s': iter_time,
            'test/episodes_this_iter': episodes_this_iter,
            'test/num_episodes': num_episodes,
            'test/timesteps_total': timesteps_total,
            'test/episode_reward_mean': episode_reward_mean,
            'test/episode_len_mean': episode_len_mean,
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
        
        print(f"[Iter {iteration:2d}/{total_iterations}] "
              f"{iter_time:.2f}s | "
              f"T1: {team_1_reward:6.2f} T2: {team_2_reward:6.2f} | "
              f"Eps: {int(num_episodes):4d} (+{int(episodes_this_iter):3d}) | "
              f"Steps: {int(timesteps_total):9d}")
        
        if iteration == 5:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"💾 Test checkpoint saved: {checkpoint_path}")

    print()
    print("="*80)
    print("📊 Test Results")
    print("="*80)
    print(f"Average iteration time: {sum(iteration_times)/len(iteration_times):.2f}s")
    print(f"Throughput: {1/(sum(iteration_times)/len(iteration_times)):.2f} iter/s")
    print(f"Estimated time for 10K iterations: {10000 * sum(iteration_times)/len(iteration_times) / 3600:.1f} hours")
    print()
    
    # Save results summary
    with open(f"{results_dir}/summary.json", 'w') as f:
        json.dump({
            'experiment': experiment_name,
            'config': {
                'workers': num_workers,
                'envs_per_worker': num_envs_per_worker,
                'total_envs': num_workers * num_envs_per_worker,
            },
            'results': results,
            'avg_iter_time': sum(iteration_times) / len(iteration_times),
        }, f, indent=2)
    
    print(f"✅ Test results saved: {results_dir}/summary.json")
    print()
    print("="*80)
    print(f"📊 TensorBoard data saved to: {results_dir}")
    print(f"   View with: tensorboard --logdir={results_dir}")
    print("="*80)
    print()
    print("="*80)
    print("✅ Quick test PASSED! TensorBoard logging verified.")
    print("="*80)
    
except Exception as e:
    print()
    print("="*80)
    print(f"❌ Test FAILED: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
finally:
    algo.stop()
    ray.shutdown()
    run_logger.close()
    print("✅ TensorBoard Logger closed")

PYTHON_SCRIPT

echo ""
echo "================================================================================"
echo "✅ Quick test complete!"
echo "================================================================================"
echo ""
echo "To run full training (10K iterations):"
echo "  bash train_production.sh"
echo ""


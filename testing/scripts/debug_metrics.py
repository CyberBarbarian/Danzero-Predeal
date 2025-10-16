#!/usr/bin/env python3
"""
Debug script to see what metrics RLlib actually returns
"""

import ray
from guandan.rllib.trainers import create_dmc_trainer

print("="*80)
print("🔍 DEBUGGING RLLIB METRICS")
print("="*80)
print()

# Initialize Ray
ray.shutdown()
ray.init(ignore_reinit_error=True, num_gpus=1)

# Create trainer
print("Creating trainer...")
algo = create_dmc_trainer(
    env_config={
        "observation_mode": "comprehensive",
        "use_internal_adapters": False,
        "max_steps": 3000,
    },
    num_workers=2,  # Use just 2 workers for quick test
    num_envs_per_worker=1,
    num_gpus=1.0,
    num_gpus_per_worker=0.0,
    lr=1e-3,
    batch_size=64,
)

print("✅ Trainer created")
print()

# Run one iteration
print("Running one training iteration...")
result = algo.train()

print()
print("="*80)
print("📊 RESULT DICTIONARY KEYS")
print("="*80)
print()

# Print all top-level keys
print("Top-level keys:")
for key in sorted(result.keys()):
    print(f"  • {key}")

print()
print("="*80)
print("📈 CHECKING KEY METRIC LOCATIONS")
print("="*80)
print()

# Check common metric locations
metrics_to_check = [
    'episodes_total',
    'timesteps_total', 
    'episode_reward_mean',
    'episode_len_mean',
    'episodes_this_iter',
    'timesteps_this_iter',
    'num_env_steps_sampled',
    'num_env_steps_trained',
]

for metric in metrics_to_check:
    value = result.get(metric, "NOT FOUND")
    print(f"  {metric:30s}: {value}")

print()

# Check nested locations
print("="*80)
print("📂 CHECKING NESTED LOCATIONS")
print("="*80)
print()

if 'env_runners' in result:
    print("env_runners section:")
    for key, value in result['env_runners'].items():
        if not key.startswith('hist_') and not key.startswith('sampler_'):
            print(f"  env_runners.{key:30s}: {value}")
    print()

if 'info' in result:
    print("info section:")
    for key, value in result['info'].items():
        print(f"  info.{key:30s}: {value}")
    print()

if 'sampler_results' in result:
    print("sampler_results section:")
    print(f"  {result['sampler_results']}")

print()
print("="*80)
print("✅ Debug complete!")
print("="*80)

algo.stop()
ray.shutdown()


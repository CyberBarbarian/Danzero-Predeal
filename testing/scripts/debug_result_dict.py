#!/usr/bin/env python3
"""
Debug script to inspect the actual result dictionary from algo.train()
to identify correct metric keys.
"""

import ray
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from guandan.rllib.trainers import create_dmc_trainer

print("="*80)
print("🔍 DEBUG: Inspecting Result Dictionary")
print("="*80)
print()

# Initialize Ray
print("Initializing Ray...")
ray.shutdown()
ray.init(
    ignore_reinit_error=True,
    num_gpus=2,
    num_cpus=192,
    object_store_memory=10 * 1024 * 1024 * 1024,
)
print("✅ Ray initialized")
print()

# Create trainer
print("Creating DMC trainer...")
algo = create_dmc_trainer(
    env_config={
        "observation_mode": "comprehensive",
        "use_internal_adapters": False,
        "max_steps": 3000,
    },
    num_workers=150,
    num_envs_per_worker=4,
    num_gpus=1.0,
    num_gpus_per_worker=0.0,
)
print("✅ Trainer created")
print()

# Run 3 iterations
print("="*80)
print("Running 3 training iterations...")
print("="*80)
print()

for iteration in range(1, 4):
    print(f"\n--- Iteration {iteration} ---")
    result = algo.train()
    
    # Print ALL top-level keys
    print(f"\nTop-level keys in result:")
    for key in sorted(result.keys()):
        print(f"  • {key}")
    
    # Print specific metric values
    print(f"\nKey metrics:")
    print(f"  result.get('timesteps_total'): {result.get('timesteps_total')}")
    print(f"  result.get('num_env_steps_sampled'): {result.get('num_env_steps_sampled')}")
    print(f"  result.get('num_env_steps_trained'): {result.get('num_env_steps_trained')}")
    print(f"  result.get('num_agent_steps_sampled'): {result.get('num_agent_steps_sampled')}")
    print(f"  result.get('num_agent_steps_trained'): {result.get('num_agent_steps_trained')}")
    
    # Check env_runners section
    if 'env_runners' in result:
        print(f"\n  env_runners keys:")
        for key in sorted(result['env_runners'].keys()):
            value = result['env_runners'][key]
            if isinstance(value, (int, float, bool, str)):
                print(f"    • {key}: {value}")
            else:
                print(f"    • {key}: <{type(value).__name__}>")
    
    # Check sampler_results
    if 'sampler_results' in result:
        print(f"\n  sampler_results keys:")
        for key in sorted(result['sampler_results'].keys()):
            value = result['sampler_results'][key]
            if isinstance(value, (int, float, bool, str)):
                print(f"    • {key}: {value}")
            else:
                print(f"    • {key}: <{type(value).__name__}>")
    
    # Check info section
    if 'info' in result:
        print(f"\n  info keys:")
        for key in sorted(result['info'].keys()):
            value = result['info'][key]
            if isinstance(value, (int, float, bool, str)):
                print(f"    • {key}: {value}")
            else:
                print(f"    • {key}: <{type(value).__name__}>")
    
    # Save full result to file for inspection
    if iteration == 3:
        output_file = 'testing/outputs/debug_result_dict.json'
        with open(output_file, 'w') as f:
            # Convert to JSON-serializable format
            def make_serializable(obj):
                if isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(v) for v in obj]
                else:
                    return str(type(obj).__name__)
            
            serializable_result = make_serializable(result)
            json.dump(serializable_result, f, indent=2)
        
        print(f"\n✅ Full result dict saved to: {output_file}")

print()
print("="*80)
print("✅ DEBUG COMPLETE")
print("="*80)
print()
print("Check the output above to identify:")
print("  1. Correct key for timesteps")
print("  2. Correct key for episodes")
print("  3. Correct key for rewards")
print()

# Cleanup
algo.stop()
ray.shutdown()


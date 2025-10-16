#!/usr/bin/env python3
import time
import json
import os
import ray
from guandan.rllib.trainers import create_dmc_trainer

# Configure RLlib to log under logs/raw for later analysis
LOCAL_DIR = os.path.abspath('logs/raw')
os.makedirs(LOCAL_DIR, exist_ok=True)

print("=== Two-GPU 1000-iteration sanity test ===")

ray.shutdown()
ray.init(ignore_reinit_error=True)

# Use the new API for proper env runner collection; modest settings to keep it snappy
algo = create_dmc_trainer(
    env_config={"observation_mode": "comprehensive", "use_internal_adapters": False},
    num_workers=1,
    num_envs_per_worker=1,
    num_gpus=2,              # two cards
    batch_size=256,
    lr=1e-3,
)

start = time.time()
last_print = start

results_path = os.path.join(LOCAL_DIR, 'two_gpus_1000_results.jsonl')
with open(results_path, 'a') as f:
    for i in range(1000):
        r = algo.train()
        # Persist lightweight line-delimited JSON for post-hoc analysis
        line = {
            'iter': r.get('training_iteration', i+1),
            'env_steps': r.get('env_steps', r.get('env_steps_total', 0)),
            'update_count': r.get('update_count', None),
            'learner_results': r.get('learner_results', {}) or r.get('info', {}).get('learner', {}),
            'time_total_s': r.get('time_total_s', None),
        }
        f.write(json.dumps(line) + "\n")
        f.flush()
        now = time.time()
        if (i+1) % 50 == 0 or (now - last_print) > 30:
            loss = None
            lr = line['learner_results']
            if isinstance(lr, dict):
                for v in lr.values():
                    if isinstance(v, dict) and 'loss' in v:
                        loss = v['loss']
                        break
                if loss is None and 'loss' in lr:
                    loss = lr['loss']
            print(f"Iter {i+1:4d} | env_steps={line['env_steps']} | loss={loss}")
            last_print = now

print(f"Done. Elapsed: {time.time()-start:.1f}s. Results: {results_path}")
ray.shutdown()

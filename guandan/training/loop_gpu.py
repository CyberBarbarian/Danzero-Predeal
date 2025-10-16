"""
GPU-optimized training loop with batched action selection and GPU data pipeline
"""
from typing import Dict, Any
import yaml
import os
import uuid
import torch

from .rollout_worker_gpu import GPURolloutWorker
from .gpu_pipeline import create_gpu_pipeline
from .learner import Learner
from .logger import Logger
from .checkpoint import save_checkpoint


def run_gpu_training_iteration(
    *,
    log_dir: str = "logs",
    num_episodes: int = 1,
    batch_size: int = 1,
    epsilon: float = 0.2,
    config_path: str | None = None,
    action_batch_size: int = 32,  # Batch size for action selection
) -> Dict[str, Any]:
    """Run GPU-optimized training iteration"""
    xpid = f"dqmc_gpu_{uuid.uuid4().hex[:8]}"
    logger = Logger(log_dir=log_dir, xpid=xpid)
    
    # Load YAML config if provided
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    # Override from config
    if 'training' in cfg:
        num_episodes = cfg['training'].get('num_episodes', num_episodes)
    if 'learner' in cfg:
        batch_size = cfg['learner'].get('batch_size', batch_size)
        lr = cfg['learner'].get('lr', 1e-3)
        lambda_clip = cfg['learner'].get('lambda_clip', 0.2)
    else:
        lr = 1e-3
        lambda_clip = 0.2
    
    if 'rollout' in cfg:
        eps_start = cfg['rollout'].get('epsilon_start', 0.2)
        eps_end = cfg['rollout'].get('epsilon_end', 0.05)
        eps_decay = cfg['rollout'].get('epsilon_decay_steps', 10000)
    else:
        eps_start = 0.2
        eps_end = 0.05
        eps_decay = 10000

    # Model configuration
    if 'model' in cfg:
        model_hidden = tuple(cfg['model'].get('hidden', [512, 512, 512, 512, 512]))
        model_activation = cfg['model'].get('activation', 'tanh')
        model_orth_init = cfg['model'].get('orthogonal_init', True)
    else:
        model_hidden = (512, 512, 512, 512, 512)
        model_activation = 'tanh'
        model_orth_init = True

    # GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create GPU-optimized components
    worker = GPURolloutWorker(
        wid=0,
        epsilon=eps_start,
        epsilon_end=eps_end,
        epsilon_decay_steps=eps_decay,
        max_steps_per_episode=cfg.get('rollout', {}).get('max_steps_per_episode', 2000),
        model_hidden=model_hidden,
        model_activation=model_activation,
        model_orthogonal_init=model_orth_init,
        batch_size=action_batch_size,
    )
    
    # GPU-optimized data pipeline
    buffer_capacity = 100000
    gpu_buffer, gpu_loader = create_gpu_pipeline(buffer_capacity, batch_size, device)
    
    # Learner with shared model
    learner = Learner(lr=lr, lambda_clip=lambda_clip, model=worker.model, device=device)

    total_steps = 0
    total_rewards = 0.0
    updates = 0

    print(f"[GPU train] xpid={xpid} episodes={num_episodes} batch_size={batch_size} lr={lr} action_batch_size={action_batch_size}", flush=True)

    for ep in range(num_episodes):
        out = worker.run_episode()
        total_steps += out['steps']
        episode_reward = out['reward'].get('agent_0', 0.0)
        total_rewards += episode_reward

        # Add samples to GPU buffer
        for sample in out['samples']:
            gpu_buffer.add(sample)

        # Training updates
        if len(gpu_buffer) >= batch_size:
            updates_per_iter = cfg.get('learner', {}).get('updates_per_iter', 10)
            for _ in range(updates_per_iter):
                # Get batch directly on GPU
                batch = gpu_loader.get_batch()
                
                # Training step (already on GPU)
                stats = learner.update_gpu_batch(batch)
                updates += 1
                
                if updates % 10 == 0:
                    print(f"[GPU train] update={updates} loss={stats['loss']:.6f} buffer={len(gpu_buffer)}")

        # Checkpointing
        checkpoint_every = cfg.get('training', {}).get('checkpoint_every_updates', 100)
        if updates > 0 and updates % checkpoint_every == 0:
            checkpoint_path = f"{log_dir}/{xpid}/checkpoint_{updates}.pt"
            save_checkpoint(
                checkpoint_path,
                worker.model.state_dict(),
                learner.optimizer.state_dict(),
                {'updates': updates, 'episodes': ep}
            )
            print(f"[GPU train] checkpoint saved at updates={updates} -> {checkpoint_path}")

        if ep % 10 == 0:
            print(f"[GPU train] ep={ep}/{num_episodes} steps+={out['steps']} total_steps={total_steps} buffer={len(gpu_buffer)} updates={updates}")

    return {
        'xpid': xpid,
        'total_steps': total_steps,
        'total_rewards': total_rewards,
        'updates': updates
    }

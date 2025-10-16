from typing import Dict, Any
import yaml
import os
import uuid

from .rollout_worker import RolloutWorker
from .replay_buffer import ReplayBuffer
from .learner import Learner
from .logger import Logger
from .checkpoint import save_checkpoint


def run_training_iteration(
    *,
    log_dir: str = "logs",
    num_episodes: int = 1,
    batch_size: int = 1,
    epsilon: float = 0.2,
    config_path: str | None = None,
) -> Dict[str, Any]:
    xpid = f"dqmc_{uuid.uuid4().hex[:8]}"
    logger = Logger(log_dir=log_dir, xpid=xpid)
    # Load YAML config if provided
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    # Override from config
    if 'training' in cfg:
        num_episodes = cfg['training'].get('num_episodes', num_episodes)
        ckpt_every = cfg['training'].get('checkpoint_every_updates', 5)
    else:
        ckpt_every = 5
    if 'learner' in cfg:
        lr = cfg['learner'].get('lr', 1e-3)
        lambda_clip = cfg['learner'].get('lambda_clip', 0.2)
        batch_size = cfg['learner'].get('batch_size', batch_size)
        updates_per_iter = cfg['learner'].get('updates_per_iter', 1)
    else:
        lr = 1e-3
        lambda_clip = 0.2
        updates_per_iter = 1
    if 'rollout' in cfg:
        epsilon = cfg['rollout'].get('epsilon_start', epsilon)
        eps_end = cfg['rollout'].get('epsilon_end', 0.05)
        eps_decay = cfg['rollout'].get('epsilon_decay_steps', 10000)
    else:
        eps_end = 0.05
        eps_decay = 10000
    model_hidden = tuple(cfg.get('model', {}).get('hidden', [512, 512, 512, 512, 256]))
    model_activation = cfg.get('model', {}).get('activation', 'relu')
    model_orth_init = bool(cfg.get('model', {}).get('orthogonal_init', False))

    worker = RolloutWorker(
        wid=0,
        epsilon=epsilon,
        epsilon_end=eps_end,
        epsilon_decay_steps=eps_decay,
        max_steps_per_episode=cfg.get('rollout', {}).get('max_steps_per_episode', 2000),
        model_hidden=model_hidden,
        model_activation=model_activation,
        model_orthogonal_init=model_orth_init,
    )
    buffer = ReplayBuffer(capacity=100000)  # Even larger capacity for H100 optimization
    learner = Learner(lr=lr, lambda_clip=lambda_clip, model=worker.model)

    total_steps = 0
    total_rewards = 0.0
    updates = 0

    print(f"[train] xpid={xpid} episodes={num_episodes} batch_size={batch_size} lr={lr} lambda_clip={lambda_clip}", flush=True)

    for ep in range(num_episodes):
        out = worker.run_episode()
        total_steps += out['steps']
        # sum reward of agent_0 for simplicity
        episode_reward = out['reward'].get('agent_0', 0.0)
        total_rewards += episode_reward
        buffer.extend(out['samples'])
        print(f"[train] ep={ep+1}/{num_episodes} steps+={out['steps']} total_steps={total_steps} buffer={len(buffer)} updates={updates}", flush=True)

        if len(buffer) >= batch_size:
            for _ in range(updates_per_iter):
                tau, action, g = buffer.sample_batch(batch_size=batch_size)
                stats = learner.update({'tau': tau, 'action': action, 'g': g})
                updates += 1
                logger.log({'loss': stats['loss'], 'step': stats['step'], 'buffer_size': len(buffer)})
                if updates % 10 == 0:
                    print(f"[train] update={updates} loss={stats['loss']:.6f} buffer={len(buffer)}", flush=True)

        # periodic checkpoint
        if updates > 0 and updates % ckpt_every == 0:
            ckpt_path = os.path.join(logger.base, f'checkpoint_{updates}.pt')
            save_checkpoint(
                ckpt_path,
                model_state=learner.model.state_dict(),
                optim_state=learner.optimizer.state_dict(),
                meta={'updates': updates, 'total_steps': total_steps}
            )
            print(f"[train] checkpoint saved at updates={updates} -> {ckpt_path}", flush=True)

    logger.save_meta({'episodes': num_episodes, 'total_steps': total_steps, 'updates': updates})
    return {'xpid': xpid, 'total_steps': total_steps, 'total_rewards': total_rewards, 'updates': updates}



from typing import Any, Dict

from ray.rllib.algorithms.algorithm import Algorithm

from .builders import init_ray, build_dmc_config
from .algorithms import DMC


def create_dmc_trainer(
    env_config: Dict[str, Any] | None = None,
    *,
    num_workers: int = 150,  # CPU-optimized: 150 workers for 2 H100s (600 parallel envs)
    num_envs_per_worker: int = 4,  # 4 envs per worker (CPU utilization optimized)
    num_gpus: int | None = None,
    num_gpus_per_worker: float = 0.0,  # CPU workers for scalability (GPU learner only)
    use_inference_gpu: bool | None = None,  # If True, reserve 1 GPU for inference; None=auto
    lr: float = 1e-3,  # Learning rate from paper
    batch_size: int = 600,  # Scaled for 600 parallel envs (paper used 64 with fewer envs)
    epsilon_start: float = 0.2,  # Epsilon start from paper
    epsilon_end: float = 0.05,  # Epsilon end from paper
    epsilon_decay_steps: int = 10000,  # Epsilon decay from paper
) -> Algorithm:
    """Create a DMC Algorithm instance with configurable distributed settings.
    
    Uses the new RLModule/Learner API stack for better GPU utilization
    and modern RLlib features.
    """

    init_ray()
    config = build_dmc_config(
        env_config,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        num_gpus=num_gpus,
        num_gpus_per_worker=num_gpus_per_worker,
        use_inference_gpu=use_inference_gpu,
    )

    config.lr = lr
    config.batch_size = batch_size
    config.epsilon_start = epsilon_start
    config.epsilon_end = epsilon_end
    config.epsilon_decay_steps = epsilon_decay_steps

    return config.build()



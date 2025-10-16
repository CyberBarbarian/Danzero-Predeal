from typing import Any, Dict

import ray

from guandan.rllib.env_registry import register_env
from guandan.rllib.models import register_models
from guandan.rllib.algorithms import DMCConfig, DMCTorchRLModule, DMCLearner
from guandan.rllib.callbacks import GuandanMetricsCallback
from ray.rllib.policy.policy import PolicySpec
from guandan.env.rllib_env import GuandanMultiAgentEnv
from guandan.rllib.connectors.env_to_module.current_agent import CurrentAgentLegalActionConnector
from guandan.rllib.connectors.module_to_env.action_mapping import EpsilonGreedyActionConnector


def build_dmc_config(
    env_config: Dict[str, Any] | None = None,
    *,
    num_workers: int = 2,
    num_envs_per_worker: int = 1,
    num_gpus: int | None = None,
    num_gpus_per_worker: float = 0.0,
    use_inference_gpu: bool | None = None,
) -> DMCConfig:
    """Build a DMC config for RLlib training using the new API stack.

    Args:
        env_config: Environment configuration dict.
        num_workers: Number of rollout workers.
        num_envs_per_worker: Environments per worker.
        num_gpus: Total GPUs to assign to learner.
        num_gpus_per_worker: Fractions of GPU per rollout worker.
    """

    env_name = register_env()
    register_models()

    tmp_env = GuandanMultiAgentEnv(env_config or {})
    any_agent_id = tmp_env.agent_ids[0]
    obs_space = tmp_env.observation_spaces[any_agent_id]
    act_space = tmp_env.action_spaces[any_agent_id]
    
    # Use the multi-agent spaces directly from the environment
    ma_obs_spaces = tmp_env.observation_spaces
    ma_act_spaces = tmp_env.action_spaces

    config = DMCConfig()
    config.environment(env=env_name, env_config=env_config or {})
    config.framework("torch")

    # Enable new API stack (RLModule + Learner + EnvRunner v2)
    config.api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
    # Set our custom Learner class (still uses training() for learner_class)
    config.training(learner_class=DMCLearner)
    # Configure connectors: Env->Module provides (tau, legal_actions); Module->Env selects epsilon-greedy action
    # Force multi-agent env runner to avoid single-agent path selection.
    from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
    config.env_runners(
        num_env_runners=num_workers,
        num_envs_per_env_runner=num_envs_per_worker,
    )
    
    # Create multiple policies to ensure is_multi_agent=True for MultiAgentEnvRunner
    policies = {}
    for i in range(4):  # 4 agents in Guandan
        agent_id = f"agent_{i}"
        policies[agent_id] = PolicySpec(
            observation_space=obs_space,
            action_space=act_space,
            config={},
        )
    
    # Configure multi-agent RL module spec
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    
    module_specs = {}
    for agent_id in policies.keys():
        module_specs[agent_id] = RLModuleSpec(
            module_class=DMCTorchRLModule,
            observation_space=obs_space,
            action_space=act_space,
            model_config={
                "tau_dim": config.tau_dim,
                "action_dim": config.action_dim,
                "model_hidden": config.model_hidden,
                "model_activation": config.model_activation,
                "model_orthogonal_init": config.model_orthogonal_init,
            }
        )
    
    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs)
    )
    
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )

    cluster_gpus = ray.cluster_resources().get("GPU", 0)
    total_gpus = int(num_gpus) if num_gpus is not None else int(cluster_gpus)

    # New default: use all GPUs for learner. Optionally reserve 1 GPU for inference.
    # use_inference_gpu semantics:
    # - True: If >=1 GPU total, reserve 1 for inference (fractionally across env runners) and rest to learner
    # - False: No inference GPU; all GPUs to learner; workers per requested num_gpus_per_worker
    # - None: Auto: if total_gpus >= 2 and user didn't set per-worker GPU explicitly (0.0), reserve 1 for inference
    auto_infer = (use_inference_gpu is None and total_gpus >= 2 and num_gpus_per_worker == 0.0)
    enable_infer = (use_inference_gpu is True) or auto_infer

    if enable_infer and total_gpus >= 1:
        # Reserve 1 for inference, rest for learner (could be 0 if total_gpus == 1)
        learner_gpus = float(max(0, total_gpus - 1))
        # Even if learner_gpus becomes 0, RLlib still works on CPU learner for testing
        per_runner_gpu = 1.0 / max(1, num_workers)
    else:
        learner_gpus = float(total_gpus) if total_gpus > 0 else 0.0
        per_runner_gpu = num_gpus_per_worker

    # Apply GPU resources
    config.learners(num_learners=1, num_gpus_per_learner=learner_gpus)
    config.resources(num_gpus=learner_gpus)
    config.env_runners(num_gpus_per_env_runner=per_runner_gpu)

    # Set multi-agent Dict spaces for the config
    # This is required for MultiAgentEnvRunner to work properly with connectors
    from gymnasium.spaces import Dict as DictSpace
    config.observation_space = DictSpace(ma_obs_spaces)
    config.action_space = DictSpace(ma_act_spaces)
    
    # Add Guandan-specific callbacks for metrics tracking
    config.callbacks(GuandanMetricsCallback)

    return config




def init_ray(address: str | None = None):
    """Initialize Ray if not already initialized."""
    if ray.is_initialized():
        return
    if address:
        ray.init(address=address)
    else:
        ray.init()



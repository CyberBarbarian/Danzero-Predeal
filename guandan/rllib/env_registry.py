from typing import Dict, Any

from ray.tune.registry import register_env as ray_register_env

from guandan.env.rllib_env import GuandanMultiAgentEnv


def _env_creator(config: Dict[str, Any]) -> GuandanMultiAgentEnv:
    # Register raw multi-agent env (new API stack will handle MA correctly when configured)
    return GuandanMultiAgentEnv(config)


def register_env(env_name: str = "guandan_ma") -> str:
    """
    Register the Guandan multi-agent environment with RLlib.

    Returns the registered env name for convenience.
    """
    ray_register_env(env_name, _env_creator)
    return env_name



import pytest

from guandan.env.rllib_env import GuandanMultiAgentEnv
from guandan.rllib.models import register_models
from guandan.rllib.env_registry import register_env


def test_env_reset_step():
    env = GuandanMultiAgentEnv({"observation_mode": "simple"})
    obs, info = env.reset()
    assert set(obs.keys()) == set(env.agent_ids)
    actions = {aid: 0 for aid in env.agent_ids}
    obs, rewards, terminated, truncated, infos = env.step(actions)
    assert isinstance(rewards, dict)


def test_model_registration():
    register_models()
    name = register_env()
    assert isinstance(name, str)



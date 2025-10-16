from typing import Any, Dict, List

import numpy as np
import torch

from guandan.env.rllib_env import make_guandan_env
from guandan.training.actor_policy import select_action_epsilon_greedy
from guandan.rllib.models import GuandanQModel
from .epsilon import EpsilonScheduler
from .replay_buffer import Sample

class RolloutWorker:
    """环境采样占位类.
    后续将包装掼蛋Env并执行step收集trajectory.
    """
    def __init__(
        self,
        wid: int,
        epsilon: float = 0.2,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        max_steps_per_episode: int = 2000,
        model_hidden: tuple = (512, 512, 512, 512, 512),
        model_activation: str = "tanh",
        model_orthogonal_init: bool = True,
    ):
        self.wid = wid
        self.epsilon = float(epsilon)
        self.eps_sched = EpsilonScheduler(start=epsilon, end=epsilon_end, decay_steps=epsilon_decay_steps)
        self.env = make_guandan_env({'observation_mode': 'comprehensive', 'use_internal_adapters': False})
        self.model = GuandanQModel(hidden=model_hidden, activation=model_activation, orthogonal_init=model_orthogonal_init)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_steps_per_episode = int(max_steps_per_episode)

    def run_episode(self) -> Dict[str, Any]:
        obs, infos = self.env.reset()
        total_reward = {aid: 0.0 for aid in self.env.agent_ids}
        steps = 0
        samples: List[Sample] = []

        terminated = {aid: False for aid in self.env.agent_ids}
        truncated = {aid: False for aid in self.env.agent_ids}

        while not any(terminated.values()) and steps < self.max_steps_per_episode:
            current = infos['agent_0']['current_player']
            current_id = f'agent_{current}'
            legal_encoded = infos[current_id].get('legal_actions_encoded_valid') or infos[current_id]['legal_actions_encoded']

            tau = obs[current_id]
            idx = select_action_epsilon_greedy(tau, legal_encoded, self.model, epsilon=self.eps_sched.value(), device=self.device)
            actions = {aid: 0 for aid in self.env.agent_ids}
            actions[current_id] = idx
            obs, rewards, terminated, truncated, infos = self.env.step(actions)
            steps += 1
            for aid, r in rewards.items():
                total_reward[aid] += r

            # Store sample; g will be filled after episode ends
            if len(legal_encoded) > 0:
                action_vec = np.asarray(legal_encoded[idx], dtype=np.float32)
            else:
                action_vec = np.zeros((54,), dtype=np.float32)
            tau_t = np.asarray(tau, dtype=np.float32)
            samples.append(Sample(tau=tau_t, action54=action_vec, g=0.0, player_id=current))

        # Assign paper-style g from env-provided cumulative rewards (round + global bonus)
        if any(terminated.values()):
            final_rewards = rewards
            for i in range(len(samples)):
                pid = samples[i].player_id
                aid = f'agent_{pid}'
                g = float(final_rewards.get(aid, 0.0))
                samples[i] = Sample(tau=samples[i].tau, action54=samples[i].action54, g=g, player_id=pid)

        # Decay epsilon
        self.eps_sched.step(steps)
        return {'reward': total_reward, 'steps': steps, 'samples': samples}

__all__ = ['RolloutWorker']


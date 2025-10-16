from typing import Any, Dict

import ray

from .parameter_server import ParameterServer
from .learner import Learner
from .rollout_worker import RolloutWorker


@ray.remote
class ParameterServerActor:
    def __init__(self):
        self.ps = ParameterServer()

    def push(self, name: str, weights: Any, step: int):
        self.ps.push(name, weights, step)

    def pull(self, name: str):
        return self.ps.pull(name)


@ray.remote
class LearnerActor:
    def __init__(self, lr: float = 1e-3, lambda_clip: float = 0.2):
        self.learner = Learner(lr=lr, lambda_clip=lambda_clip)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return self.learner.update(batch)

    def get_weights(self):
        return self.learner.model.state_dict()

    def set_weights(self, weights):
        self.learner.model.load_state_dict(weights)


@ray.remote
class RolloutWorkerActor:
    def __init__(self, epsilon: float = 0.2, epsilon_end: float = 0.05, epsilon_decay_steps: int = 10000, max_steps_per_episode: int = 2000):
        self.worker = RolloutWorker(
            wid=0,
            epsilon=epsilon,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            max_steps_per_episode=max_steps_per_episode,
        )

    def set_weights(self, weights):
        self.worker.model.load_state_dict(weights)

    def run_episode(self):
        return self.worker.run_episode()



from typing import List

import numpy as np
import torch

from guandan.rllib.models import GuandanQModel


def select_action_epsilon_greedy(
    tau_513: np.ndarray,
    legal_actions_encoded: List[np.ndarray],
    model: GuandanQModel,
    epsilon: float = 0.1,
    device: str = "cpu",
) -> int:
    """
    Select an action index from legal_actions_encoded using ε-greedy on Q(τ,a).
    Returns the index within the provided legal list.
    """
    if len(legal_actions_encoded) == 0:
        return 0
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, len(legal_actions_encoded)))

    tau_t = torch.tensor(tau_513, dtype=torch.float32, device=device)
    acts_t = torch.tensor(np.stack(legal_actions_encoded, axis=0), dtype=torch.float32, device=device)
    q_vals = model.evaluate_legal_actions(tau_t, acts_t)
    return int(torch.argmax(q_vals).item())



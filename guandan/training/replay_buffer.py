from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class Sample:
    tau: np.ndarray        # (513,)
    action54: np.ndarray   # (54,)
    g: float               # target return per old-code: round_value + global_bonus
    player_id: int


class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
        self.capacity = capacity
        self.buffer: Deque[Sample] = deque(maxlen=capacity)

    def push(self, sample: Sample):
        self.buffer.append(sample)

    def extend(self, samples: List[Sample]):
        for s in samples:
            self.push(s)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert len(self.buffer) >= batch_size
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        taus, actions, g_list = [], [], []
        for i in idx:
            s = self.buffer[i]
            taus.append(s.tau.astype(np.float32))
            actions.append(s.action54.astype(np.float32))
            g_list.append(np.float32(s.g))
        return (
            np.stack(taus, axis=0),
            np.stack(actions, axis=0),
            np.asarray(g_list, dtype=np.float32),
        )



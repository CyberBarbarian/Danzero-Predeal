import numpy as np
from typing import List, Dict

from .utils import CardToNum


def encode_action_to_54d(action_cards: List[str]) -> np.ndarray:
    """
    Encode a list of card strings into a 54-dimensional count vector {0,1,2}.

    action_cards examples: ["H3", "S3", ...] or [] for PASS
    """
    vec = np.zeros(54, dtype=np.int32)
    for c in action_cards or []:
        if c in CardToNum:
            vec[CardToNum[c]] += 1
    return vec


def batch_encode_legal_actions(legal_actions: List[List]) -> List[np.ndarray]:
    """
    Convert env legal action entries to 54-d vectors.
    Each legal action format can vary; we expect either:
    - [] (PASS)
    - A structure where the last element is the card list (strings)
    - Or a flat list of card strings
    """
    encoded: List[np.ndarray] = []
    for action in legal_actions:
        if not action:
            encoded.append(np.zeros(54, dtype=np.int32))
            continue
        # Try to find list of card strings
        if isinstance(action[-1], list) and all(isinstance(x, str) for x in action[-1]):
            encoded.append(encode_action_to_54d(action[-1]))
        elif all(isinstance(x, str) for x in action):
            encoded.append(encode_action_to_54d(action))
        else:
            encoded.append(np.zeros(54, dtype=np.int32))
    return encoded



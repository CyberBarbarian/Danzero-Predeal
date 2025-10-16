"""Custom module->env connector mapping Q-based action selection to env action index."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from typing import Union

EpisodeType = Union[SingleAgentEpisode, MultiAgentEpisode]
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.core.columns import Columns


@PublicAPI
class EpsilonGreedyActionConnector(ConnectorV2):
    """Selects action using epsilon-greedy over Q-values and maps back to env index."""

    def __init__(self, epsilon_start: float = 0.2, epsilon_end: float = 0.05, epsilon_decay_steps: int = 10000):
        super().__init__()
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step_count = 0

    def __call__(
        self,
        *,
        rl_module,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # If actions are already set, return as-is
        if Columns.ACTIONS in batch:
            return batch

        q_values = batch.get("q_values")
        legal_actions = batch.get("legal_actions")

        if q_values is None or legal_actions is None:
            raise ValueError("Missing q_values or legal_actions in module output batch")

        # Convert to numpy if needed
        if hasattr(q_values, 'cpu'):
            q_values = q_values.cpu().numpy()
        if hasattr(legal_actions, 'cpu'):
            legal_actions = legal_actions.cpu().numpy()

        # Q-values should now be (batch_size, num_actions=54)
        # Legal_actions should be (batch_size, num_actions=54) with 1 for legal, 0 for illegal
        
        # Ensure 2D arrays
        if q_values.ndim == 1:
            q_values = np.expand_dims(q_values, axis=0)
        if legal_actions.ndim == 1:
            legal_actions = np.expand_dims(legal_actions, axis=0)
        
        batch_size = q_values.shape[0]
        num_actions = q_values.shape[1]
        
        # Process each item in the batch
        actions = []
        for i in range(batch_size):
            q_vals_i = q_values[i]  # (num_actions,)
            legal_i = legal_actions[i]  # (num_actions,)
            
            # Find legal action indices
            legal_indices = np.where(legal_i > 0)[0]
            
            if len(legal_indices) == 0:
                # Fallback: if no legal actions, pick action 0
                action_idx = 0
            else:
                # Calculate current epsilon
                epsilon = self._get_current_epsilon()
                
                # Decide whether to explore or exploit
                should_explore = explore if explore is not None else (np.random.random() < epsilon)
                
                if should_explore:
                    # Random action among legal actions
                    action_idx = int(np.random.choice(legal_indices))
                else:
                    # Greedy action: pick legal action with highest Q-value
                    legal_q_values = q_vals_i[legal_indices]
                    best_legal_idx = np.argmax(legal_q_values)
                    action_idx = int(legal_indices[best_legal_idx])
            
            actions.append(action_idx)
            
            # Update step count for epsilon decay
            self.step_count += 1

        # Set actions in batch as array
        batch[Columns.ACTIONS] = np.array(actions)
        return batch

    def _get_current_epsilon(self) -> float:
        """Calculate current epsilon value based on decay schedule."""
        if self.step_count >= self.epsilon_decay_steps:
            return self.epsilon_end
        
        progress = self.step_count / self.epsilon_decay_steps
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

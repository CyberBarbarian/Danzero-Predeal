"""Custom env->module connector for DanZero DMC.

Extracts the active agent's observation and the legal action encodings from the
episode so the RLModule receives a (513-D τ, N×54 action set) pair.
"""

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
class CurrentAgentLegalActionConnector(ConnectorV2):
    """Pulls current-agent observations and legal actions into module batch.

    Assumes the environment stores the legal action encodings in the last step's
    info dict under `legal_actions_encoded_valid` (preferred) or
    `legal_actions_encoded`.
    """

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
        # Process each episode and extract current agent's observation and legal actions
        for episode in episodes:
            if isinstance(episode, MultiAgentEpisode):
                # For multi-agent episodes, find the active agent only
                # This avoids batch dimension mismatches
                current_player = None
                try:
                    # Try to get current player from episode infos
                    if hasattr(episode, 'get_infos') and len(episode.agent_episodes) > 0:
                        # Get info from any agent episode that has data
                        for agent_id, sa_episode in episode.agent_episodes.items():
                            if len(sa_episode.infos) > 0:
                                latest_info = sa_episode.get_infos(-1)
                                current_player = latest_info.get("current_player")
                                break
                except Exception:
                    current_player = None
                
                if current_player is not None:
                    current_agent_id = f"agent_{current_player}"
                    if current_agent_id in episode.agent_episodes:
                        sa_episode = episode.agent_episodes[current_agent_id]
                        if len(sa_episode.observations) > 0:
                            self._write_observation_and_legals(batch, sa_episode, current_agent_id)
                else:
                    # Fallback: process first agent with observations
                    for agent_id, sa_episode in episode.agent_episodes.items():
                        if len(sa_episode.observations) > 0:
                            self._write_observation_and_legals(batch, sa_episode, agent_id)
                            break
            else:
                # Single agent episode
                self._write_observation_and_legals(batch, episode)

        return batch

    def _write_observation_and_legals(
        self, module_batch: Dict[str, Any], sa_episode: SingleAgentEpisode, agent_id: str = None
    ) -> None:
        # Latest observation for current agent (513-D tau)
        obs = sa_episode.get_observations(-1)
        obs_array = np.asarray(obs, dtype=np.float32)
        
        # Ensure observation has consistent shape
        if len(obs_array.shape) == 0:
            obs_array = obs_array.reshape(1)
        elif len(obs_array.shape) > 1:
            obs_array = obs_array.flatten()
            
        ConnectorV2.add_batch_item(
            module_batch,
            Columns.OBS,
            obs_array,
            sa_episode,
        )

        # Also store as 'tau' for compatibility with our RLModule
        ConnectorV2.add_batch_item(
            module_batch,
            "tau",
            obs_array,
            sa_episode,
        )

        # Extract legal actions from info dict
        info = sa_episode.get_infos(-1) if len(sa_episode.infos) > 0 else {}
        legal = info.get("legal_actions_encoded_valid") or info.get(
            "legal_actions_encoded"
        )
        
        if legal is None or len(legal) == 0:
            # Fallback: provide a default legal action if none found
            legal_arr = np.zeros((1, 54), dtype=np.float32)
        else:
            # Ensure legal actions are properly formatted with consistent shapes
            try:
                # Convert to numpy array first
                legal_list = []
                for action in legal:
                    if isinstance(action, (list, np.ndarray)):
                        action_arr = np.asarray(action, dtype=np.float32)
                        # Ensure each action is exactly 54 dimensions
                        if action_arr.size == 54:
                            legal_list.append(action_arr.reshape(54))
                        else:
                            # Pad or truncate to 54 dimensions
                            padded = np.zeros(54, dtype=np.float32)
                            copy_size = min(54, action_arr.size)
                            padded[:copy_size] = action_arr.flatten()[:copy_size]
                            legal_list.append(padded)
                    else:
                        # Invalid action format, use zero vector
                        legal_list.append(np.zeros(54, dtype=np.float32))
                
                if legal_list:
                    legal_arr = np.stack(legal_list, axis=0)  # Shape: (num_actions, 54)
                else:
                    legal_arr = np.zeros((1, 54), dtype=np.float32)
                    
            except Exception as e:
                # Fallback on any conversion error
                legal_arr = np.zeros((1, 54), dtype=np.float32)

        ConnectorV2.add_batch_item(
            module_batch,
            "legal_actions",
            legal_arr,
            sa_episode,
        )

        # Store info for potential use by other connectors
        ConnectorV2.add_batch_item(
            module_batch,
            Columns.INFOS,
            info,
            sa_episode,
        )


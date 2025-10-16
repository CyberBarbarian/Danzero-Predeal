"""
Tribute Phase Filter Connector

Per paper specification (METHOD Section E):
"As heuristic rules are employed during the tribute phase, data samples 
generated in this stage are not retained in the training process."

This connector filters out tribute phase samples to prevent biased training.
"""

from typing import Any, Dict, List, Optional

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.typing import EpisodeType


class TributePhaseFilterConnector(ConnectorV2):
    """
    Filters out tribute phase samples from training data.
    
    According to the paper, tribute phase is handled by heuristic rules,
    not learned behavior. Therefore, samples from tribute phase should
    be excluded from the training process to avoid:
    1. Learning biased behavior from heuristics
    2. Mixing learned and heuristic decisions
    3. Overfitting to tribute phase patterns
    
    The connector checks the 'exclude_from_training' flag in episode info
    and marks samples accordingly for filtering by the learner.
    """
    
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Filter tribute phase samples from the batch.
        
        Args:
            rl_module: The RLModule instance
            batch: The data batch being processed
            episodes: List of episodes
            explore: Whether in exploration mode
            shared_data: Shared data across connectors
            **kwargs: Additional arguments
            
        Returns:
            Modified batch with tribute phase samples marked for exclusion
        """
        # Check each episode for tribute phase markers
        for episode in episodes:
            # Get the latest info
            if hasattr(episode, 'get_infos') and len(episode.get_infos()) > 0:
                latest_info = episode.get_infos(-1)
                
                # Check if this sample is from tribute phase
                if latest_info.get('exclude_from_training', False):
                    # Mark this sample for exclusion
                    # The learner can check this flag and skip these samples
                    if 'exclude_mask' not in batch:
                        batch['exclude_mask'] = []
                    batch['exclude_mask'].append(True)
                else:
                    if 'exclude_mask' not in batch:
                        batch['exclude_mask'] = []
                    batch['exclude_mask'].append(False)
        
        return batch
    
    @staticmethod
    def should_exclude_from_training(info: Dict[str, Any]) -> bool:
        """
        Helper method to check if a sample should be excluded from training.
        
        Args:
            info: Info dict from environment step
            
        Returns:
            True if sample is from tribute phase and should be excluded
        """
        return info.get('exclude_from_training', False) or info.get('in_tribute_phase', False)

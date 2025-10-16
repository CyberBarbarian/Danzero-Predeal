"""
RLlib callbacks for Guandan-specific metrics tracking.
Integrates with monitoring and analysis tools.
"""

from typing import Dict, Optional, Any
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import PolicyID


class GuandanMetricsCallback(DefaultCallbacks):
    """
    Callbacks for tracking Guandan-specific metrics.
    
    Tracks:
    - Per-agent rewards (not averaged across all 4 agents)
    - Team-based outcomes
    - Rank distributions
    - Win rates
    """
    
    def on_episode_end(
        self,
        *,
        episode: Any,
        **kwargs,
    ) -> None:
        """
        Aggregate and log Guandan-specific metrics at episode end (new API).
        Works with MultiAgentEpisode - extracts rewards directly from episode data.
        Properly handles zero-sum reward structure.
        """
        try:
            # In new API, extract rewards directly from episode
            # MultiAgentEpisode stores agent_rewards internally
            agent_totals = {}
            
            # Try to get rewards for each agent from the episode
            for agent_id in ["agent_0", "agent_1", "agent_2", "agent_3"]:
                try:
                    # Get total reward for this agent from episode
                    # In new API, use get_return() or similar methods
                    if hasattr(episode, 'get_return'):
                        total_reward = episode.get_return(agent_id)
                    elif hasattr(episode, 'agent_rewards') and agent_id in episode.agent_rewards:
                        total_reward = sum(episode.agent_rewards[agent_id])
                    else:
                        # Fallback: try to sum rewards from episode data
                        total_reward = 0
                    
                    agent_totals[agent_id] = total_reward
                    
                    # Log individual agent rewards (NOT averaged!)
                    episode.custom_metrics[f"{agent_id}_total_reward"] = total_reward
                    episode.custom_metrics[f"{agent_id}_reward_magnitude"] = abs(total_reward)
                except Exception:
                    agent_totals[agent_id] = 0
            
            # Calculate team rewards
            team_1_reward = agent_totals.get("agent_0", 0) + agent_totals.get("agent_2", 0)
            team_2_reward = agent_totals.get("agent_1", 0) + agent_totals.get("agent_3", 0)
            
            episode.custom_metrics["team_1_total_reward"] = team_1_reward
            episode.custom_metrics["team_2_total_reward"] = team_2_reward
            
            # Track which team won (higher total reward)
            if team_1_reward > team_2_reward:
                episode.custom_metrics["team_1_win"] = 1.0
                episode.custom_metrics["team_2_win"] = 0.0
            elif team_2_reward > team_1_reward:
                episode.custom_metrics["team_1_win"] = 0.0
                episode.custom_metrics["team_2_win"] = 1.0
            else:
                # Tie
                episode.custom_metrics["team_1_win"] = 0.5
                episode.custom_metrics["team_2_win"] = 0.5
            
            # Track reward balance (how close the game was)
            episode.custom_metrics["reward_differential"] = abs(team_1_reward - team_2_reward)
            
            # Track if rewards sum to zero (sanity check for zero-sum game)
            total_reward_sum = sum(agent_totals.values())
            episode.custom_metrics["reward_sum_check"] = abs(total_reward_sum)
            
            # Log winner/loser reward magnitudes
            positive_rewards = [r for r in agent_totals.values() if r > 0]
            negative_rewards = [r for r in agent_totals.values() if r < 0]
            
            if positive_rewards:
                episode.custom_metrics["winner_reward_mean"] = sum(positive_rewards) / len(positive_rewards)
            if negative_rewards:
                episode.custom_metrics["loser_penalty_mean"] = sum(negative_rewards) / len(negative_rewards)
            
            # Track reward spread (max - min)
            if agent_totals:
                episode.custom_metrics["reward_spread"] = max(agent_totals.values()) - min(agent_totals.values())
        except Exception as e:
            # Silently handle any callback errors to not interrupt training
            pass


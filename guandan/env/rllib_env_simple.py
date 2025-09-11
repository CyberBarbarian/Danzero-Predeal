"""
Simplified RLLib MultiAgentEnv wrapper for Guandan game.

This module provides a MultiAgentEnv interface that wraps the existing
Guandan game environment to enable integration with Ray RLLib for
distributed reinforcement learning training.

This simplified version bypasses the problematic agent files and focuses
on the core environment functionality.

Author: DanZero Team
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .context import Context
from .engine import GameEnv
from .player import Player
from .table import Table
from .card_deck import CardDeck
from .observation_extractor import extract_observation
from .utils import legalaction, give_type


class GuandanMultiAgentEnvSimple(MultiAgentEnv):
    """
    Simplified MultiAgentEnv wrapper for Guandan card game.
    
    This class adapts the existing game logic to RLLib's MultiAgentEnv interface
    while bypassing the problematic agent files that contain null bytes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Guandan MultiAgentEnv.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        
        # Initialize the underlying game environment
        self.ctx = Context()
        self.env = GameEnv(self.ctx)
        
        # Agent identifiers for RLLib
        self.agent_ids = ["agent_0", "agent_1", "agent_2", "agent_3"]
        
        # Define observation and action spaces
        self.observation_spaces = self._define_observation_spaces()
        self.action_spaces = self._define_action_spaces()
        
        # Game state tracking
        self.current_agent = None
        self.episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.episode_steps = 0
        self.episode_done = False
        
        # Configuration
        self.config = config or {}
        
        # Legal actions cache
        self.current_legal_actions = []
        self.current_legal_actions_mask = np.zeros(100, dtype=np.float32)
        
    def _define_observation_spaces(self) -> Dict[str, gym.Space]:
        """
        Define observation spaces for each agent.
        
        Based on the observation extractor, observations include:
        - Hand cards (54 cards max)
        - Public information (8 dims: 4 players, 2 pieces of info each)
        - Game state (20 dims: ranks, current player, last action, etc.)
        - Legal actions mask (100 dims: maximum legal actions)
        - Action history (30 dims: recent action history)
        
        Total: 212 dimensions
        
        Returns:
            Dictionary mapping agent IDs to observation spaces
        """
        obs_dim = 212  # Total observation dimension
        
        return {
            agent_id: gym.spaces.Box(
                low=0, high=1, shape=(obs_dim,), dtype=np.float32
            ) for agent_id in self.agent_ids
        }
    
    def _define_action_spaces(self) -> Dict[str, gym.Space]:
        """
        Define action spaces for each agent.
        
        Actions are discrete indices into the legal action list.
        The maximum number of legal actions varies by game state.
        
        Returns:
            Dictionary mapping agent IDs to action spaces
        """
        max_actions = 100  # Maximum number of legal actions
        
        return {
            agent_id: gym.spaces.Discrete(max_actions) for agent_id in self.agent_ids
        }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment and return initial observations.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observations, infos) for all agents
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset the underlying environment
        self.env.ctx = Context()
        self.env.ctx.table = Table()
        self.env.ctx.card_decks = CardDeck()
        self.env.ctx.players = {}
        self.env.ctx.players_id_list = []
        for i in range(4):
            self.env.ctx.players[i] = Player(i)
            self.env.ctx.players_id_list.append(i)
        
        # Reset tracking variables
        self.episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.episode_steps = 0
        self.current_agent = None
        self.episode_done = False
        
        # Initialize a simple game state
        self.env.ctx.cur_rank = '2'
        self.env.ctx.player_waiting = 0
        self.env.ctx.win_order = []
        self.env.ctx.wind = False
        self.env.ctx.trick_pass = 0
        self.env.ctx.recv_wind = False
        self.env.ctx.last_action = None
        self.env.ctx.last_playid = None
        self.env.ctx.last_max_action = None
        self.env.ctx.last_max_playid = None
        self.env.round_end = False
        self.env.episode_end = False
        
        # Initialize player ranks
        for i in range(4):
            self.env.ctx.players[i].update_rank(self.env.ctx.cur_rank)
        
        # Get initial observations for all agents
        observations = {}
        infos = {}
        
        for i, agent_id in enumerate(self.agent_ids):
            obs = self._get_observation(agent_id, i)
            observations[agent_id] = obs
            infos[agent_id] = {"agent_id": agent_id, "player_id": i}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
        """
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}
        
        # Check if episode is done
        if self.episode_done:
            for agent_id in self.agent_ids:
                observations[agent_id] = self._get_observation(agent_id, 0)
                rewards[agent_id] = 0.0
                terminateds[agent_id] = True
                truncateds[agent_id] = False
                infos[agent_id] = {"episode_done": True}
            return observations, rewards, terminateds, truncateds, infos
        
        # Get current player
        current_player = self.env.ctx.player_waiting
        current_agent_id = f"agent_{current_player}"
        
        # Get legal actions for current player
        last_type, last_value = give_type(self.env.ctx)
        legal_actions = legalaction(self.env.ctx, last_type=last_type, last_value=last_value)
        self.current_legal_actions = legal_actions
        
        # Update legal actions mask
        self.current_legal_actions_mask.fill(0.0)
        num_legal = min(len(legal_actions), 100)
        self.current_legal_actions_mask[:num_legal] = 1.0
        
        # Get action for current player
        if current_agent_id in actions:
            action_idx = actions[current_agent_id]
            if 0 <= action_idx < len(legal_actions):
                # Execute the action
                action = legal_actions[action_idx]
                self.env.update(action)
            else:
                # Invalid action, use first legal action (PASS if available)
                action = legal_actions[0] if legal_actions else []
                self.env.update(action)
        else:
            # No action provided, use first legal action
            action = legal_actions[0] if legal_actions else []
            self.env.update(action)
        
        # Check if round or episode is done
        if self.env.round_end:
            self.env.upgrade()
            if self.env.episode_end:
                self.episode_done = True
        
        # Calculate rewards (simplified)
        rewards = self._calculate_rewards()
        
        # Get observations for all agents
        for i, agent_id in enumerate(self.agent_ids):
            observations[agent_id] = self._get_observation(agent_id, i)
            terminateds[agent_id] = self.episode_done
            truncateds[agent_id] = False
            infos[agent_id] = {
                "current_player": current_player,
                "legal_actions_count": len(legal_actions),
                "episode_steps": self.episode_steps
            }
        
        self.episode_steps += 1
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _get_observation(self, agent_id: str, player_id: int) -> np.ndarray:
        """
        Extract observation for a specific agent.
        
        Args:
            agent_id: RLLib agent identifier
            player_id: Internal player ID (0-3)
            
        Returns:
            Observation vector as numpy array
        """
        # Create a mock JSON message for observation extraction
        # This simulates the JSON structure that would come from the actual game
        mock_message = self._create_mock_message(player_id)
        
        # Extract observation using the observation extractor
        obs = extract_observation(mock_message, player_id)
        
        return obs
    
    def _create_mock_message(self, player_id: int) -> str:
        """
        Create a mock JSON message for observation extraction.
        
        Args:
            player_id: Player ID (0-3)
            
        Returns:
            JSON string representing the game state
        """
        import json
        
        # Mock hand cards (simplified)
        hand_cards = ["H2", "H3", "S4", "C5"] if player_id == 0 else ["H6", "H7", "S8", "C9"]
        
        # Mock public info
        public_info = [{"rest": 25}, {"rest": 26}, {"rest": 27}, {"rest": 24}]
        
        # Mock legal actions
        action_list = [["PASS", "PASS", "PASS"], ["Single", "2", ["H2"]]]
        
        # Mock current action
        cur_action = ["PASS", "PASS", "PASS"]
        greater_action = ["PASS", "PASS", "PASS"]
        
        message = {
            "type": "act",
            "stage": "play",
            "handCards": hand_cards,
            "myPos": player_id,
            "selfRank": 1,
            "oppoRank": 1,
            "curRank": 1,
            "publicInfo": public_info,
            "actionList": action_list,
            "curAction": cur_action,
            "greaterAction": greater_action,
            "curPos": self.env.ctx.player_waiting,
            "greaterPos": self.env.ctx.player_waiting
        }
        
        return json.dumps(message)
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """
        Calculate rewards for all agents.
        
        Returns:
            Dictionary mapping agent IDs to rewards
        """
        # Simplified reward structure
        # In a real implementation, this would be based on game outcomes
        rewards = {}
        
        for agent_id in self.agent_ids:
            # Basic reward: small positive for continuing, larger for winning
            if self.episode_done:
                # Check if this agent won
                player_id = int(agent_id.split("_")[1])
                if player_id in self.env.ctx.win_order[:2]:  # Top 2 players
                    rewards[agent_id] = 1.0
                else:
                    rewards[agent_id] = -0.5
            else:
                rewards[agent_id] = 0.01  # Small positive reward for continuing
        
        return rewards
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame if applicable
        """
        if mode == "human":
            print(f"Episode step: {self.episode_steps}")
            print(f"Current player: {self.env.ctx.player_waiting}")
            print(f"Episode done: {self.episode_done}")
            print(f"Win order: {self.env.ctx.win_order}")
        
        return None
    
    def close(self):
        """Close the environment and clean up resources."""
        pass


# Factory function for easy instantiation
def make_guandan_env_simple(config: Optional[Dict[str, Any]] = None) -> GuandanMultiAgentEnvSimple:
    """
    Factory function to create a simplified Guandan MultiAgentEnv instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured GuandanMultiAgentEnvSimple instance
    """
    return GuandanMultiAgentEnvSimple(config)


if __name__ == "__main__":
    # Test the environment
    env = make_guandan_env_simple()
    obs, info = env.reset()
    print("Environment reset successful!")
    print(f"Agent IDs: {list(obs.keys())}")
    print(f"Observation shapes: {[obs[aid].shape for aid in obs.keys()]}")
    
    # Test step
    actions = {agent_id: 0 for agent_id in env.agent_ids}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    print("Environment step successful!")
    print(f"Rewards: {rewards}")
    print(f"Terminated: {terminateds}")

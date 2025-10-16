"""
RLLib MultiAgentEnv wrapper with integrated agents for Guandan game.

This module provides a MultiAgentEnv interface that integrates existing
rule-based agents with RLLib for distributed reinforcement learning training.

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
from .comprehensive_observation_extractor import (
    ComprehensiveObservationExtractor, 
    ObservationMode, 
    create_simple_extractor,
    create_comprehensive_extractor
)
from .agent_adapter import create_agent_adapter
from .utils import legalaction, give_type
from .action_encoder import batch_encode_legal_actions


class GuandanMultiAgentEnv(MultiAgentEnv):
    """
    MultiAgentEnv wrapper for Guandan card game with integrated agents.
    
    This class integrates existing rule-based agents with RLLib's MultiAgentEnv interface,
    allowing for hybrid training where some agents are rule-based and others can be
    trained with reinforcement learning.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Guandan MultiAgentEnv with agents.
        
        Args:
            config: Configuration dictionary with agent types and settings
        """
        super().__init__()
        
        # Initialize the underlying game environment
        self.ctx = Context()
        self.env = GameEnv(self.ctx)
        
        # Agent identifiers for RLLib
        self.agent_ids = ["agent_0", "agent_1", "agent_2", "agent_3"]
        # Required for new API stack multi-agent detection
        self.agents = self.agent_ids.copy()
        self.possible_agents = self.agent_ids.copy()
        
        # Configuration
        self.config = config or {}
        # Prefer internal rule-based adapters unless disabled
        self.use_internal_adapters: bool = bool(self.config.get('use_internal_adapters', True))
        
        # Observation mode configuration
        self.observation_mode = self.config.get('observation_mode', 'simple')
        if self.observation_mode == 'comprehensive':
            self.obs_mode = ObservationMode.COMPREHENSIVE
            self.obs_dim = 513
        else:
            self.obs_mode = ObservationMode.SIMPLE
            self.obs_dim = 212
        
        # Initialize comprehensive observation extractor
        self.observation_extractor = ComprehensiveObservationExtractor(
            mode=self.obs_mode,
            config=self.config.get('observation_config', {})
        )
        
        # Define observation and action spaces
        self.observation_spaces = self._define_observation_spaces()
        self.action_spaces = self._define_action_spaces()
        
        # Game state tracking
        self.current_agent = None
        self.episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.episode_steps = 0
        self.episode_done = False
        # Track last round info for paper-style rewards
        self._last_round_first: Optional[int] = None
        self._last_round_partner_pos: Optional[int] = None  # 1-indexed (2: Follower, 3: Third, 4: Dweller)
        self._last_round_rank: Optional[str] = None
        # Accumulate per-round values per agent for strict paper-style shaping
        self._cumulative_round_values = {agent_id: 0.0 for agent_id in self.agent_ids}
        # Track global ±1 bonus per team (0: agents 0/2, 1: agents 1/3)
        self._global_bonus_team = [0.0, 0.0]
        # Tribute phase tracking (PAPER REQUIREMENT: exclude from training)
        self._in_tribute_phase = False
        self._tribute_phase_steps = 0
        # Configurable step limit with reasonable default for Guandan's multi-round structure
        self.max_steps = self.config.get('max_steps', 3000)  # Reasonable timeout for round-based evaluation
        
        # Agent configuration - specify which agents are rule-based vs RL
        self.agent_types = self.config.get('agent_types', {
            'agent_0': 'ai1',  # Rule-based agent
            'agent_1': 'ai2',  # Rule-based agent
            'agent_2': 'ai3',  # Rule-based agent
            'agent_3': 'ai4'   # Rule-based agent
        })
        
        # Initialize agent adapters only if enabled
        self.agent_adapters = {}
        if self.use_internal_adapters:
            for agent_id in self.agent_ids:
                agent_type = self.agent_types.get(agent_id, 'ai1')
                self.agent_adapters[agent_id] = create_agent_adapter(agent_type, int(agent_id.split('_')[1]), self.config)
        
        # Legal actions cache
        self.current_legal_actions = []
        self.current_legal_actions_mask = np.zeros(100, dtype=np.float32)
        
    def _define_observation_spaces(self) -> Dict[str, gym.Space]:
        """
        Define observation spaces for each agent.
        
        Returns:
            Dictionary mapping agent IDs to observation spaces
        """
        return {
            agent_id: gym.spaces.Box(
                low=0, high=1, shape=(self.obs_dim,), dtype=np.float32
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
        self._cumulative_round_values = {agent_id: 0.0 for agent_id in self.agent_ids}
        self._global_bonus_team = [0.0, 0.0]
        self._in_tribute_phase = False
        self._tribute_phase_steps = 0
        
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
        
        # Deal cards to players
        self.env.deal_cards()
        
        # Initialize game state properly
        self.env.battle_init()
        
        # Prepare legal actions after reset
        last_type, last_value = give_type(self.env.ctx)
        legal_actions = legalaction(self.env.ctx, last_type=last_type, last_value=last_value)
        encoded_legal = batch_encode_legal_actions(legal_actions)

        # Get initial observations for all agents
        observations = {}
        infos = {}
        current_player = self.env.ctx.player_waiting
        
        for i, agent_id in enumerate(self.agent_ids):
            obs = self._get_observation(agent_id, i)
            observations[agent_id] = obs
            infos[agent_id] = {
                "agent_id": agent_id,
                "player_id": i,
                "current_player": current_player,
                "legal_actions_count": len(legal_actions),
                "legal_actions_encoded": encoded_legal,
                "episode_steps": self.episode_steps,
                "in_tribute_phase": self._in_tribute_phase,
                "exclude_from_training": self._in_tribute_phase,
            }
        
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
        
        # Check if episode is done or max steps reached
        if self.episode_done or self.episode_steps >= self.max_steps:
            if self.episode_steps >= self.max_steps:
                print(f"Episode terminated due to max steps ({self.max_steps})")
            for agent_id in self.agent_ids:
                observations[agent_id] = self._get_observation(agent_id, 0)
                rewards[agent_id] = 0.0
                terminateds[agent_id] = True
                truncateds[agent_id] = self.episode_steps >= self.max_steps
                infos[agent_id] = {
                    "episode_done": True, 
                    "truncated": self.episode_steps >= self.max_steps,
                    "in_tribute_phase": self._in_tribute_phase,
                    "exclude_from_training": self._in_tribute_phase,
                }
            # Add required '__all__' key for multi-agent environments
            terminateds['__all__'] = True
            truncateds['__all__'] = self.episode_steps >= self.max_steps
            return observations, rewards, terminateds, truncateds, infos
        
        # Get current player
        current_player = self.env.ctx.player_waiting
        current_agent_id = f"agent_{current_player}"
        
        # PAPER REQUIREMENT: Detect tribute phase to exclude samples from training
        # Check game stage via message dict (defined in GameEnv parent class)
        try:
            stage = getattr(self.env, 'message', {}).get('stage', 'play')
            self._in_tribute_phase = stage in ['tribute', 'anti-tribute', 'back']
            if self._in_tribute_phase:
                self._tribute_phase_steps += 1
        except (AttributeError, TypeError):
            # Fallback: assume play phase if message not available
            self._in_tribute_phase = False
        
        # Get legal actions for current player
        last_type, last_value = give_type(self.env.ctx)
        legal_actions = legalaction(self.env.ctx, last_type=last_type, last_value=last_value)
        self.current_legal_actions = legal_actions
        
        # Update legal actions mask
        self.current_legal_actions_mask.fill(0.0)
        num_legal = min(len(legal_actions), 100)
        self.current_legal_actions_mask[:num_legal] = 1.0
        
        # Filter out empty actions for agent selection
        valid_legal_actions = [action for action in legal_actions if action and len(action) > 0]
        
        # Choose action: prefer provided action if available; otherwise use adapter if enabled
        if current_agent_id in actions and actions[current_agent_id] is not None and not np.isnan(actions[current_agent_id]).__bool__():
            action_idx = int(actions[current_agent_id])
        elif self.use_internal_adapters and current_agent_id in self.agent_adapters:
            action_idx = self.agent_adapters[current_agent_id].get_action(
                self._get_observation(current_agent_id, current_player), 
                legal_actions, 
                self.env.ctx
            )
        else:
            action_idx = 0
        
        # Map action index back to original legal actions list
        if 0 <= action_idx < len(valid_legal_actions):
            # Valid action - find the corresponding action in original list
            selected_action = valid_legal_actions[action_idx]
            # Find the index in the original legal_actions list
            try:
                original_idx = legal_actions.index(selected_action)
                action = legal_actions[original_idx]
            except ValueError:
                # Fallback to first valid action
                action = valid_legal_actions[0] if valid_legal_actions else []
        else:
            # Invalid action, use first valid action
            action = valid_legal_actions[0] if valid_legal_actions else []
        
        # Ensure action has the correct format (add last_type if missing)
        if len(action) > 0 and len(action) < 56:  # Action should have 54 cards + last_type + last_value
            # Add last_type and last_value to action
            last_type, last_value = give_type(self.env.ctx)
            action = action + [last_type, last_value]
        self.env.update(action)
        
        # Check if round or episode is done IMMEDIATELY after update
        if self.env.round_end:
            print(f"DEBUG: Round ended, calling upgrade()")
            # Capture pre-upgrade rank and partner position for reward calculation
            try:
                pre_rank = self.env.ctx.cur_rank
                first = self.env.ctx.win_order[0] if len(self.env.ctx.win_order) >= 1 else None
                partner_pos = None
                if first is not None:
                    partner = (first + 2) % 4
                    for pos, pid in enumerate(self.env.ctx.win_order):
                        if pid == partner:
                            partner_pos = pos + 1  # 1-indexed
                            break
                self._last_round_first = first
                self._last_round_partner_pos = partner_pos
                self._last_round_rank = pre_rank
                # Strict per-round value accumulation per paper
                if first is not None and partner_pos is not None:
                    # Determine magnitude
                    if pre_rank == 'A' and partner_pos != 4:
                        magnitude = 0.0
                    elif partner_pos == 2:
                        magnitude = 3.0
                    elif partner_pos == 3:
                        magnitude = 2.0
                    else:
                        magnitude = 1.0
                    partner = (first + 2) % 4
                    for i, agent_id in enumerate(self.agent_ids):
                        sign = 1.0 if (i == first or i == partner) else -1.0
                        self._cumulative_round_values[agent_id] += sign * magnitude
                    # Update global ±1 bonus when a team wins the episode
                    if self.env.episode_end:
                        winner_team = 0 if first in (0, 2) else 1
                        loser_team = 1 - winner_team
                        self._global_bonus_team[winner_team] = 1.0
                        self._global_bonus_team[loser_team] = -1.0
            except Exception as e:
                print(f"WARNING: Failed to capture last round info: {e}")
            
            self.env.upgrade()
            print(f"DEBUG: After upgrade - episode_end: {self.env.episode_end}, episode_done: {self.episode_done}")
            if self.env.episode_end:
                self.episode_done = True
                print(f"DEBUG: Episode completed! Setting episode_done = True")
            else:
                # Start new round
                print(f"DEBUG: Starting new round")
                self.env.battle_init()
        
        # If episode is done, return immediately with proper rewards
        if self.episode_done:
            # Calculate rewards for completed episode
            rewards = self._calculate_rewards()
            
            # Get observations for all agents
            for i, agent_id in enumerate(self.agent_ids):
                observations[agent_id] = self._get_observation(agent_id, i)
                terminateds[agent_id] = True
                truncateds[agent_id] = False
                infos[agent_id] = {
                    "current_player": current_player,
                    "legal_actions_count": len(legal_actions),
                    "episode_steps": self.episode_steps,
                    "episode_done": True,
                    "in_tribute_phase": self._in_tribute_phase,
                    "exclude_from_training": self._in_tribute_phase,
                }
            
            # Add required '__all__' key for multi-agent environments
            terminateds['__all__'] = True
            truncateds['__all__'] = False
            
            self.episode_steps += 1
            return observations, rewards, terminateds, truncateds, infos
        
        # Calculate rewards for ongoing episode
        rewards = self._calculate_rewards()
        
        # Get observations for all agents
        encoded_legal = batch_encode_legal_actions(legal_actions)
        encoded_valid_legal = batch_encode_legal_actions(valid_legal_actions)
        for i, agent_id in enumerate(self.agent_ids):
            observations[agent_id] = self._get_observation(agent_id, i)
            terminateds[agent_id] = self.episode_done
            truncateds[agent_id] = False
            infos[agent_id] = {
                "current_player": current_player,
                "legal_actions_count": len(legal_actions),
                "episode_steps": self.episode_steps,
                "legal_actions_encoded": encoded_legal,
                "legal_actions_encoded_valid": encoded_valid_legal,
                "in_tribute_phase": self._in_tribute_phase,
                "exclude_from_training": self._in_tribute_phase,
            }
        
        # Add required '__all__' key for multi-agent environments
        terminateds['__all__'] = self.episode_done
        truncateds['__all__'] = False
        
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
        # Create a JSON message for observation extraction
        message = self._create_observation_message(player_id)
        
        # Extract observation using the comprehensive observation extractor
        obs = self.observation_extractor.extract_observation(message, player_id)
        
        return obs
    
    def _create_observation_message(self, player_id: int) -> str:
        """
        Create a JSON message for observation extraction.
        
        Args:
            player_id: Player ID (0-3)
            
        Returns:
            JSON string representing the game state
        """
        import json
        
        # Get hand cards for the player
        hand_cards = self.env.ctx.players[player_id].handcards_in_str.split() if self.env.ctx.players[player_id].handcards_in_str else []
        
        # Get public info (card counts for all players)
        public_info = []
        for i in range(4):
            card_count = len(self.env.ctx.players[i].handcards_in_list) if hasattr(self.env.ctx.players[i], 'handcards_in_list') else 0
            public_info.append({'rest': card_count})
        
        # Get legal actions
        last_type, last_value = give_type(self.env.ctx)
        legal_actions = legalaction(self.env.ctx, last_type=last_type, last_value=last_value)
        
        # Convert legal actions to the format expected by observation extractor
        action_list = []
        for action in legal_actions:
            action_formatted = self._format_action_for_observation(action)
            action_list.append(action_formatted)
        
        # Get current action info
        cur_action = self._format_action_for_observation(self.env.ctx.last_action) if self.env.ctx.last_action else [None, None, None]
        greater_action = self._format_action_for_observation(self.env.ctx.last_max_action) if self.env.ctx.last_max_action else [None, None, None]
        
        # Create the message
        message = {
            "type": "act",
            "stage": "play",
            "handCards": hand_cards,
            "myPos": player_id,
            "selfRank": self.env.ctx.players[player_id].my_rank + 1,
            "oppoRank": self.env.ctx.players[(player_id + 1) % 4].my_rank + 1,
            "curRank": self.env.ctx.cur_rank,
            "publicInfo": public_info,
            "actionList": action_list,
            "indexRange": max(0, len(action_list) - 1),  # Ensure non-negative indexRange
            "curAction": cur_action,
            "greaterAction": greater_action,
            "curPos": self.env.ctx.player_waiting,
            "greaterPos": self.env.ctx.last_max_playid if self.env.ctx.last_max_playid is not None else -1
        }
        
        return json.dumps(message)
    
    def _format_action_for_observation(self, action) -> List:
        """
        Format action for observation extraction.
        
        Args:
            action: Action from game context
            
        Returns:
            Formatted action list
        """
        if action is None or len(action) == 0:
            return [None, None, None]
        
        # Use the game's action_form method if available
        if hasattr(self.env, 'action_form'):
            return self.env.action_form(action)
        else:
            # Fallback formatting
            return ["PASS", "PASS", "PASS"]
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """
        Calculate rewards for all agents using the game core's win/loss logic.
        
        Returns:
            Dictionary mapping agent IDs to rewards
        """
        rewards = {}
        
        for agent_id in self.agent_ids:
            if self.episode_done:
                idx = int(agent_id.split('_')[1])
                base = self._cumulative_round_values.get(agent_id, 0.0)
                bonus = self._global_bonus_team[0] if idx in (0, 2) else self._global_bonus_team[1]
                rewards[agent_id] = base + bonus
            else:
                # Game still ongoing
                rewards[agent_id] = 0.0
        
        return rewards
    
    def get_agent_action(self, agent_id: str, observation: np.ndarray) -> int:
        """
        Get action from a specific agent.
        
        Args:
            agent_id: Agent identifier
            observation: Current observation
            
        Returns:
            Action index
        """
        if agent_id not in self.agent_adapters:
            return 0
        
        # Get legal actions
        last_type, last_value = give_type(self.env.ctx)
        legal_actions = legalaction(self.env.ctx, last_type=last_type, last_value=last_value)
        
        # Get action from agent adapter
        return self.agent_adapters[agent_id].get_action(observation, legal_actions, self.env.ctx)
    
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
            print(f"Agent types: {self.agent_types}")
        
        return None
    
    def close(self):
        """Close the environment and clean up resources."""
        pass


# Factory function for easy instantiation
def make_guandan_env(config: Optional[Dict[str, Any]] = None) -> GuandanMultiAgentEnv:
    """
    Factory function to create a Guandan MultiAgentEnv with agents instance.
    
    Args:
        config: Optional configuration dictionary. Can include:
            - observation_mode: "simple" or "comprehensive"
            - observation_config: Configuration for observation extractor
            - agent_types: Dictionary mapping agent IDs to agent types
            
    Returns:
        Configured GuandanMultiAgentEnv instance
    """
    return GuandanMultiAgentEnv(config)


def make_guandan_env_simple() -> GuandanMultiAgentEnv:
    """Create a Guandan environment with simple observation mode (212 dimensions)."""
    config = {'observation_mode': 'simple'}
    return GuandanMultiAgentEnv(config)


def make_guandan_env_comprehensive() -> GuandanMultiAgentEnv:
    """Create a Guandan environment with comprehensive observation mode (513 dimensions)."""
    config = {'observation_mode': 'comprehensive'}
    return GuandanMultiAgentEnv(config)


if __name__ == "__main__":
    # Test the environment
    config = {
        'agent_types': {
            'agent_0': 'ai1',
            'agent_1': 'ai2', 
            'agent_2': 'ai3',
            'agent_3': 'ai4'
        }
    }
    
    env = make_guandan_env(config)
    obs, info = env.reset()
    print("Environment reset successful!")
    print(f"Agent IDs: {list(obs.keys())}")
    print(f"Observation shapes: {[obs[aid].shape for aid in obs.keys()]}")
    print(f"Agent types: {env.agent_types}")
    
    # Test step
    actions = {agent_id: 0 for agent_id in env.agent_ids}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    print("Environment step successful!")
    print(f"Rewards: {rewards}")
    print(f"Terminated: {terminateds}")

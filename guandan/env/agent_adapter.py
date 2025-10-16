"""
Agent Adapter for RLLib Integration

This module provides adapters to integrate existing rule-based agents
with the RLLib MultiAgentEnv interface. It bridges the gap between
RLLib's action-based interface and the existing JSON-based agent communication.

Author: DanZero Team
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
# Import agents with proper error handling
try:
    from ..agent.baselines.rule.ai1.client import Ai1_agent
except ImportError:
    print("Warning: ai1 agent not available")
    Ai1_agent = None

try:
    from ..agent.baselines.rule.ai2.client import Ai2_agent
except ImportError:
    print("Warning: ai2 agent not available")
    Ai2_agent = None

try:
    from ..agent.baselines.rule.ai3.client import Ai3_agent
except ImportError:
    print("Warning: ai3 agent not available")
    Ai3_agent = None

try:
    from ..agent.baselines.rule.ai4.client import Ai4_agent
except ImportError:
    print("Warning: ai4 agent not available")
    Ai4_agent = None

try:
    from ..agent.baselines.rule.ai6.client import Ai6_agent
except ImportError:
    print("Warning: ai6 agent not available")
    Ai6_agent = None

try:
    from ..agent.random_agent import RandomAgent
except ImportError:
    print("Warning: random agent not available")
    RandomAgent = None
from .utils import legalaction, give_type


class AgentAdapter:
    """
    Base adapter class for integrating existing agents with RLLib.
    
    This class provides a unified interface that converts between
    RLLib's action format and the existing agent's JSON message format.
    """
    
    def __init__(self, agent_class, agent_id: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent adapter.
        
        Args:
            agent_class: The agent class to wrap
            agent_id: Agent ID (0-3)
            config: Optional configuration dictionary
        """
        self.agent_id = agent_id
        self.agent = agent_class(agent_id)
        self.config = config or {}
        
        # Track game state for proper message creation
        self.last_message = None
        self.last_legal_actions = []
        
    def get_action(self, observation: np.ndarray, legal_actions: List, game_context: Any) -> int:
        """
        Get action from the agent based on observation and legal actions.
        
        Args:
            observation: Current observation vector
            legal_actions: List of legal actions
            game_context: Game context object
            
        Returns:
            Action index (0-based) into the legal actions list
        """
        # Ensure we have legal actions
        if not legal_actions or len(legal_actions) == 0:
            return 0
        
        # Create JSON message for the agent
        message = self._create_agent_message(legal_actions, game_context)
        
        # Get action from the agent
        try:
            action_index = self.agent.received_message(message)
            # Ensure action index is valid
            if 0 <= action_index < len(legal_actions):
                return action_index
            else:
                # Fallback to first action (usually PASS)
                return 0
        except Exception as e:
            print(f"Error getting action from agent {self.agent_id}: {e}")
            return 0
    
    def _create_agent_message(self, legal_actions: List, game_context: Any) -> str:
        """
        Create a JSON message that the existing agent expects.
        
        Args:
            legal_actions: List of legal actions
            game_context: Game context object
            
        Returns:
            JSON string message
        """
        # Get hand cards for this agent
        hand_cards = game_context.players[self.agent_id].handcards_in_str.split() if game_context.players[self.agent_id].handcards_in_str else []
        
        # Get public info (card counts for all players)
        public_info = []
        for i in range(4):
            card_count = len(game_context.players[i].handcards_in_list) if hasattr(game_context.players[i], 'handcards_in_list') else 0
            public_info.append({'rest': card_count})
        
        # Convert legal actions to the format expected by agents
        # Filter out empty actions first
        valid_actions = [action for action in legal_actions if action and len(action) > 0]
        
        action_list = []
        for action in valid_actions:
            action_formatted = self._format_action_for_agent(action, game_context)
            action_list.append(action_formatted)
        
        # If no valid actions, add a default PASS action
        if not action_list:
            action_list.append([None, None, None])
        
        # Get current action info
        cur_action = self._format_action_for_agent(game_context.last_action, game_context) if game_context.last_action else [None, None, None]
        greater_action = self._format_action_for_agent(game_context.last_max_action, game_context) if game_context.last_max_action else [None, None, None]
        
        # Create the message
        message = {
            "type": "act",
            "stage": "play",
            "handCards": hand_cards,
            "myPos": self.agent_id,
            "selfRank": game_context.players[self.agent_id].my_rank + 1,
            "oppoRank": game_context.players[(self.agent_id + 1) % 4].my_rank + 1,
            "curRank": game_context.cur_rank,
            "publicInfo": public_info,
            "actionList": action_list,
            "indexRange": max(0, len(action_list) - 1),  # Based on filtered action_list
            "curAction": cur_action,
            "greaterAction": greater_action,
            "curPos": game_context.player_waiting,
            "greaterPos": game_context.last_max_playid if game_context.last_max_playid is not None else -1
        }
        
        return json.dumps(message)
    
    def _format_action_for_agent(self, action, game_context) -> List:
        """
        Format action for agent consumption.
        
        Args:
            action: Action from game context
            game_context: Game context object
            
        Returns:
            Formatted action list
        """
        if action is None or len(action) == 0:
            return [None, None, None]
        
        # CRITICAL FIX: action_form is in the Game class, not game_context
        # We need to implement the action_form logic here directly
        from .utils import NumToCard
        
        res_list = []
        for (index, ele) in enumerate(action[:54]):
            if ele > 0:
                val = ele * [NumToCard[index]]
                res_list += val
        
        if len(action) >= 56:
            action_type = action[-1]
            action_value = action[-2]
            return [action_type, action_value, res_list]
        else:
            # Fallback for incomplete actions
            return ["PASS", "PASS", "PASS"]


class AgentAdapterFactory:
    """
    Factory class for creating agent adapters.
    """
    
    @staticmethod
    def create_adapter(agent_type: str, agent_id: int, config: Optional[Dict[str, Any]] = None) -> AgentAdapter:
        """
        Create an agent adapter for the specified agent type.
        
        Args:
            agent_type: Type of agent ('ai1', 'ai2', 'ai3', 'ai4', 'ai6', 'random')
            agent_id: Agent ID (0-3)
            config: Optional configuration dictionary
            
        Returns:
            AgentAdapter instance
        """
        agent_classes = {
            'ai1': Ai1_agent,
            'ai2': Ai2_agent,
            'ai3': Ai3_agent,
            'ai4': Ai4_agent,
            'ai6': Ai6_agent,
            'random': RandomAgent
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = agent_classes[agent_type]
        if agent_class is None:
            raise ImportError(f"Agent type {agent_type} is not available")
        
        return AgentAdapter(agent_class, agent_id, config)


# Convenience function for easy access
def create_agent_adapter(agent_type: str, agent_id: int, config: Optional[Dict[str, Any]] = None) -> AgentAdapter:
    """
    Create an agent adapter for the specified agent type.
    
    Args:
        agent_type: Type of agent ('ai1', 'ai2', 'ai3', 'ai4', 'ai6', 'random')
        agent_id: Agent ID (0-3)
        config: Optional configuration dictionary
        
    Returns:
        AgentAdapter instance
    """
    return AgentAdapterFactory.create_adapter(agent_type, agent_id, config)

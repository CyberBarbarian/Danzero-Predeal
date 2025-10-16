"""
Observation extraction utilities for Guandan RLLib environment.

This module provides functions to convert the existing JSON-based
observation system to standardized numpy arrays for RLLib compatibility.

Author: DanZero Team
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple
from .utils import CardToNum, RANK1, RANK2


class ObservationExtractor:
    """
    Extracts standardized observations from Guandan game state.
    
    This class converts the complex JSON-based observation system
    to fixed-size numpy arrays suitable for neural network training.
    """
    
    def __init__(self):
        """Initialize the observation extractor."""
        # Define observation dimensions
        self.hand_cards_dim = 54  # Maximum 54 cards in hand
        self.public_info_dim = 4 * 2  # 4 players, 2 pieces of info each
        self.game_state_dim = 20  # Game state information
        self.legal_actions_dim = 100  # Maximum legal actions
        self.action_history_dim = 10 * 3  # Last 10 actions, 3 components each
        
        # Total observation dimension
        self.total_obs_dim = (
            self.hand_cards_dim + 
            self.public_info_dim + 
            self.game_state_dim + 
            self.legal_actions_dim + 
            self.action_history_dim
        )
    
    def extract_observation(self, message: str, player_id: int) -> np.ndarray:
        """
        Extract observation from JSON message.
        
        Args:
            message: JSON message from environment
            player_id: Player ID (0-3)
            
        Returns:
            Standardized observation vector
        """
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            # Return zero observation if message is invalid
            return np.zeros(self.total_obs_dim, dtype=np.float32)
        
        # Extract different components
        hand_cards = self._extract_hand_cards(data, player_id)
        public_info = self._extract_public_info(data, player_id)
        game_state = self._extract_game_state(data, player_id)
        legal_actions = self._extract_legal_actions(data, player_id)
        action_history = self._extract_action_history(data, player_id)
        
        # Combine all components
        observation = np.concatenate([
            hand_cards,
            public_info,
            game_state,
            legal_actions,
            action_history
        ]).astype(np.float32)
        
        return observation
    
    def _extract_hand_cards(self, data: Dict[str, Any], player_id: int) -> np.ndarray:
        """
        Extract hand cards representation.
        
        Args:
            data: Parsed JSON data
            player_id: Player ID
            
        Returns:
            Hand cards vector
        """
        hand_cards = np.zeros(self.hand_cards_dim, dtype=np.float32)
        
        if 'handCards' in data and isinstance(data['handCards'], list):
            for card_str in data['handCards']:
                if card_str in CardToNum:
                    card_idx = CardToNum[card_str]
                    hand_cards[card_idx] = 1.0
        
        return hand_cards
    
    def _extract_public_info(self, data: Dict[str, Any], player_id: int) -> np.ndarray:
        """
        Extract public information about other players.
        
        Args:
            data: Parsed JSON data
            player_id: Player ID
            
        Returns:
            Public info vector
        """
        public_info = np.zeros(self.public_info_dim, dtype=np.float32)
        
        if 'publicInfo' in data and isinstance(data['publicInfo'], list):
            for i, player_info in enumerate(data['publicInfo']):
                if i < 4 and isinstance(player_info, dict):
                    # Card count
                    if 'rest' in player_info:
                        public_info[i * 2] = min(player_info['rest'] / 27.0, 1.0)  # Normalize
                    # Other public info can be added here
                    public_info[i * 2 + 1] = 0.0  # Placeholder
        
        return public_info
    
    def _extract_game_state(self, data: Dict[str, Any], player_id: int) -> np.ndarray:
        """
        Extract game state information.
        
        Args:
            data: Parsed JSON data
            player_id: Player ID
            
        Returns:
            Game state vector
        """
        game_state = np.zeros(self.game_state_dim, dtype=np.float32)
        idx = 0
        
        # Player position
        if 'myPos' in data:
            game_state[idx] = data['myPos'] / 3.0  # Normalize to [0, 1]
        idx += 1
        
        # Self rank
        if 'selfRank' in data:
            game_state[idx] = data['selfRank'] / 13.0  # Normalize to [0, 1]
        idx += 1
        
        # Opponent rank
        if 'oppoRank' in data:
            game_state[idx] = data['oppoRank'] / 13.0  # Normalize to [0, 1]
        idx += 1
        
        # Current rank
        if 'curRank' in data:
            # Handle both string and numeric rank values
            if isinstance(data['curRank'], str):
                # Convert string rank to numeric value
                rank_mapping = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
                rank_value = rank_mapping.get(data['curRank'], 0)
            else:
                rank_value = data['curRank']
            game_state[idx] = rank_value / 13.0  # Normalize to [0, 1]
        idx += 1
        
        # Current position
        if 'curPos' in data:
            game_state[idx] = data['curPos'] / 3.0 if data['curPos'] >= 0 else 0.0
        idx += 1
        
        # Greater position
        if 'greaterPos' in data:
            game_state[idx] = data['greaterPos'] / 3.0 if data['greaterPos'] >= 0 else 0.0
        idx += 1
        
        # Stage information (one-hot encoding)
        stage_encoding = np.zeros(6, dtype=np.float32)  # 6 possible stages
        if 'stage' in data:
            stage_map = {
                'beginning': 0, 'play': 1, 'tribute': 2, 
                'anti-tribute': 3, 'back': 4, 'episodeOver': 5
            }
            if data['stage'] in stage_map:
                stage_encoding[stage_map[data['stage']]] = 1.0
        
        game_state[idx:idx+6] = stage_encoding
        idx += 6
        
        # Fill remaining dimensions with zeros
        remaining = self.game_state_dim - idx
        if remaining > 0:
            game_state[idx:idx+remaining] = 0.0
        
        return game_state
    
    def _extract_legal_actions(self, data: Dict[str, Any], player_id: int) -> np.ndarray:
        """
        Extract legal actions mask.
        
        Args:
            data: Parsed JSON data
            player_id: Player ID
            
        Returns:
            Legal actions mask
        """
        legal_actions = np.zeros(self.legal_actions_dim, dtype=np.float32)
        
        if 'actionList' in data and isinstance(data['actionList'], list):
            num_actions = min(len(data['actionList']), self.legal_actions_dim)
            legal_actions[:num_actions] = 1.0
        
        return legal_actions
    
    def _extract_action_history(self, data: Dict[str, Any], player_id: int) -> np.ndarray:
        """
        Extract recent action history.
        
        Args:
            data: Parsed JSON data
            player_id: Player ID
            
        Returns:
            Action history vector
        """
        action_history = np.zeros(self.action_history_dim, dtype=np.float32)
        
        # Extract current action
        if 'curAction' in data and isinstance(data['curAction'], list):
            action_vec = self._encode_action(data['curAction'])
            action_history[:len(action_vec)] = action_vec
        
        # Extract greater action
        if 'greaterAction' in data and isinstance(data['greaterAction'], list):
            action_vec = self._encode_action(data['greaterAction'])
            start_idx = 3
            end_idx = start_idx + len(action_vec)
            if end_idx <= self.action_history_dim:
                action_history[start_idx:end_idx] = action_vec
        
        return action_history
    
    def _encode_action(self, action: List[Any]) -> np.ndarray:
        """
        Encode action to vector representation.
        
        Args:
            action: Action list [type, value, cards]
            
        Returns:
            Encoded action vector
        """
        if not action or len(action) < 3:
            return np.zeros(3, dtype=np.float32)
        
        action_vec = np.zeros(3, dtype=np.float32)
        
        # Action type (normalized)
        type_mapping = {
            'PASS': 0.0, 'Single': 0.1, 'Pair': 0.2, 'Trips': 0.3,
            'Bomb': 0.4, 'ThreeWithTwo': 0.5, 'Straight': 0.6,
            'ThreePair': 0.7, 'TwoTrips': 0.8, 'StraightFlush': 0.9,
            'tribute': 0.95, 'back': 1.0
        }
        action_vec[0] = type_mapping.get(action[0], 0.0)
        
        # Action value (normalized)
        if action[1] is None:
            action_vec[1] = 0.0
        elif isinstance(action[1], str):
            # Convert card value to number
            value_mapping = {
                '2': 0.0, '3': 0.1, '4': 0.2, '5': 0.3, '6': 0.4,
                '7': 0.5, '8': 0.6, '9': 0.7, 'T': 0.8, 'J': 0.9,
                'Q': 1.0, 'K': 1.1, 'A': 1.2, 'B': 1.3, 'R': 1.4,
                'JOKER': 1.5
            }
            action_vec[1] = value_mapping.get(action[1], 0.0)
        else:
            action_vec[1] = float(action[1]) / 20.0  # Normalize
        
        # Number of cards
        if isinstance(action[2], list):
            action_vec[2] = min(len(action[2]) / 10.0, 1.0)  # Normalize
        else:
            action_vec[2] = 0.0
        
        return action_vec


# Global instance for easy access
observation_extractor = ObservationExtractor()


def extract_observation(message: str, player_id: int) -> np.ndarray:
    """
    Convenience function to extract observation from message.
    
    Args:
        message: JSON message from environment
        player_id: Player ID (0-3)
        
    Returns:
        Standardized observation vector
    """
    return observation_extractor.extract_observation(message, player_id)

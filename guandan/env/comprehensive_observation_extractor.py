"""
Comprehensive Observation Extractor for Guandan RLLib Environment.

This module provides a unified observation space extractor that supports both
simple and comprehensive modes, integrating all observation space expansions
from existing agents into a centralized, configurable system.

Features:
- Simple mode: 212 dimensions (compatible with existing RLLib integration)
- Comprehensive mode: 513 dimensions (paper-compliant)
- Configurable observation components
- RLLib integration
- Backward compatibility with existing agents

Author: DanZero Team
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum
from .utils import CardToNum, RANK1, RANK2


class ObservationMode(Enum):
    """Observation space modes."""
    SIMPLE = "simple"           # 212 dimensions - basic observation
    COMPREHENSIVE = "comprehensive"  # 513 dimensions - paper-compliant


class ComprehensiveObservationExtractor:
    """
    Unified observation extractor supporting multiple modes and configurations.
    
    This class centralizes all observation space logic, providing a single
    interface for extracting observations in different modes while maintaining
    compatibility with existing agents and RLLib integration.
    """
    
    def __init__(self, mode: ObservationMode = ObservationMode.SIMPLE, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the comprehensive observation extractor.
        
        Args:
            mode: Observation mode (SIMPLE or COMPREHENSIVE)
            config: Additional configuration options
        """
        self.mode = mode
        self.config = config or {}
        
        # Initialize dimension specifications based on mode
        self._init_dimensions()
        
        # Card mapping for encoding
        self.card_to_index = self._create_card_mapping()
        
        # Initialize observation components
        self._init_components()
    
    def _init_dimensions(self):
        """Initialize dimension specifications based on mode."""
        if self.mode == ObservationMode.SIMPLE:
            # Simple mode: 212 dimensions (existing RLLib integration)
            self.hand_cards_dim = 54
            self.public_info_dim = 8  # 4 players × 2 info
            self.game_state_dim = 20
            self.legal_actions_dim = 100
            self.action_history_dim = 30
            self.total_obs_dim = 212
        else:
            # Comprehensive mode: 513 dimensions (paper-compliant)
            self.hand_cards_dim = 54      # [0-53]: Hand cards
            self.remaining_cards_dim = 54  # [54-107]: Remaining cards
            self.last_move_dim = 54       # [108-161]: Last move to cover
            self.partner_move_dim = 54    # [162-215]: Partner last move
            self.card_counts_dim = 84     # [216-299]: Card counts for 3 players
            self.played_cards_dim = 162   # [300-461]: Played cards for 3 players
            self.team_levels_dim = 40     # [462-501]: Team levels
            self.wild_flags_dim = 12      # [501-513]: Wild card flags
            self.total_obs_dim = 513
    
    def _create_card_mapping(self) -> Dict[str, int]:
        """Create mapping from card strings to indices (0-53)."""
        return CardToNum.copy()
    
    def _init_components(self):
        """Initialize observation component extractors."""
        # Component flags for configurable observation space
        self.components = {
            'hand_cards': True,
            'public_info': True,
            'game_state': True,
            'legal_actions': True,
            'action_history': True,
            'remaining_cards': self.mode == ObservationMode.COMPREHENSIVE,
            'last_move_to_cover': self.mode == ObservationMode.COMPREHENSIVE,
            'partner_last_move': self.mode == ObservationMode.COMPREHENSIVE,
            'card_counts': self.mode == ObservationMode.COMPREHENSIVE,
            'played_cards': self.mode == ObservationMode.COMPREHENSIVE,
            'team_levels': self.mode == ObservationMode.COMPREHENSIVE,
            'wild_flags': self.mode == ObservationMode.COMPREHENSIVE
        }
        
        # Override with config if provided
        for component, enabled in self.config.get('components', {}).items():
            if component in self.components:
                self.components[component] = enabled
    
    def extract_observation(self, message: str, player_id: int) -> np.ndarray:
        """
        Extract observation from JSON message.
        
        Args:
            message: JSON message from environment
            player_id: Player ID (0-3)
            
        Returns:
            Observation vector as numpy array
        """
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return np.zeros(self.total_obs_dim, dtype=np.float32)
        
        # Initialize observation vector
        obs = np.zeros(self.total_obs_dim, dtype=np.float32)
        current_idx = 0
        
        # Extract components based on mode and configuration
        if self.mode == ObservationMode.SIMPLE:
            current_idx = self._extract_simple_observation(obs, data, player_id, current_idx)
        else:
            current_idx = self._extract_comprehensive_observation(obs, data, player_id, current_idx)
        
        # Ensure all values are within [0.0, 1.0] range
        obs = np.clip(obs, 0.0, 1.0)
        
        return obs
    
    def _extract_simple_observation(self, obs: np.ndarray, data: Dict[str, Any], 
                                   player_id: int, start_idx: int) -> int:
        """Extract simple observation (212 dimensions)."""
        current_idx = start_idx
        
        # Hand cards (54 dimensions)
        if self.components['hand_cards']:
            current_idx = self._extract_hand_cards(obs, data, player_id, current_idx)
        
        # Public info (8 dimensions)
        if self.components['public_info']:
            current_idx = self._extract_public_info(obs, data, player_id, current_idx)
        
        # Game state (20 dimensions)
        if self.components['game_state']:
            current_idx = self._extract_game_state(obs, data, player_id, current_idx)
        
        # Legal actions (100 dimensions)
        if self.components['legal_actions']:
            current_idx = self._extract_legal_actions(obs, data, player_id, current_idx)
        
        # Action history (30 dimensions)
        if self.components['action_history']:
            current_idx = self._extract_action_history(obs, data, player_id, current_idx)
        
        return current_idx
    
    def _extract_comprehensive_observation(self, obs: np.ndarray, data: Dict[str, Any], 
                                         player_id: int, start_idx: int) -> int:
        """Extract comprehensive observation (513 dimensions)."""
        current_idx = start_idx
        
        # Hand cards [0-53]
        if self.components['hand_cards']:
            current_idx = self._extract_hand_cards(obs, data, player_id, current_idx)
        
        # Remaining cards [54-107]
        if self.components['remaining_cards']:
            current_idx = self._extract_remaining_cards(obs, data, player_id, current_idx)
        
        # Last move to cover [108-161]
        if self.components['last_move_to_cover']:
            current_idx = self._extract_last_move_to_cover(obs, data, player_id, current_idx)
        
        # Partner last move [162-215]
        if self.components['partner_last_move']:
            current_idx = self._extract_partner_last_move(obs, data, player_id, current_idx)
        
        # Card counts for other players [216-299]
        if self.components['card_counts']:
            current_idx = self._extract_card_counts(obs, data, player_id, current_idx)
        
        # Played cards for other players [300-461]
        if self.components['played_cards']:
            current_idx = self._extract_played_cards(obs, data, player_id, current_idx)
        
        # Team levels [462-501]
        if self.components['team_levels']:
            current_idx = self._extract_team_levels(obs, data, player_id, current_idx)
        
        # Wild card flags [501-513]
        if self.components['wild_flags']:
            current_idx = self._extract_wild_flags(obs, data, player_id, current_idx)
        
        return current_idx
    
    def _extract_hand_cards(self, obs: np.ndarray, data: Dict[str, Any], 
                           player_id: int, start_idx: int) -> int:
        """Extract hand cards representation."""
        if 'handCards' in data and isinstance(data['handCards'], list):
            for card_str in data['handCards']:
                if card_str in self.card_to_index:
                    card_idx = self.card_to_index[card_str]
                    obs[start_idx + card_idx] = 1.0
        
        return start_idx + self.hand_cards_dim
    
    def _extract_public_info(self, obs: np.ndarray, data: Dict[str, Any], 
                            player_id: int, start_idx: int) -> int:
        """Extract public information about other players."""
        if 'publicInfo' in data and isinstance(data['publicInfo'], list):
            for i, player_info in enumerate(data['publicInfo']):
                if i < 4 and isinstance(player_info, dict):
                    # Card count
                    if 'rest' in player_info:
                        obs[start_idx + i * 2] = min(player_info['rest'] / 27.0, 1.0)
                    # Other public info can be added here
                    obs[start_idx + i * 2 + 1] = 0.0  # Placeholder
        
        return start_idx + self.public_info_dim
    
    def _extract_game_state(self, obs: np.ndarray, data: Dict[str, Any], 
                           player_id: int, start_idx: int) -> int:
        """Extract game state information."""
        idx = start_idx
        
        # Player position
        if 'myPos' in data:
            obs[idx] = data['myPos'] / 3.0
        idx += 1
        
        # Self rank
        if 'selfRank' in data:
            obs[idx] = data['selfRank'] / 13.0
        idx += 1
        
        # Opponent rank
        if 'oppoRank' in data:
            obs[idx] = data['oppoRank'] / 13.0
        idx += 1
        
        # Current rank
        if 'curRank' in data:
            if isinstance(data['curRank'], str):
                rank_mapping = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                              '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
                rank_value = rank_mapping.get(data['curRank'], 0)
            else:
                rank_value = data['curRank']
            obs[idx] = rank_value / 13.0
        idx += 1
        
        # Current position
        if 'curPos' in data:
            obs[idx] = data['curPos'] / 3.0 if data['curPos'] >= 0 else 0.0
        idx += 1
        
        # Greater position
        if 'greaterPos' in data:
            obs[idx] = data['greaterPos'] / 3.0 if data['greaterPos'] >= 0 else 0.0
        idx += 1
        
        # Stage information (one-hot encoding)
        stage_encoding = np.zeros(6, dtype=np.float32)
        if 'stage' in data:
            stage_map = {
                'beginning': 0, 'play': 1, 'tribute': 2, 
                'anti-tribute': 3, 'back': 4, 'episodeOver': 5
            }
            if data['stage'] in stage_map:
                stage_encoding[stage_map[data['stage']]] = 1.0
        
        obs[idx:idx+6] = stage_encoding
        idx += 6
        
        # Fill remaining dimensions with zeros
        remaining = self.game_state_dim - (idx - start_idx)
        if remaining > 0:
            obs[idx:idx+remaining] = 0.0
        
        return start_idx + self.game_state_dim
    
    def _extract_legal_actions(self, obs: np.ndarray, data: Dict[str, Any], 
                              player_id: int, start_idx: int) -> int:
        """Extract legal actions mask."""
        if 'actionList' in data and isinstance(data['actionList'], list):
            num_actions = min(len(data['actionList']), self.legal_actions_dim)
            obs[start_idx:start_idx+num_actions] = 1.0
        
        return start_idx + self.legal_actions_dim
    
    def _extract_action_history(self, obs: np.ndarray, data: Dict[str, Any], 
                               player_id: int, start_idx: int) -> int:
        """Extract recent action history."""
        # Extract current action
        if 'curAction' in data and isinstance(data['curAction'], list):
            action_vec = self._encode_action(data['curAction'])
            obs[start_idx:start_idx+len(action_vec)] = action_vec
        
        # Extract greater action
        if 'greaterAction' in data and isinstance(data['greaterAction'], list):
            action_vec = self._encode_action(data['greaterAction'])
            start_idx += 3
            end_idx = start_idx + len(action_vec)
            if end_idx <= start_idx + self.action_history_dim:
                obs[start_idx:end_idx] = action_vec
        
        return start_idx + self.action_history_dim
    
    def _extract_remaining_cards(self, obs: np.ndarray, data: Dict[str, Any], 
                                player_id: int, start_idx: int) -> int:
        """Extract remaining cards (comprehensive mode)."""
        # For now, we don't have access to played cards in the current message format
        # This would require additional game state tracking
        # Set all remaining card positions to 0 for now
        # TODO: Implement proper remaining cards tracking
        return start_idx + self.remaining_cards_dim
    
    def _extract_last_move_to_cover(self, obs: np.ndarray, data: Dict[str, Any], 
                                   player_id: int, start_idx: int) -> int:
        """Extract last move to cover (comprehensive mode)."""
        # Check if player is leading (curPos == -1 means leading)
        if data.get('curPos', 0) == -1:
            return start_idx + self.last_move_dim
        
        # Extract current action that needs to be covered
        if 'curAction' in data and isinstance(data['curAction'], list):
            self._encode_move(obs, start_idx, data['curAction'])
        
        return start_idx + self.last_move_dim
    
    def _extract_partner_last_move(self, obs: np.ndarray, data: Dict[str, Any], 
                                  player_id: int, start_idx: int) -> int:
        """Extract partner last move (comprehensive mode)."""
        # For now, we don't have partner move information in the current message format
        # This would require additional game state tracking
        # Set all partner move positions to 0 for now
        # TODO: Implement proper partner move tracking
        return start_idx + self.partner_move_dim
    
    def _extract_card_counts(self, obs: np.ndarray, data: Dict[str, Any], 
                            player_id: int, start_idx: int) -> int:
        """Extract card counts for other 3 players (comprehensive mode)."""
        if 'publicInfo' in data and isinstance(data['publicInfo'], list):
            other_players = [(player_id + i) % 4 for i in [1, 2, 3]]  # Other 3 players
            
            for i, other_player_id in enumerate(other_players):
                if other_player_id < len(data['publicInfo']):
                    player_info = data['publicInfo'][other_player_id]
                    if isinstance(player_info, dict) and 'rest' in player_info:
                        # Store card count in the first dimension for each player
                        # The paper specifies 84 dimensions total (28 per player)
                        obs[start_idx + i * 28] = min(player_info['rest'] / 27.0, 1.0)
        
        return start_idx + self.card_counts_dim
    
    def _extract_played_cards(self, obs: np.ndarray, data: Dict[str, Any], 
                             player_id: int, start_idx: int) -> int:
        """Extract played cards for other 3 players (comprehensive mode)."""
        # For now, we don't have played cards information in the current message format
        # This would require additional game state tracking
        # Set all played card positions to 0 for now
        # TODO: Implement proper played cards tracking
        return start_idx + self.played_cards_dim
    
    def _extract_team_levels(self, obs: np.ndarray, data: Dict[str, Any], 
                            player_id: int, start_idx: int) -> int:
        """Extract team levels (comprehensive mode)."""
        # Our team level (13 dimensions)
        if 'selfRank' in data:
            self_rank = data['selfRank']
            if isinstance(self_rank, (int, float)):
                obs[start_idx] = self_rank / 13.0  # Store in first dimension
        
        # Opponent team level (13 dimensions)
        if 'oppoRank' in data:
            oppo_rank = data['oppoRank']
            if isinstance(oppo_rank, (int, float)):
                obs[start_idx + 13] = oppo_rank / 13.0  # Store in first dimension
        
        # Current level (13 dimensions)
        if 'curRank' in data:
            cur_rank = data['curRank']
            if isinstance(cur_rank, str):
                # Convert string rank to numeric
                rank_mapping = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                              '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
                rank_value = rank_mapping.get(cur_rank, 0)
            else:
                rank_value = cur_rank
            obs[start_idx + 26] = rank_value / 13.0  # Store in first dimension
        
        return start_idx + self.team_levels_dim
    
    def _extract_wild_flags(self, obs: np.ndarray, data: Dict[str, Any], 
                           player_id: int, start_idx: int) -> int:
        """Extract wild card flags (comprehensive mode)."""
        if 'handCards' in data and isinstance(data['handCards'], list):
            hand_cards = data['handCards']
            
            # Check for wild cards in hand
            wild_cards = [card for card in hand_cards if card in ['BJ', 'RJ']]
            
            # Flag 1: Has wild cards
            obs[start_idx] = 1.0 if wild_cards else 0.0
            
            # Flag 2: Number of wild cards
            obs[start_idx + 1] = min(len(wild_cards) / 2.0, 1.0)  # Normalize
            
            # Flags 3-12: Suitability for various combinations
            # This would require more complex logic to determine
            # For now, set basic flags based on hand composition
            obs[start_idx + 2] = 1.0 if len(hand_cards) >= 5 else 0.0  # Can form straights
            obs[start_idx + 3] = 1.0 if len(hand_cards) >= 2 else 0.0  # Can form pairs
            obs[start_idx + 4] = 1.0 if len(hand_cards) >= 3 else 0.0  # Can form trips
            obs[start_idx + 5] = 1.0 if len(hand_cards) >= 4 else 0.0  # Can form bombs
            obs[start_idx + 6] = 1.0 if len(hand_cards) >= 6 else 0.0  # Can form three pairs
            obs[start_idx + 7] = 1.0 if len(hand_cards) >= 8 else 0.0  # Can form two trips
            obs[start_idx + 8] = 1.0 if len(hand_cards) >= 5 else 0.0  # Can form straight flush
            obs[start_idx + 9] = 1.0 if wild_cards else 0.0  # Has joker bombs
            if start_idx + 10 < self.total_obs_dim:
                obs[start_idx + 10] = 0.0  # Placeholder
            if start_idx + 11 < self.total_obs_dim:
                obs[start_idx + 11] = 0.0  # Placeholder
        
        return start_idx + self.wild_flags_dim
    
    def _encode_action(self, action: List[Any]) -> np.ndarray:
        """Encode action to vector representation."""
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
                'Q': 1.0, 'K': 1.0, 'A': 1.0, 'B': 1.0, 'R': 1.0,
                'JOKER': 1.0
            }
            action_vec[1] = min(value_mapping.get(action[1], 0.0), 1.0)
        else:
            action_vec[1] = float(action[1]) / 20.0  # Normalize
        
        # Number of cards
        if isinstance(action[2], list):
            action_vec[2] = min(len(action[2]) / 10.0, 1.0)  # Normalize
        else:
            action_vec[2] = 0.0
        
        return action_vec
    
    def _encode_move(self, obs: np.ndarray, start_idx: int, move: List[Any]):
        """Encode a move into the observation vector."""
        if not move or len(move) < 2:
            return
        
        move_type = move[0]
        move_cards = move[1] if len(move) > 1 and isinstance(move[1], list) else []
        
        # Encode move type
        type_mapping = {
            'PASS': 0, 'Single': 1, 'Pair': 2, 'Trips': 3,
            'Bomb': 4, 'ThreeWithTwo': 5, 'Straight': 6,
            'ThreePair': 7, 'TwoTrips': 8, 'StraightFlush': 9,
            'tribute': 10, 'back': 11
        }
        obs[start_idx] = type_mapping.get(move_type, 0)
        
        # Encode cards in the move
        for i, card_str in enumerate(move_cards[:53]):  # Max 53 cards
            if card_str in self.card_to_index:
                obs[start_idx + 1 + self.card_to_index[card_str]] = 1.0
    
    def get_observation_space(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Get observation space specification.
        
        Returns:
            Tuple of (dimension, low_bounds, high_bounds)
        """
        low_bounds = np.zeros(self.total_obs_dim, dtype=np.float32)
        high_bounds = np.ones(self.total_obs_dim, dtype=np.float32)
        
        return self.total_obs_dim, low_bounds, high_bounds
    
    def get_component_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about observation components.
        
        Returns:
            Dictionary mapping component names to their specifications
        """
        info = {}
        current_idx = 0
        
        if self.mode == ObservationMode.SIMPLE:
            components = [
                ('hand_cards', self.hand_cards_dim),
                ('public_info', self.public_info_dim),
                ('game_state', self.game_state_dim),
                ('legal_actions', self.legal_actions_dim),
                ('action_history', self.action_history_dim)
            ]
        else:
            components = [
                ('hand_cards', self.hand_cards_dim),
                ('remaining_cards', self.remaining_cards_dim),
                ('last_move_to_cover', self.last_move_dim),
                ('partner_last_move', self.partner_move_dim),
                ('card_counts', self.card_counts_dim),
                ('played_cards', self.played_cards_dim),
                ('team_levels', self.team_levels_dim),
                ('wild_flags', self.wild_flags_dim)
            ]
        
        for name, dim in components:
            info[name] = {
                'dimensions': dim,
                'start_idx': current_idx,
                'end_idx': current_idx + dim,
                'enabled': self.components.get(name, False)
            }
            current_idx += dim
        
        return info


# Factory functions for easy instantiation
def create_simple_extractor(config: Optional[Dict[str, Any]] = None) -> ComprehensiveObservationExtractor:
    """Create a simple observation extractor (212 dimensions)."""
    return ComprehensiveObservationExtractor(ObservationMode.SIMPLE, config)


def create_comprehensive_extractor(config: Optional[Dict[str, Any]] = None) -> ComprehensiveObservationExtractor:
    """Create a comprehensive observation extractor (513 dimensions)."""
    return ComprehensiveObservationExtractor(ObservationMode.COMPREHENSIVE, config)


# Global instances for backward compatibility
simple_extractor = create_simple_extractor()
comprehensive_extractor = create_comprehensive_extractor()


def extract_observation(message: str, player_id: int, mode: str = "simple") -> np.ndarray:
    """
    Convenience function to extract observation from message.
    
    Args:
        message: JSON message from environment
        player_id: Player ID (0-3)
        mode: Observation mode ("simple" or "comprehensive")
        
    Returns:
        Observation vector as numpy array
    """
    if mode == "comprehensive":
        return comprehensive_extractor.extract_observation(message, player_id)
    else:
        return simple_extractor.extract_observation(message, player_id)

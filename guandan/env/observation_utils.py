"""
Utility functions for observation space management and compatibility.

This module provides utilities for loading configurations, maintaining
compatibility with existing agents, and managing observation space
transitions.

Author: DanZero Team
"""

import yaml
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
from .comprehensive_observation_extractor import (
    ComprehensiveObservationExtractor,
    ObservationMode,
    create_simple_extractor,
    create_comprehensive_extractor
)


def load_observation_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load observation configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "observation_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Return default configuration if file not found
        return {
            'observation_mode': 'simple',
            'observation_config': {
                'components': {
                    'hand_cards': True,
                    'public_info': True,
                    'game_state': True,
                    'legal_actions': True,
                    'action_history': True,
                    'remaining_cards': False,
                    'last_move_to_cover': False,
                    'partner_last_move': False,
                    'card_counts': False,
                    'played_cards': False,
                    'team_levels': False,
                    'wild_flags': False
                }
            },
            'agent_types': {
                'agent_0': 'ai1',
                'agent_1': 'ai2',
                'agent_2': 'ai3',
                'agent_3': 'ai4'
            }
        }


def create_observation_extractor(config: Optional[Dict[str, Any]] = None) -> ComprehensiveObservationExtractor:
    """
    Create observation extractor based on configuration.
    
    Args:
        config: Configuration dictionary. If None, loads from file.
        
    Returns:
        Configured observation extractor
    """
    if config is None:
        config = load_observation_config()
    
    mode_str = config.get('observation_mode', 'simple')
    mode = ObservationMode.COMPREHENSIVE if mode_str == 'comprehensive' else ObservationMode.SIMPLE
    
    observation_config = config.get('observation_config', {})
    
    return ComprehensiveObservationExtractor(mode=mode, config=observation_config)


def get_observation_space_info(extractor: ComprehensiveObservationExtractor) -> Dict[str, Any]:
    """
    Get detailed information about the observation space.
    
    Args:
        extractor: Observation extractor instance
        
    Returns:
        Dictionary with observation space information
    """
    dim, low_bounds, high_bounds = extractor.get_observation_space()
    component_info = extractor.get_component_info()
    
    return {
        'total_dimensions': dim,
        'mode': extractor.mode.value,
        'low_bounds': low_bounds,
        'high_bounds': high_bounds,
        'components': component_info
    }


def validate_observation_compatibility(obs1: np.ndarray, obs2: np.ndarray) -> bool:
    """
    Validate that two observations are compatible (same shape and type).
    
    Args:
        obs1: First observation array
        obs2: Second observation array
        
    Returns:
        True if compatible, False otherwise
    """
    return (obs1.shape == obs2.shape and 
            obs1.dtype == obs2.dtype and
            np.isfinite(obs1).all() and 
            np.isfinite(obs2).all())


def convert_observation_mode(obs: np.ndarray, from_mode: str, to_mode: str) -> np.ndarray:
    """
    Convert observation between different modes.
    
    Args:
        obs: Input observation array
        from_mode: Source mode ("simple" or "comprehensive")
        to_mode: Target mode ("simple" or "comprehensive")
        
    Returns:
        Converted observation array
    """
    if from_mode == to_mode:
        return obs.copy()
    
    if from_mode == "simple" and to_mode == "comprehensive":
        # Convert from 212 to 513 dimensions
        # Pad with zeros for additional components
        converted = np.zeros(513, dtype=obs.dtype)
        converted[:212] = obs
        return converted
    
    elif from_mode == "comprehensive" and to_mode == "simple":
        # Convert from 513 to 212 dimensions
        # Take only the first 212 dimensions
        return obs[:212].copy()
    
    else:
        raise ValueError(f"Unsupported conversion from {from_mode} to {to_mode}")


def create_agent_compatible_extractor(agent_type: str, config: Optional[Dict[str, Any]] = None) -> ComprehensiveObservationExtractor:
    """
    Create observation extractor compatible with specific agent type.
    
    Args:
        agent_type: Agent type identifier
        config: Configuration dictionary
        
    Returns:
        Agent-compatible observation extractor
    """
    if config is None:
        config = load_observation_config()
    
    # Agent-specific configurations
    agent_configs = {
        'ai1': {'observation_mode': 'simple'},
        'ai2': {'observation_mode': 'simple'},
        'ai3': {'observation_mode': 'simple'},
        'ai4': {'observation_mode': 'simple'},
        'ai6': {'observation_mode': 'simple'},
        'rllib': {'observation_mode': 'comprehensive'}  # RLLib agents use comprehensive mode
    }
    
    agent_config = agent_configs.get(agent_type, {'observation_mode': 'simple'})
    merged_config = {**config, **agent_config}
    
    return create_observation_extractor(merged_config)


def print_observation_space_summary(extractor: ComprehensiveObservationExtractor):
    """
    Print a summary of the observation space.
    
    Args:
        extractor: Observation extractor instance
    """
    info = get_observation_space_info(extractor)
    
    print(f"Observation Space Summary:")
    print(f"  Mode: {info['mode']}")
    print(f"  Total Dimensions: {info['total_dimensions']}")
    print(f"  Components:")
    
    for name, comp_info in info['components'].items():
        status = "✓" if comp_info['enabled'] else "✗"
        print(f"    {status} {name}: {comp_info['dimensions']} dims [{comp_info['start_idx']}:{comp_info['end_idx']}]")


# Backward compatibility functions
def extract_observation_simple(message: str, player_id: int) -> np.ndarray:
    """Backward compatibility function for simple observation extraction."""
    extractor = create_simple_extractor()
    return extractor.extract_observation(message, player_id)


def extract_observation_comprehensive(message: str, player_id: int) -> np.ndarray:
    """Backward compatibility function for comprehensive observation extraction."""
    extractor = create_comprehensive_extractor()
    return extractor.extract_observation(message, player_id)


# Legacy function names for backward compatibility
def extract_paper_observation(message: str, player_id: int) -> np.ndarray:
    """Legacy function name for paper-compliant observation extraction."""
    return extract_observation_comprehensive(message, player_id)


def extract_observation(message: str, player_id: int, mode: str = "simple") -> np.ndarray:
    """Legacy function for observation extraction with mode selection."""
    if mode == "comprehensive":
        return extract_observation_comprehensive(message, player_id)
    else:
        return extract_observation_simple(message, player_id)

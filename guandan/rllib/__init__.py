"""
RLlib integration scaffolding for Guandan.

This package will hold RLlib-specific adapters for DMC training:
- Environment registration helpers
- DMC algorithm configuration
- Multi-agent setup for self-play

TODOs:
- Implement DMC algorithm integration with RLLib
- Add distributed training support
- Wire callbacks to existing logger in `guandan/training/logger.py`
"""

from .env_registry import register_env
from .trainers import create_dmc_trainer

__all__ = [
    "register_env",
    "create_dmc_trainer",
]



# -*- coding: utf-8 -*-
"""
Centralized logging configuration for DanZero agents.
Provides efficient log management with configurable levels.
"""

import logging
import os
from typing import Optional


class AgentLogger:
    """Centralized logger for all agents with configurable verbosity."""
    
    _loggers = {}
    _default_level = logging.WARNING  # Default to minimal logging
    
    @classmethod
    def get_logger(cls, agent_name: str, level: Optional[int] = None) -> logging.Logger:
        """Get or create a logger for the specified agent."""
        if agent_name not in cls._loggers:
            logger = logging.getLogger(f"danzero.agent.{agent_name}")
            
            # Set level
            if level is None:
                level = cls._default_level
            logger.setLevel(level)
            
            # Avoid duplicate handlers
            if not logger.handlers:
                # Create console handler
                handler = logging.StreamHandler()
                handler.setLevel(level)
                
                # Create formatter
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                )
                handler.setFormatter(formatter)
                
                # Add handler to logger
                logger.addHandler(handler)
                
                # Prevent propagation to root logger
                logger.propagate = False
            
            cls._loggers[agent_name] = logger
        
        return cls._loggers[agent_name]
    
    @classmethod
    def set_default_level(cls, level: int):
        """Set the default logging level for all agents."""
        cls._default_level = level
        for logger in cls._loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
    
    @classmethod
    def set_agent_level(cls, agent_name: str, level: int):
        """Set logging level for a specific agent."""
        if agent_name in cls._loggers:
            logger = cls._loggers[agent_name]
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
    
    @classmethod
    def disable_logging(cls):
        """Disable all agent logging."""
        cls.set_default_level(logging.CRITICAL + 1)
    
    @classmethod
    def enable_debug_logging(cls):
        """Enable debug logging for all agents."""
        cls.set_default_level(logging.DEBUG)
    
    @classmethod
    def enable_info_logging(cls):
        """Enable info logging for all agents."""
        cls.set_default_level(logging.INFO)


def get_agent_logger(agent_name: str, level: Optional[int] = None) -> logging.Logger:
    """Convenience function to get an agent logger."""
    return AgentLogger.get_logger(agent_name, level)


def configure_logging_from_env():
    """Configure logging based on environment variables."""
    log_level = os.getenv('DANZERO_LOG_LEVEL', 'WARNING').upper()
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
        'DISABLE': logging.CRITICAL + 1
    }
    
    if log_level in level_map:
        AgentLogger.set_default_level(level_map[log_level])
    else:
        AgentLogger.set_default_level(logging.WARNING)


# Initialize logging configuration from environment
configure_logging_from_env()

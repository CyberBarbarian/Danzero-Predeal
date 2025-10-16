# Developer Guide - DanZero Guandan Framework

A comprehensive guide for developers working on the DanZero Guandan reinforcement learning framework.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Code Structure](#code-structure)
4. [Adding New Features](#adding-new-features)
5. [Testing](#testing)
6. [Debugging](#debugging)
7. [Performance Optimization](#performance-optimization)
8. [Contributing](#contributing)
9. [Code Style](#code-style)
10. [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Ray 2.0+
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/danzero.git
cd danzero

# Create virtual environment
python -m venv danvenv
source danvenv/bin/activate  # On Windows: danvenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "from guandan.env.rllib_env import make_guandan_env; env = make_guandan_env(); print('Environment ready!')"
```

## Development Environment

### IDE Setup

#### VS Code

1. Install Python extension
2. Install Pylance for type checking
3. Configure settings:

```json
{
    "python.defaultInterpreterPath": "./danvenv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

#### PyCharm

1. Open project directory
2. Configure Python interpreter to use virtual environment
3. Enable type checking
4. Configure code style (Black formatter)

### Environment Variables

```bash
# Development mode
export DANZERO_DEBUG=1
export DANZERO_LOG_LEVEL=DEBUG

# Training configuration
export RAY_DISABLE_IMPORT_WARNING=1
export CUDA_VISIBLE_DEVICES=0
```

## Code Structure

### Project Layout

```
DanZero/
├── guandan/                 # Main package
│   ├── env/                # Game environment
│   ├── agent/              # Agent implementations
│   ├── training/           # Training framework
│   └── config.py           # Configuration
├── tests/                  # Test suite
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── examples/               # Example code
```

### Key Directories

#### `guandan/env/`

Contains the game environment implementation:

- `engine.py`: Core game logic
- `rllib_env.py`: RLLib integration
- `utils.py`: Card utilities and legal actions
- `observation_extractor.py`: Observation space implementation

#### `guandan/agent/`

Contains agent implementations:

- `agents.py`: Agent registry
- `baselines/`: Rule-based agents
- `torch/`: Neural network agents

#### `guandan/training/`

Contains training framework:

- `ray_app.py`: Ray training orchestrator
- `learner.py`: Model training
- `rollout_worker.py`: Data collection

## Adding New Features

### Adding New Agents

1. **Create agent class**:

```python
# guandan/agent/baselines/new_agent.py
class NewAgent:
    def __init__(self, id):
        self.id = id
        self.name = "new_agent"
    
    def received_message(self, message):
        # Implement agent logic
        return action_index
```

2. **Register agent**:

```python
# guandan/agent/agents.py
from .baselines.new_agent import NewAgent

agent_cls = {
    # ... existing agents
    'new_agent': NewAgent,
}
```

3. **Add tests**:

```python
# tests/test_new_agent.py
def test_new_agent():
    agent = NewAgent(id=0)
    action = agent.received_message(test_message)
    assert isinstance(action, int)
```

### Adding New Observation Components

1. **Define component**:

```python
# guandan/env/observation_components.py
class NewComponent:
    def __init__(self):
        self.dimensions = 10
    
    def extract(self, message, player_id):
        # Extract component data
        return np.zeros(self.dimensions)
```

2. **Integrate with extractor**:

```python
# guandan/env/observation_extractor.py
class ObservationExtractor:
    def __init__(self):
        self.components = {
            'new_component': NewComponent(),
            # ... other components
        }
    
    def extract_observation(self, message, player_id):
        obs = []
        obs.extend(self.components['new_component'].extract(message, player_id))
        # ... other components
        return np.array(obs)
```

### Adding New Game Rules

1. **Modify game engine**:

```python
# guandan/env/engine.py
class GameEnv:
    def new_rule_method(self):
        # Implement new rule
        pass
```

2. **Update legal actions**:

```python
# guandan/env/utils.py
def legal_actions(cards_list, ctx):
    actions = []
    # ... existing actions
    
    # Add new rule actions
    if new_rule_condition:
        actions.append(new_action_index)
    
    return actions
```

3. **Add tests**:

```python
# tests/test_new_rules.py
def test_new_rule():
    env = GameEnv(Context())
    # Test new rule implementation
    assert env.new_rule_method() == expected_result
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_environment.py

# Run with coverage
python -m pytest --cov=guandan tests/

# Run with verbose output
python -m pytest -v tests/
```

### Test Structure

```
tests/
├── test_environment.py      # Environment tests
├── test_agents.py          # Agent tests
├── test_training.py        # Training tests
├── test_utils.py           # Utility tests
├── fixtures/               # Test fixtures
└── integration/            # Integration tests
```

### Writing Tests

```python
# tests/test_example.py
import pytest
from guandan.env.engine import GameEnv
from guandan.env.context import Context

class TestGameEnv:
    def setup_method(self):
        self.ctx = Context()
        self.env = GameEnv(self.ctx)
    
    def test_game_initialization(self):
        self.env.battle_init()
        assert self.env.ctx.players is not None
    
    def test_legal_actions(self):
        self.env.battle_init()
        actions = self.env.get_legal_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0
    
    @pytest.mark.parametrize("action", [0, 1, 2, 3])
    def test_valid_actions(self, action):
        self.env.battle_init()
        legal_actions = self.env.get_legal_actions()
        if action in legal_actions:
            # Test that valid action can be executed
            self.env.step(action)
```

### Test Fixtures

```python
# tests/fixtures/game_fixtures.py
import pytest
from guandan.env.engine import GameEnv
from guandan.env.context import Context

@pytest.fixture
def game_env():
    ctx = Context()
    env = GameEnv(ctx)
    env.battle_init()
    return env

@pytest.fixture
def sample_message():
    return {
        "hand_cards": ["AS", "KS", "QS"],
        "current_player": 0,
        "last_move": []
    }
```

## Debugging

### Debug Tools

#### Environment Debugging

```python
# Enable debug mode
env = make_guandan_env({'debug': True})

# Print game state
print(env.get_game_state())

# Print legal actions
print(env.get_legal_actions())

# Print observation
obs = env.reset()
print(f"Observation shape: {obs[list(obs.keys())[0]].shape}")
```

#### Agent Debugging

```python
# Enable agent debugging
agent = agent_cls['ai1'](id=0)
agent.debug = True

# Get detailed action info
action_info = agent.get_action_info(message)
print(action_info)
```

#### Training Debugging

```python
# Enable training debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor training progress
from guandan.training.logger import TrainingLogger
logger = TrainingLogger(debug=True)
```

### Common Debug Issues

1. **Action Index Errors**:
   - Check legal actions before executing
   - Verify action space dimensions

2. **Observation Shape Mismatch**:
   - Ensure observation mode consistency
   - Check component dimensions

3. **Memory Issues**:
   - Monitor memory usage
   - Use smaller batch sizes
   - Clear unused variables

## Performance Optimization

### Profiling

```python
# Profile code execution
import cProfile
import pstats

def profile_function():
    # Your code here
    pass

cProfile.run('profile_function()', 'profile_output.prof')

# Analyze results
stats = pstats.Stats('profile_output.prof')
stats.sort_stats('cumulative').print_stats(10)
```

### Optimization Techniques

1. **Vectorization**:
   ```python
   # Use NumPy operations instead of loops
   import numpy as np
   
   # Slow
   result = []
   for i in range(len(data)):
       result.append(data[i] * 2)
   
   # Fast
   result = data * 2
   ```

2. **Caching**:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def expensive_calculation(param):
       # Expensive computation
       return result
   ```

3. **Batch Processing**:
   ```python
   # Process multiple games simultaneously
   def process_batch(games):
       return [process_game(game) for game in games]
   ```

### Memory Management

```python
# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Clear unused variables
del large_variable
import gc
gc.collect()
```

## Contributing

### Workflow

1. **Fork repository**
2. **Create feature branch**:
   ```bash
   git checkout -b feature/new-feature
   ```
3. **Make changes**
4. **Add tests**
5. **Run tests**:
   ```bash
   python -m pytest tests/
   ```
6. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add new feature"
   ```
7. **Push branch**:
   ```bash
   git push origin feature/new-feature
   ```
8. **Create pull request**

### Pull Request Guidelines

1. **Clear description** of changes
2. **Reference issues** if applicable
3. **Include tests** for new functionality
4. **Update documentation** if needed
5. **Ensure all tests pass**

### Code Review Process

1. **Automated checks** must pass
2. **At least one reviewer** approval required
3. **No merge conflicts**
4. **Documentation updated**

## Code Style

### Python Style Guide

Follow PEP 8 with these additions:

```python
# Type hints
def function_name(param: int) -> str:
    return str(param)

# Docstrings
def complex_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is invalid
    """
    pass

# Line length: 88 characters (Black default)
# Import order: standard, third-party, local
import os
import sys

import numpy as np
import torch

from guandan.env.engine import GameEnv
```

### Naming Conventions

- **Classes**: PascalCase (`GameEnv`)
- **Functions/Variables**: snake_case (`get_legal_actions`)
- **Constants**: UPPER_CASE (`MAX_CARDS`)
- **Private methods**: Leading underscore (`_internal_method`)

### Documentation

```python
class ExampleClass:
    """Brief description of class.
    
    Longer description if needed.
    
    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2
    """
    
    def __init__(self, param: int):
        """Initialize ExampleClass.
        
        Args:
            param: Description of param
        """
        self.attr1 = param
        self.attr2 = None
    
    def public_method(self, param: str) -> bool:
        """Brief description of method.
        
        Args:
            param: Description of param
        
        Returns:
            Description of return value
        """
        return True
    
    def _private_method(self):
        """Private method description."""
        pass
```

## Release Process

### Version Numbering

Follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `setup.py`
2. **Update changelog** in `CHANGELOG.md`
3. **Run full test suite**
4. **Update documentation**
5. **Create release tag**:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```
6. **Build and upload** to PyPI

### Changelog Format

```markdown
## [1.0.0] - 2024-01-01

### Added
- New feature 1
- New feature 2

### Changed
- Changed behavior 1
- Changed behavior 2

### Fixed
- Bug fix 1
- Bug fix 2

### Removed
- Removed feature 1
- Removed feature 2
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Check virtual environment activation
   - Verify package installation
   - Check Python path

2. **CUDA Errors**:
   - Verify CUDA installation
   - Check PyTorch CUDA compatibility
   - Set CUDA_VISIBLE_DEVICES

3. **Ray Errors**:
   - Check Ray installation
   - Verify cluster configuration
   - Check resource allocation

### Getting Help

1. **Check documentation**
2. **Search existing issues**
3. **Create new issue** with:
   - Error message
   - Steps to reproduce
   - Environment details
   - Expected vs actual behavior

### Development Resources

- **API Documentation**: `docs/api/API_REFERENCE.md`
- **Project Structure**: `docs/reference/PROJECT_STRUCTURE.md`
- **Observation Space**: `docs/reference/OBSERVATION_SPACE.md`
- **Unified Extractor**: `docs/reference/UNIFIED_EXTRACTOR.md`

## License

Apache License 2.0 - see LICENSE file for details.

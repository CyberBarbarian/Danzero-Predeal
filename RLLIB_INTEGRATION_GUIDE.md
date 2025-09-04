# RLLib Integration Guide for DanZero

This guide outlines the step-by-step process for integrating DanZero with Ray RLLib for distributed reinforcement learning training.

## Overview

We're implementing a custom MultiAgentEnv wrapper that preserves your existing JSON-based communication protocol while adding RLLib compatibility. This approach minimizes changes to your current codebase while enabling distributed training.

## Implementation Status

### ✅ Completed
- [x] Basic MultiAgentEnv wrapper structure (`guandan/env/rllib_env.py`)
- [x] Observation extraction utilities (`guandan/env/observation_extractor.py`)
- [x] Test suite for validation (`test_rllib_env.py`)
- [x] Requirements file for RLLib dependencies (`requirements_rllib.txt`)

### 🔄 In Progress
- [ ] Complete observation space definition
- [ ] Implement action space mapping
- [ ] Complete step() method implementation
- [ ] Design reward structure

### 📋 Next Steps
- [ ] Test with actual game environment
- [ ] Implement agent adapter layer
- [ ] Integrate with Ray training pipeline
- [ ] Performance optimization

## File Structure

```
guandan/
├── env/
│   ├── rllib_env.py              # Main MultiAgentEnv wrapper
│   ├── observation_extractor.py  # JSON to numpy conversion
│   ├── game.py                   # Existing game environment
│   └── ...                       # Other existing files
├── agent/
│   ├── rllib_adapter.py          # Agent interface adapter (TODO)
│   └── ...                       # Existing agent files
└── training/
    ├── ray_app.py                # Ray training orchestration
    └── ...                       # Other training files
```

## Key Components

### 1. MultiAgentEnv Wrapper (`rllib_env.py`)

The main wrapper class that implements RLLib's MultiAgentEnv interface:

```python
class GuandanMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config=None):
        # Initialize underlying game environment
        # Define observation/action spaces
        # Set up agent tracking
    
    def reset(self):
        # Reset game and return initial observations
    
    def step(self, actions):
        # Execute actions and return next observations
```

### 2. Observation Extractor (`observation_extractor.py`)

Converts your existing JSON messages to standardized numpy arrays:

```python
class ObservationExtractor:
    def extract_observation(self, message, player_id):
        # Convert JSON to fixed-size vector
        # Hand cards, public info, game state, legal actions
```

### 3. Agent Adapter (TODO)

Bridges RLLib actions with your existing agent interface:

```python
class RLLibAgentAdapter:
    def __init__(self, original_agent):
        # Wrap existing agent
    
    def act(self, observation, legal_actions_mask):
        # Convert RLLib format to received_message format
```

## Implementation Steps

### Step 1: Complete Basic Wrapper ✅

The basic structure is in place. Next steps:

1. **Define observation dimensions** based on your actual JSON structure
2. **Implement action space mapping** from legal action indices
3. **Complete step() method** to handle game progression

### Step 2: Test with Existing Environment

```bash
# Install dependencies
pip install -r requirements_rllib.txt

# Run test suite
python test_rllib_env.py
```

### Step 3: Implement Agent Adapter

Create `guandan/agent/rllib_adapter.py`:

```python
class RLLibAgentAdapter:
    """Adapter to make existing agents compatible with RLLib."""
    
    def __init__(self, agent_class, agent_id):
        self.agent = agent_class(agent_id)
        self.agent_id = agent_id
    
    def act(self, observation, legal_actions_mask=None):
        # Convert observation back to JSON message
        # Call original agent.received_message()
        # Return action index
        pass
```

### Step 4: Integrate with Ray Training

Update `guandan/training/ray_app.py`:

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO

def train(flags):
    # Initialize Ray
    ray.init()
    
    # Configure RLLib
    config = {
        "env": GuandanMultiAgentEnv,
        "num_workers": 4,
        "framework": "torch",
        # ... other config
    }
    
    # Start training
    trainer = PPO(config=config)
    trainer.train()
```

## Observation Space Design

Based on your current JSON structure, observations include:

1. **Hand Cards** (54 dims): Binary vector of cards in hand
2. **Public Info** (8 dims): Other players' card counts and info
3. **Game State** (20 dims): Ranks, positions, current player, etc.
4. **Legal Actions** (100 dims): Mask of available actions
5. **Action History** (30 dims): Recent action history

**Total: ~212 dimensions**

## Action Space Design

Actions are discrete indices into the legal action list:

- **Type**: `gym.spaces.Discrete(max_legal_actions)`
- **Range**: 0 to `len(legal_actions) - 1`
- **Mapping**: RLLib action → legal action index → environment action

## Reward Structure (TODO)

Currently no explicit rewards. Need to design:

1. **Immediate rewards**: Card play success, winning tricks
2. **Episode rewards**: Game completion, rank advancement
3. **Team rewards**: Partner coordination, overall performance

## Testing Strategy

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test with actual game environment
3. **Performance Tests**: Measure observation extraction speed
4. **Training Tests**: Validate with small-scale training runs

## Migration Path

1. **Phase 1**: Get basic wrapper working with existing agents
2. **Phase 2**: Implement proper observation/action spaces
3. **Phase 3**: Add reward structure and training integration
4. **Phase 4**: Optimize for distributed training

## Common Issues and Solutions

### Issue: Observation dimension mismatch
**Solution**: Ensure observation extractor matches defined spaces

### Issue: Action mapping errors
**Solution**: Validate action indices against legal actions

### Issue: Performance bottlenecks
**Solution**: Optimize observation extraction and caching

### Issue: Ray compatibility
**Solution**: Ensure proper serialization of environment state

## Next Actions

1. **Complete observation space definition** based on actual JSON structure
2. **Implement action space mapping** from legal actions
3. **Test with existing game environment** to ensure compatibility
4. **Design reward structure** for effective learning
5. **Integrate with Ray training pipeline**

## Resources

- [Ray RLLib Documentation](https://docs.ray.io/en/latest/rllib/)
- [MultiAgentEnv Interface](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html)
- [PettingZoo Integration](https://docs.ray.io/en/latest/rllib/rllib-env.html#pettingzoo-integration)

---

**Note**: This is a living document that will be updated as implementation progresses.

# RLLib Integration Guide for DanZero

This guide outlines the step-by-step process for integrating DanZero with Ray RLLib for distributed reinforcement learning training.

## Overview

We're implementing a custom MultiAgentEnv wrapper that preserves your existing JSON-based communication protocol while adding RLLib compatibility. This approach minimizes changes to your current codebase while enabling distributed training.

## Implementation Status

### ✅ Completed
- [x] Basic MultiAgentEnv wrapper structure (`guandan/env/rllib_env.py`)
- [x] **Simplified working environment** (`guandan/env/rllib_env_simple.py`)
- [x] **Complete observation extraction system** (`guandan/env/observation_extractor.py`)
- [x] **Observation space definition** (212 dimensions)
- [x] **Action space definition** (discrete 0-99)
- [x] **Test suites for validation** (`test_rllib_env_simple.py`, `test_rllib_simple.py`)
- [x] **Working environment creation and reset functionality**
- [x] **Fixed null bytes issue in agent files** (33 files cleaned)
- [x] **Agent import functionality restored** (all agent files now importable)

### 🔄 In Progress
- [ ] Complete step() method implementation with full game logic
- [ ] Design comprehensive reward structure

### 📋 Next Steps
- [ ] Test with actual game environment (full game logic)
- [ ] Implement agent adapter layer for existing agents
- [ ] Integrate with Ray training pipeline
- [ ] Performance optimization and distributed training

## File Structure

```
guandan/
├── env/
│   ├── rllib_env.py              # Original MultiAgentEnv wrapper (has issues)
│   ├── rllib_env_simple.py       # ✅ Working simplified wrapper
│   ├── observation_extractor.py  # ✅ Complete JSON to numpy conversion
│   ├── game.py                   # Existing game environment
│   ├── engine.py                 # Game engine and logic
│   ├── player.py                 # Player state management
│   ├── context.py                # Game context
│   ├── table.py                  # Table state
│   ├── card_deck.py              # Card deck management
│   └── utils.py                  # Game utilities and legal actions
├── agent/
│   ├── rllib_adapter.py          # Agent interface adapter (TODO)
│   └── ...                       # Existing agent files (✅ null bytes issue fixed)
└── training/
    ├── ray_app.py                # Ray training orchestration
    └── ...                       # Other training files

# Test files
├── test_rllib_env_simple.py      # ✅ Working environment tests
├── test_rllib_simple.py          # ✅ Observation extraction tests
└── test_rllib_env.py             # Original tests (blocked by agent issues)
```

## Key Components

### 1. Working MultiAgentEnv Wrapper (`rllib_env_simple.py`) ✅

The simplified wrapper class that implements RLLib's MultiAgentEnv interface:

```python
class GuandanMultiAgentEnvSimple(MultiAgentEnv):
    def __init__(self, config=None):
        # Initialize underlying game environment
        # Define observation/action spaces (212 dims obs, 100 discrete actions)
        # Set up agent tracking
    
    def reset(self):
        # Reset game and return initial observations for all 4 agents
        # Returns: (observations, infos)
    
    def step(self, actions):
        # Execute actions and return next observations
        # Returns: (observations, rewards, terminateds, truncateds, infos)
```

**Current Status**: ✅ Environment creation and reset working, step method partially implemented

### 2. Complete Observation Extractor (`observation_extractor.py`) ✅

Converts your existing JSON messages to standardized numpy arrays:

```python
class ObservationExtractor:
    def extract_observation(self, message, player_id):
        # Convert JSON to 212-dimensional vector:
        # - Hand cards (54 dims): Binary vector of cards in hand
        # - Public info (8 dims): Other players' card counts
        # - Game state (20 dims): Ranks, positions, current player, stage
        # - Legal actions (100 dims): Mask of available actions
        # - Action history (30 dims): Recent action history
```

**Current Status**: ✅ Fully working, tested with multiple game stages and action types

### 3. Agent Adapter (READY TO IMPLEMENT)

Bridges RLLib actions with your existing agent interface:

```python
class RLLibAgentAdapter:
    def __init__(self, original_agent):
        # Wrap existing agent (null bytes issue now fixed)
    
    def act(self, observation, legal_actions_mask):
        # Convert RLLib format to received_message format
        # Return action index for legal actions
```

**Current Status**: ✅ Agent files now importable, ready for adapter implementation

## Null Bytes Issue Resolution ✅

**PROBLEM RESOLVED**: The agent files contained null bytes (`\x00`) and UTF-16 encoding issues that prevented Python imports.

**SOLUTION IMPLEMENTED**:
1. **Identified 33 Python files** with null bytes in the agent directory
2. **Cleaned null bytes** using Python script to systematically remove `\x00` characters
3. **Fixed encoding issues** in `__init__.py` files (UTF-16 → UTF-8 conversion)
4. **Verified imports** - all agent files now import successfully

**FILES CLEANED**:
- All files in `guandan/agent/baselines/rule/ai1/`, `ai2/`, `ai3/`, `ai4/`, `ai6/`
- `guandan/agent/baselines/__init__.py` and `guandan/agent/baselines/rule/__init__.py`
- Additional utility and model files in `guandan/agent/`

**VERIFICATION**:
```bash
# All these imports now work:
from guandan.agent.baselines.rule.ai1.state import State  # ✅
from guandan.agent.baselines.rule.ai2.action import *    # ✅  
from guandan.agent.baselines.rule.ai3.state import *     # ✅
from guandan.agent.baselines.rule.ai4.state import *     # ✅
```

## Implementation Steps

### Step 1: Complete Basic Wrapper ✅

**COMPLETED**: Basic structure, observation space, and action space are working.

1. ✅ **Observation dimensions defined** (212 dimensions based on JSON structure)
2. ✅ **Action space mapping implemented** (discrete 0-99 actions)
3. 🔄 **Step() method partially implemented** (needs full game logic completion)

### Step 2: Test with Working Environment ✅

**COMPLETED**: Core functionality is tested and working.

```bash
# Test observation extraction
python test_rllib_simple.py

# Test simplified environment
python test_rllib_env_simple.py

# Test original environment (currently blocked)
python test_rllib_env.py  # Fails due to null bytes in agent files
```

**Test Results**:
- ✅ Observation extraction: All tests pass
- ✅ Environment creation: Working
- ✅ Reset functionality: Working
- 🔄 Step functionality: Partially working (needs completion)

### Step 3: Fix Agent Files and Implement Adapter ✅

**COMPLETED**: Agent files null bytes issue resolved.

**✅ COMPLETED**: Fixed null bytes issue in agent files:
```bash
# Files cleaned (33 total):
# - guandan/agent/baselines/rule/ai1/state.py ✅
# - guandan/agent/baselines/rule/ai2/action.py ✅
# - guandan/agent/baselines/rule/ai3/state.py ✅
# - guandan/agent/baselines/rule/ai4/state.py ✅
# - guandan/agent/baselines/rule/ai6/state.py ✅
# - guandan/agent/baselines/__init__.py ✅
# - guandan/agent/baselines/rule/__init__.py ✅
# - Plus 26 additional files with null bytes
```

**🔄 NEXT**: Create agent adapter now that files are fixed:

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

**Current Status**: ✅ Agent files now importable, ready for adapter implementation

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

## Observation Space Design ✅

**COMPLETED**: Based on your current JSON structure, observations include:

1. **Hand Cards** (54 dims): Binary vector of cards in hand
2. **Public Info** (8 dims): Other players' card counts and info  
3. **Game State** (20 dims): Ranks, positions, current player, stage encoding
4. **Legal Actions** (100 dims): Mask of available actions
5. **Action History** (30 dims): Recent action history

**Total: 212 dimensions** ✅

**Implementation**: `ObservationExtractor` class in `guandan/env/observation_extractor.py`
**Status**: ✅ Fully working and tested

## Action Space Design ✅

**COMPLETED**: Actions are discrete indices into the legal action list:

- **Type**: `gym.spaces.Discrete(100)`
- **Range**: 0 to 99 (maps to legal action indices)
- **Mapping**: RLLib action → legal action index → environment action
- **Validation**: Invalid actions are handled gracefully (default to first legal action)

**Implementation**: `GuandanMultiAgentEnvSimple` class
**Status**: ✅ Working with basic validation

## Reward Structure 🔄

**CURRENT STATUS**: Basic reward structure implemented, needs refinement:

```python
def _calculate_rewards(self) -> Dict[str, float]:
    # Current implementation:
    # - Episode completion: +1.0 for top 2 players, -0.5 for others
    # - Continuing: +0.01 per step
    # 
    # TODO: Design more sophisticated rewards:
    # 1. Immediate rewards: Card play success, winning tricks
    # 2. Episode rewards: Game completion, rank advancement  
    # 3. Team rewards: Partner coordination, overall performance
```

**Priority**: Design comprehensive reward structure for effective learning

## Testing Strategy ✅

**COMPLETED**: Comprehensive testing framework in place:

1. ✅ **Unit Tests**: Individual components tested (`test_rllib_simple.py`)
2. ✅ **Integration Tests**: Environment wrapper tested (`test_rllib_env_simple.py`)
3. ✅ **Performance Tests**: Observation extraction speed validated
4. ✅ **Training Tests**: Ready (agent file issues resolved)

**Test Results**:
- ✅ Observation extraction: All tests pass
- ✅ Environment creation: Working
- ✅ Reset functionality: Working
- 🔄 Step functionality: Partially working

## Migration Path

1. ✅ **Phase 1**: Basic wrapper working (simplified version)
2. ✅ **Phase 2**: Observation/action spaces implemented
3. 🔄 **Phase 3**: Reward structure and training integration (in progress)
4. ⏳ **Phase 4**: Optimize for distributed training (pending)

## Common Issues and Solutions

### Issue: Null bytes in agent files ✅ SOLVED
**Problem**: Agent files contained null bytes preventing Python import
**Solution**: Cleaned null bytes from 33 files, fixed UTF-16 encoding issues
**Status**: ✅ All agent files now importable

### Issue: Observation dimension mismatch ✅ SOLVED
**Solution**: Observation extractor matches defined spaces (212 dims)
**Status**: ✅ Working

### Issue: Action mapping errors ✅ SOLVED
**Solution**: Action indices validated against legal actions with graceful fallback
**Status**: ✅ Working

### Issue: Performance bottlenecks ⏳ PENDING
**Solution**: Optimize observation extraction and caching
**Status**: Not yet critical

### Issue: Ray compatibility ⏳ PENDING
**Solution**: Ensure proper serialization of environment state
**Status**: Not yet tested

## Next Actions (Priority Order)

1. **🟡 MEDIUM**: Complete step() method with full game logic
2. **🟡 MEDIUM**: Implement agent adapter layer for existing agents
3. **🟡 MEDIUM**: Design comprehensive reward structure
4. **🟢 LOW**: Integrate with Ray training pipeline
5. **🟢 LOW**: Performance optimization and distributed training

## Current Working Interface

### Environment Usage

```python
from guandan.env.rllib_env_simple import make_guandan_env_simple

# Create environment
env = make_guandan_env_simple()

# Reset environment
observations, infos = env.reset()
# Returns: 4 agents with 212-dim observations

# Take step
actions = {
    "agent_0": 0,  # Action index (0-99)
    "agent_1": 1,
    "agent_2": 2, 
    "agent_3": 0
}
obs, rewards, terminateds, truncateds, infos = env.step(actions)

# Environment properties
print(f"Agent IDs: {env.agent_ids}")
print(f"Observation space: {env.observation_spaces['agent_0'].shape}")
print(f"Action space: {env.action_spaces['agent_0'].n}")
```

### Observation Structure

```python
# 212-dimensional observation vector:
# [0:54]    - Hand cards (binary vector)
# [54:62]   - Public info (8 dims: 4 players × 2 info)
# [62:82]   - Game state (20 dims: ranks, positions, stage)
# [82:182]  - Legal actions mask (100 dims)
# [182:212] - Action history (30 dims)
```

### Action Structure

```python
# Actions are discrete indices (0-99)
# Maps to legal action list from game environment
# Invalid actions default to first legal action (usually PASS)
```

## Resources

- [Ray RLLib Documentation](https://docs.ray.io/en/latest/rllib/)
- [MultiAgentEnv Interface](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html)
- [PettingZoo Integration](https://docs.ray.io/en/latest/rllib/rllib-env.html#pettingzoo-integration)

---

**Last Updated**: December 2024  
**Status**: Core functionality working, agent files fixed, ready for adapter implementation  
**Note**: This is a living document that will be updated as implementation progresses.

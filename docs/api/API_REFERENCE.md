# API Reference - DanZero Guandan Framework

Available APIs for the DanZero Guandan reinforcement learning framework.

## Table of Contents

1. [Environment API](#environment-api)
2. [Agent API](#agent-api)
3. [Training API](#training-api)
4. [Utility Functions](#utility-functions)
5. [Configuration](#configuration)
6. [Examples](#examples)

## Environment API

### Core Environment Classes

#### `GameEnv`

The main game environment class that handles Guandan game logic.

```python
from guandan.env.engine import GameEnv
from guandan.env.context import Context

# Create game environment
ctx = Context()
env = GameEnv(ctx)

# Initialize game
env.battle_init()

# Get current player
current_player = env.ctx.player_waiting

# Get legal actions for current player
legal_actions = env.get_legal_actions()

# Make a move
env.step(action_index)

# Check if game is done
is_done = env.is_done()

# Get winner information
winner_team = env.ctx.winner_team
game_result = env.ctx.game_result
```

**Key Methods:**

- `battle_init()`: Initialize a new game
- `step(action_index)`: Execute an action
- `get_legal_actions()`: Get available actions for current player
- `is_done()`: Check if game is finished
- `determine_winner()`: Determine game winner based on Guandan rules

#### `GuandanMultiAgentEnv`

RLLib MultiAgentEnv wrapper for distributed training.

```python
from guandan.env.rllib_env import make_guandan_env

# Create environment with default configuration
env = make_guandan_env()

# Create environment with custom agent types
config = {
    'agent_types': {
        'agent_0': 'ai1',
        'agent_1': 'ai2',
        'agent_2': 'ai3',
        'agent_3': 'ai4'
    }
}
env = make_guandan_env(config)

# Environment interface
obs = env.reset()
obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

**Key Methods:**

- `reset()`: Reset environment and return initial observations
- `step(actions)`: Execute actions for all agents
- `render()`: Render current game state
- `close()`: Clean up environment resources

### Observation Space

#### Simple Mode (212 dimensions)

```python
from guandan.env.observation_extractor import extract_observation_simple

# Extract simple observation
obs = extract_observation_simple(json_message, player_id)
print(f"Observation shape: {obs.shape}")  # (212,)
```

**Components:**
- Hand cards (54D)
- Public info (8D)
- Game state (20D)
- Legal actions (100D)
- Action history (30D)

#### Comprehensive Mode (513 dimensions)

```python
from guandan.env.observation_extractor import extract_observation_comprehensive

# Extract comprehensive observation
obs = extract_observation_comprehensive(json_message, player_id)
print(f"Observation shape: {obs.shape}")  # (513,)
```

**Components:**
- [0-53]: Hand cards (54D)
- [54-107]: Remaining cards (54D)
- [108-161]: Last move to cover (54D)
- [162-215]: Partner last move (54D)
- [216-299]: Card counts for other players (84D)
- [300-461]: Played cards for other players (162D)
- [462-501]: Team levels (40D)
- [501-513]: Wild card flags (12D)

## Agent API

### Agent Registry

```python
from guandan.agent.agents import agent_cls

# Available agent types
print(agent_cls.keys())  # ['ai1', 'ai2', 'ai3', 'ai4', 'ai6', 'torch', 'random']

# Create agent
agent = agent_cls['ai1'](id=0)

# Get action from agent
action_index = agent.received_message(json_message)
```

### Rule-based Agents

#### AI1 Agent

Complex rule-based strategy with comprehensive decision making.

```python
from guandan.agent.agents import agent_cls

agent = agent_cls['ai1'](id=0)
action = agent.received_message(game_message)
```

**Features:**
- Advanced card combination analysis
- Strategic planning
- Partner coordination
- Tribute optimization

#### AI2 Agent

Phased decision-making strategy.

```python
agent = agent_cls['ai2'](id=1)
action = agent.received_message(game_message)
```

**Features:**
- Multi-phase decision process
- Risk assessment
- Adaptive strategy selection

#### AI3 Agent

Experimental/adversarial strategy.

```python
agent = agent_cls['ai3'](id=2)
action = agent.received_message(game_message)
```

**Features:**
- Adversarial play style
- Experimental tactics
- Aggressive strategies

#### AI4 Agent

Alternative heuristic approach.

```python
agent = agent_cls['ai4'](id=3)
action = agent.received_message(game_message)
```

**Features:**
- Alternative evaluation metrics
- Different decision trees
- Complementary strategies

#### AI6 Agent

Additional heuristic strategy.

```python
agent = agent_cls['ai6'](id=0)
action = agent.received_message(game_message)
```

**Features:**
- Specialized tactics
- Unique evaluation functions
- Experimental approaches

### Neural Network Agents

#### Torch Agent

PyTorch-based neural network agent.

```python
agent = agent_cls['torch'](id=0)
action = agent.received_message(game_message)
```

**Features:**
- Deep neural network
- Pre-trained models
- GPU acceleration support

### Random Agent

Baseline random strategy for testing.

```python
agent = agent_cls['random'](id=0)
action = agent.received_message(game_message)
```

## Training API

**Note**: Training components are in development. Basic structure exists but full functionality is not yet implemented.

### Basic Training Structure

```python
# Training components exist but are not fully functional
from guandan.training.ray_app import RayTrainingApp
from guandan.training.parameter_server import ParameterServer
from guandan.training.rollout_worker import RolloutWorker
from guandan.training.learner import Learner

# These classes exist but methods may not be fully implemented
```

## Utility Functions

### Card Utilities

```python
from guandan.env.utils import CardToNum, RANK1, RANK2

# Card to number mapping
card_num = CardToNum['AS']  # 12 (Ace of Spades)

# Rank mappings  
rank1_val = RANK1['A']  # 13
rank2_val = RANK2['A']  # 12

# Legal action generation
from guandan.env.utils import legal_actions
actions = legal_actions(cards_list, ctx)
```

## Configuration

### Environment Configuration

```python
config = {
    'max_steps': 10000,
    'agent_types': {
        'agent_0': 'ai1',
        'agent_1': 'ai2',
        'agent_2': 'ai3',
        'agent_3': 'ai4'
    },
    'observation_mode': 'comprehensive',
    'observation_config': {
        'components': {
            'hand_cards': True,
            'public_info': True,
            'game_state': True,
            'legal_actions': True,
            'action_history': True,
            'remaining_cards': True,
            'last_move_to_cover': True,
            'partner_last_move': True,
            'card_counts': True,
            'played_cards': True,
            'team_levels': True,
            'wild_flags': True
        }
    }
}
```

### Training Configuration

```python
training_config = {
    'xpid': 'danzero_experiment',
    'total_frames': 10000000,
    'num_actors': 4,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon': 0.1,
    'replay_buffer_size': 1000000,
    'target_update_freq': 1000
}
```

## Examples

### Basic Game Loop

```python
from guandan.env.engine import GameEnv
from guandan.env.context import Context
from guandan.agent.agents import agent_cls

# Initialize game
ctx = Context()
env = GameEnv(ctx)
env.battle_init()

# Create agents
agents = {
    0: agent_cls['ai1'](id=0),
    1: agent_cls['ai2'](id=1),
    2: agent_cls['ai3'](id=2),
    3: agent_cls['ai4'](id=3)
}

# Game loop
while not env.is_done():
    current_player = env.ctx.player_waiting
    agent = agents[current_player]
    
    # Get legal actions
    legal_actions = env.get_legal_actions()
    
    # Get action from agent
    action = agent.received_message(env.get_game_message())
    
    # Execute action
    env.step(action)

# Get results
winner_team = env.ctx.winner_team
print(f"Winner: {winner_team}")
```

### RLLib Environment

```python
from guandan.env.rllib_env import make_guandan_env

# Create environment
env = make_guandan_env()

# Basic usage
obs = env.reset()
actions = {agent_id: env.action_space.sample() for agent_id in env.agent_ids}
obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

## Contributing

When extending the API:

1. **Follow naming conventions**: Use descriptive names
2. **Add type hints**: Include type annotations
3. **Write docstrings**: Document all public methods
4. **Add tests**: Include unit tests for new functionality
5. **Update documentation**: Keep this API doc current

## License

Apache License 2.0 - see LICENSE file for details.

# User Guide - DanZero Guandan Framework

A comprehensive guide for users of the DanZero Guandan reinforcement learning framework.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Environment Configuration](#environment-configuration)
5. [Agent Types](#agent-types)
6. [Tournament Testing](#tournament-testing)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

## Quick Start

### Basic Environment Usage

```python
from guandan.env.rllib_env import make_guandan_env

# Create environment with default configuration
env = make_guandan_env()

# Reset and get observations
obs = env.reset()
print(f"Observation shape: {obs[list(obs.keys())[0]].shape}")

# Step through environment
actions = {agent_id: 0 for agent_id in env.agent_ids}
obs, rewards, terminateds, truncateds, infos = env.step(actions)
```

### Custom Agent Configuration

```python
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
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Ray 2.0+

### Setup

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
```

### Verify Installation

```bash
# Test basic functionality
python -c "from guandan.env.rllib_env import make_guandan_env; env = make_guandan_env(); print('Environment ready!')"
```

## Basic Usage

### Environment Interface

```python
from guandan.env.rllib_env import make_guandan_env

# Create environment
env = make_guandan_env()

# Environment properties
print(f"Agent IDs: {env.agent_ids}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Reset environment
obs = env.reset()
print(f"Initial observations: {list(obs.keys())}")

# Step through environment
actions = {agent_id: env.action_space.sample() for agent_id in env.agent_ids}
obs, rewards, terminateds, truncateds, infos = env.step(actions)

# Check if episode is done
done = all(terminateds.values()) or all(truncateds.values())
```

### Agent Usage

```python
from guandan.agent.agents import agent_cls

# Create agents
agents = {
    0: agent_cls['ai1'](id=0),
    1: agent_cls['ai2'](id=1),
    2: agent_cls['ai3'](id=2),
    3: agent_cls['ai4'](id=3)
}

# Get action from agent
action = agents[0].received_message(game_message)
```

## Environment Configuration

### Observation Modes

#### Simple Mode (212 dimensions)
```python
config = {'observation_mode': 'simple'}
env = make_guandan_env(config)
```

#### Comprehensive Mode (513 dimensions)
```python
config = {'observation_mode': 'comprehensive'}
env = make_guandan_env(config)
```

### Custom Observation Components

```python
config = {
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
env = make_guandan_env(config)
```

## Agent Types

### Rule-based Agents

| Agent | Description | Best For |
|-------|-------------|----------|
| **ai1** | Complex rule-based strategy | Main baseline, most sophisticated |
| **ai2** | Phased decision-making | Strategic planning |
| **ai3** | Experimental/adversarial | Testing and experimentation |
| **ai4** | Alternative heuristic | Complementary approach |
| **ai6** | Specialized strategy | Specific tactics |

### Neural Network Agents

| Agent | Description | Requirements |
|-------|-------------|--------------|
| **torch** | PyTorch-based neural network | GPU recommended |
| **random** | Random baseline | Testing and comparison |

### Agent Usage Examples

```python
from guandan.agent.agents import agent_cls

# Create different agent types
ai1_agent = agent_cls['ai1'](id=0)
ai2_agent = agent_cls['ai2'](id=1)
torch_agent = agent_cls['torch'](id=2)
random_agent = agent_cls['random'](id=3)

# Get actions
action1 = ai1_agent.received_message(message)
action2 = ai2_agent.received_message(message)
action3 = torch_agent.received_message(message)
action4 = random_agent.received_message(message)
```

## Tournament Testing

### Basic Tournament

```bash
# Run default tournament (100 games)
python agent_vs_agent_test.py

# Run with custom parameters
python agent_vs_agent_test.py --num_games 50 --max_steps 10000
```

### Tournament Configuration

```python
from agent_vs_agent_test import AgentTournament

# Create tournament with custom configuration
tournament = AgentTournament(
    num_games=100,
    max_steps=10000,
    agent_config={
        'agent_0': 'ai1',
        'agent_1': 'ai2',
        'agent_2': 'ai3',
        'agent_3': 'ai4'
    }
)

# Run tournament
results = tournament.run_tournament()

# Print results
print(f"Team 1 wins: {results['team1_wins']}")
print(f"Team 2 wins: {results['team2_wins']}")
print(f"Draws: {results['draws']}")
```

## Troubleshooting

### Common Issues

#### Environment Not Initializing
```python
# Check environment creation
try:
    env = make_guandan_env()
    print("Environment created successfully")
except Exception as e:
    print(f"Error: {e}")
    # Check dependencies
    import guandan.env.engine
    import guandan.env.utils
```

#### Observation Shape Mismatch
```python
# Check observation mode
config = {'observation_mode': 'simple'}  # or 'comprehensive'
env = make_guandan_env(config)

# Verify observation shape
obs = env.reset()
print(f"Observation shape: {obs[list(obs.keys())[0]].shape}")

# Expected shapes:
# Simple mode: (212,)
# Comprehensive mode: (513,)
```

#### Action Space Issues
```python
# Check legal actions before executing
legal_actions = env.get_legal_actions()
print(f"Legal actions: {legal_actions}")

# Verify action is legal
if action in legal_actions:
    env.step(action)
else:
    print(f"Action {action} not legal")
```

### Debug Mode

```python
# Enable debug mode
env = make_guandan_env({'debug': True})

# Print detailed game state
print(env.get_game_state())

# Print legal actions with details
legal_actions = env.get_legal_actions()
print(f"Legal actions: {legal_actions}")
```

## FAQ

### Q: What's the difference between simple and comprehensive observation modes?

**A**: Simple mode uses 212 dimensions and is faster, while comprehensive mode uses 513 dimensions and follows the paper specification exactly. Use simple mode for faster training, comprehensive mode for research.

### Q: How do I choose the right agent type?

**A**: 
- `ai1`: Most sophisticated rule-based agent
- `ai2`: Phased decision-making strategy
- `ai3`: Experimental/adversarial approach
- `ai4`: Alternative heuristic method
- `ai6`: Additional specialized strategy
- `torch`: Neural network agent (requires GPU)
- `random`: Baseline for testing

### Q: Why is my training not improving?

**A**: Check:
1. Learning rate (try 0.001, 0.0001, 0.00001)
2. Reward scaling
3. Observation mode
4. Batch size
5. Number of actors
6. Model architecture

### Q: How do I handle out of memory errors?

**A**: 
1. Reduce batch size
2. Use simple observation mode
3. Reduce number of actors
4. Clear unused variables
5. Use CPU instead of GPU

### Q: Can I use custom observation spaces?

**A**: Yes, you can customize observation components in the configuration:

```python
config = {
    'observation_config': {
        'components': {
            'hand_cards': True,
            'public_info': True,
            'game_state': True,
            'legal_actions': True,
            'action_history': False,  # Disable if not needed
            # ... other components
        }
    }
}
```

### Q: How do I add new agents?

**A**: 
1. Create agent class in `guandan/agent/baselines/`
2. Register in `guandan/agent/agents.py`
3. Add tests in `tests/test_agents.py`
4. Update documentation

### Q: What's the best way to debug agent behavior?

**A**: 
1. Enable debug mode
2. Use `get_action_info()` method
3. Print legal actions
4. Monitor agent's decision process
5. Use tournament testing

### Q: How do I optimize training performance?

**A**: 
1. Use simple observation mode
2. Optimize batch size
3. Use appropriate number of actors
4. Enable GPU if available
5. Profile and optimize bottlenecks

## Getting Help

### Before Asking for Help

1. **Check this guide** for common solutions
2. **Search existing issues** on GitHub
3. **Run diagnostic commands**:
   ```bash
   python -c "import guandan; print('Import successful')"
   python -c "from guandan.env.rllib_env import make_guandan_env; env = make_guandan_env(); print('Environment ready')"
   ```
4. **Check logs** for error messages
5. **Try minimal examples** to isolate the issue

### When Creating an Issue

Include:
1. **Error message** (full traceback)
2. **Steps to reproduce**
3. **Environment details**:
   - Python version
   - Operating system
   - Package versions
4. **Expected vs actual behavior**
5. **Minimal code example**

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Comprehensive guides and API docs
- **Examples**: Code examples and tutorials

## License

Apache License 2.0 - see LICENSE file for details.

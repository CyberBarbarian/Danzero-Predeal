# Project Structure - DanZero Guandan AI Framework

This repository contains a comprehensive reinforcement learning framework for the Guandan card game, evolved from the original DouZero Doudizhu codebase. The project focuses on building scalable distributed training systems and multiple baseline agent strategies.

## Overview

The project is structured around three main components:

- **Game Environment**: Complete Guandan game logic and state management
- **Agent System**: Multiple rule-based and learning-based AI strategies  
- **Training Framework**: Distributed reinforcement learning pipeline using Ray

## Directory Structure

### Root Level

```text
DanZero/
├── train.py                    # Main training entry point
├── setup.py                    # Package configuration
├── requirements.txt            # Python dependencies
├── requirements_lock.txt       # Locked dependency versions
├── PROJECT_STRUCTURE.md        # This documentation
├── README.md                   # Project overview and usage
├── RLLIB_INTEGRATION_GUIDE.md  # RLLib integration documentation
├── LICENSE                     # Apache 2.0 license
├── actor.sh                    # Training script
├── get_most_recent.sh          # Model retrieval script
├── kill.sh                     # Process termination script
├── logs/                        # Game log files directory
├── most_recent_model/          # Model checkpoint directory
├── Danvenv/                    # Python virtual environment
├── archive/                    # Legacy Doudizhu code archive
└── guandan/                    # Main package directory
```

### Main Package (`guandan/`)

#### Configuration

- **`config.py`** - Centralized training parameters and hyperparameters for DanZero framework

#### DeepMind Control Integration (`dmc/`)

```text
dmc/
└── (empty)              # DeepMind Control integration (placeholder)
```

#### Game Environment (`env/`)

```text
env/
├── __init__.py
├── game.py                    # Main game environment and self-play driver
├── engine.py                  # Game rules, stage flow, and state transitions
├── card_deck.py               # Two-deck card generation and dealing logic
├── player.py                  # Player state and data structures
├── context.py                 # Game context and shared state
├── table.py                   # Table state management
├── utils.py                   # Card pattern analysis and legal action generation
├── move_detector.py           # Move validation and detection
├── move_generator.py          # Legal move generation
├── move_selector.py           # Move selection utilities
├── observation_extractor.py   # JSON to numpy observation conversion (212-dim)
└── rllib_env.py              # RLLib MultiAgentEnv wrapper (complete with agents)
```

#### Agent System (`agent/`)

```text
agent/
├── __init__.py
├── agents.py            # Agent registry and factory
├── random_agent.py      # Random baseline strategy
├── ai1/                 # Complex rule-based strategy (migrated)
├── ai2/                 # Phased decision-making strategy (migrated)
├── ai3/                 # Experimental/adversarial strategy (migrated)
├── ai4/                 # Alternative heuristic approach (migrated)
├── ai6/                 # Additional heuristic strategy (migrated)
├── mc/                  # Monte Carlo baseline (legacy)
├── baselines/           # Rule-based AI strategies
│   ├── __init__.py
│   ├── README.md        # Baseline strategies documentation
│   ├── rule/            # Rule-based AI implementations
│   └── legacy/          # Archived baseline implementations
│       └── mc/          # Monte Carlo baseline (archived)
└── torch/               # Neural network-based agents
    ├── actor.py         # Actor network implementation
    ├── client.py        # Torch agent client
    ├── model.py         # Neural network models
    ├── util.py          # Utility functions
    ├── ppo20000.pth     # Pre-trained PPO model
    └── q_network.ckpt   # Pre-trained Q-network model
```

#### Training Framework (`training/`)

```text
training/
├── __init__.py
├── ray_app.py           # Ray-based training orchestrator
├── parameter_server.py  # Distributed parameter server
├── rollout_worker.py    # Self-play data collection worker
├── learner.py           # Model training and optimization
├── checkpoint.py        # Model checkpoint management
└── logger.py            # Training metrics and logging
```

### Archive (`archive/`)

```text
archive/
└── doudizhu/            # Original DouZero Doudizhu implementation
    ├── README_doudizhu_original.md
    ├── evaluate.py
    ├── ADP_test.py
    ├── generate_eval_data.py
    ├── sl_test.py
    ├── test.py
    └── most_recent_model/
        ├── landlord.ckpt
        ├── landlord_up.ckpt
        └── landlord_down.ckpt
```

## Key Components

### Game Environment

- **Complete Guandan Implementation**: Four-player team-based card game with tribute/return mechanics
- **State Management**: Comprehensive game state tracking including hand cards, table state, and game phases
- **Action Space**: Legal move generation and validation for all card combinations
- **Rule Engine**: Stage-based game flow with level progression and tribute mechanics

### Agent Strategies

- **Rule-based Baselines**: Multiple heuristic strategies (ai1-ai6) with different decision-making approaches
- **Random Agent**: Simple random baseline for sanity checking
- **Neural Network Agents**: Legacy PPO and Q-network implementations
- **Monte Carlo**: Archived Monte Carlo tree search baseline

### Training Framework

- **Ray Integration**: Distributed training pipeline for scalable self-play
- **Parameter Server**: Centralized model parameter management
- **Rollout Workers**: Parallel data collection from self-play games
- **Learner**: Model training and optimization with experience replay

## Dependencies

### Core Dependencies

- **PyTorch**: Deep learning framework for neural network agents
- **Ray**: Distributed computing framework for training
- **RLCard**: Card game environment utilities
- **GitPython**: Version control integration

### Development Dependencies

- **Dill**: Advanced serialization for Python objects
- **Lightning Utilities**: PyTorch Lightning utilities
- **Opt-Einsum**: Optimized Einstein summation operations

## Architecture Overview

```text
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Training      │    │   Game           │    │   Agent         │
│   Framework     │    │   Environment    │    │   System        │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Ray App       │◄──►│ • Game Engine    │◄──►│ • Rule Agents   │
│ • Parameter     │    │ • State Mgmt     │    │ • Neural Agents │
│   Server        │    │ • Action Space   │    │ • Random Agent  │
│ • Rollout       │    │ • Move Detection │    │ • Agent Registry│
│   Workers       │    │ • Card Deck      │    │                 │
│ • Learner       │    │ • Player Context │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │              ┌──────────────────┐
         │              │   RLLib          │
         └──────────────►│   Integration    │
                        ├──────────────────┤
                        │ • MultiAgentEnv  │
                        │ • Observation    │
                        │   Extractor      │
                        │ • Test Suites    │
                        └──────────────────┘
```

## Development Status

### Completed

- ✅ Guandan game environment implementation
- ✅ Rule-based agent strategies (ai1-ai6)
- ✅ Agent registry and factory system
- ✅ RLLib MultiAgentEnv integration
- ✅ Paper-compliant observation space (513-dimensional)
- ✅ Win/loss logic integration in game core
- ✅ Tribute system with proper limits (10 points max)
- ✅ Rank progression and upgrade mechanics
- ✅ Tournament system with proper win detection

### In Progress

- 🔄 BaseAgent interface unification
- 🔄 Training framework implementation
- 🔄 Performance optimization and testing

### Planned

- 📋 Complete training pipeline
- 📋 Model evaluation and benchmarking
- 📋 Advanced training strategies

## Usage

### Quick Start

#### Traditional Agent Usage

```python
from guandan.agent.agents import agent_cls

# Create a rule-based agent
agent = agent_cls['ai1'](id=0)

# Process game message and get action
action_index = agent.received_message(game_message)
```

#### RLLib Environment Usage

```python
from guandan.env.rllib_env import GuandanMultiAgentEnv

# Create environment
env = GuandanMultiAgentEnv()

# Reset and get observations
obs = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Step through environment
action = env.action_space.sample()  # Random action
obs, rewards, dones, infos = env.step(action)
```

### Training

```bash
# Start distributed training
python train.py --xpid danzero_experiment --total_frames 10000000
```

### Testing

```bash
# Test RLLib environment
python -c "from guandan.env.rllib_env import make_guandan_env; env = make_guandan_env(); print('Environment ready!')"

# View game logs
ls logs/
```

## Contributing

This project welcomes contributions in:

- Guandan rule implementations and edge cases
- New agent strategies and algorithms
- Training pipeline optimizations
- Performance optimizations
- Documentation and examples

## License

Apache License 2.0 - see LICENSE file for details.

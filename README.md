# DanZero: Guandan Reinforcement Learning Framework

A scalable distributed reinforcement learning framework for Guandan card game, evolved from DouZero codebase.

## 🚀 Quick Start

```bash
# 1) Activate your environment (project default)
source /mnt/project_modelware/lizikang/Danvenv/bin/activate

# 2) Ensure project is on PYTHONPATH
export PYTHONPATH=$(pwd)

# 3) GPU-optimized DMC training
python scripts/train_dqmc_gpu.py --config configs/dqmc_large.yaml --episodes 10 --batch_size 1024

# 4) Run tournament between agents
python agent_vs_agent_test.py
```

## 📚 Documentation

All documentation is organized in the `docs/` directory:

| Document | Description | Audience |
|----------|-------------|----------|
| **[Project Summary](docs/PROJECT_SUMMARY.md)** | Comprehensive project overview | All users |
| **[User Guide](docs/user/USER_GUIDE.md)** | Complete user guide with examples | End users |
| **[Developer Guide](docs/developer/DEVELOPER_GUIDE.md)** | Development and contribution guide | Developers |
| **[API Reference](docs/api/API_REFERENCE.md)** | Complete API documentation | All users |
| **[Technical Reference](docs/reference/TECHNICAL_REFERENCE.md)** | Technical specifications and implementation details | Developers |
| **[Logging Improvements](docs/LOGGING_IMPROVEMENTS.md)** | Centralized logging and usage | All users |

## ✨ Features

- **Complete Guandan Environment**: Full game logic with RLLib MultiAgentEnv integration
- **Paper-Compliant Observations**: 513-dimensional observation space following research paper specification
- **Rule-based Agents**: Multiple AI strategies (ai1-ai6) with unified interface
- **Training Framework**: DMC training loop with replay buffer and learner (baseline); Ray integration scaffolds
- **Tournament System**: Agent vs agent testing with proper win/loss detection
- **Zero Draws**: Proper Guandan rules implementation with clear winners

## 🎯 Current Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Environment** | ✅ Complete | Guandan game engine with RLLib MultiAgentEnv wrapper |
| **Rule-based Agents** | ✅ Complete | ai1-ai6 agents with basic interface |
| **Game Rules** | ✅ Complete | Full Guandan rules with win/loss logic |
| **Tribute System** | ✅ Complete | Tribute/return mechanics (10pt max) |
| **Rank Progression** | ✅ Complete | Level progression mechanics |
| **DMC Training Pipeline** | ✅ Complete | Paper-compliant DMC with GPU optimization |
| **GPU Optimization** | ✅ Complete | Batched action selection, mixed precision, GPU data pipeline |
| **Agent Interface** | ✅ Complete | BaseAgent unification implemented |

## 🏗️ Architecture

### Environment

- **Game Engine**: Complete Guandan game logic with 4-player support
- **RLLib Integration**: MultiAgentEnv wrapper with paper-compliant observations
- **Agent Adapter**: Bridges rule-based agents with RLLib interface

### Agents

- **Rule-based**: ai1, ai2, ai3, ai4, ai6 - Heuristic strategies
- **Neural Network**: torch - PyTorch-based agents
- **Random**: Random baseline for testing

### Training (guandan/training)

- `actor_policy.py`: ε-greedy over Q(τ,a) for legal action selection
- `rollout_worker.py`: self-play rollout, sample collection with terminal rewards
- `rollout_worker_gpu.py`: GPU-optimized rollout with batched action selection
- `replay_buffer.py`: typed storage and minibatch sampling
- `gpu_pipeline.py`: GPU-optimized data pipeline with tensor buffers
- `learner.py`: DMC update with mixed precision training and gradient clipping
- `loop.py`: single-process training loop + checkpointing
- `loop_gpu.py`: GPU-optimized training loop with batched operations
- `logger.py`, `checkpoint.py`: lightweight logging and checkpoints

## 🧪 Testing

```bash
# Quick test (2 minutes, 20 iterations)
bash testing/scripts/train_test.sh

# Test environment
python -c "from guandan.env.rllib_env import make_guandan_env; env = make_guandan_env(); print('Environment ready!')"

# Run tournament
python agent_vs_agent_test.py

# Run unit tests
python -m pytest testing/unit_tests/
```

## 🤝 Contributing

See [Developer Guide](docs/developer/DEVELOPER_GUIDE.md) for detailed contribution guidelines.

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

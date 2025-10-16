# DanZero: Complete Project Documentation

## 🎯 Project Overview
DanZero is a scalable distributed reinforcement learning framework for the Guandan card game, evolved from the DouZero codebase. The project implements Deep Monte Carlo (DMC) learning with Ray RLlib integration for distributed training.

## 🚀 Quick Start
```bash
# Activate environment
source /mnt/project_modelware/lizikang/Danvenv/bin/activate

# Quick test (2 minutes)
bash testing/scripts/train_test.sh

# Full training (11 hours)
bash train_production.sh

# Analyze results
python analyze_results.py results/danzero_production_*/training_stats.json
```

## 📁 Project Structure

### Root Directory (Production Files)
```
DanZero/
├── README.md                    # Main project documentation
├── LICENSE                      # Apache 2.0 license
├── requirements.txt            # Python dependencies
├── requirements_lock.txt       # Locked dependency versions
├── setup.py                    # Package setup
├── train_production.sh         # Main production training script
├── train.py                    # Legacy training script
├── analyze_results.py          # Results analysis script
├── evaluate_checkpoint.py      # Checkpoint evaluation script
├── guandan/                    # Main game environment and RLlib integration
├── configs/                    # Training configuration files
├── scripts/                    # Production scripts and utilities
├── docs/                       # Comprehensive documentation
├── testing/                    # Test scripts, logs, and outputs
├── checkpoints/                # Production model checkpoints
├── logs/                       # Production training logs
└── results/                    # Production training results
```

### Testing Directory Structure
```
testing/
├── README.md                   # Testing documentation
├── scripts/                    # Test scripts
│   ├── train_test.sh          # Quick test (20 iterations)
│   ├── cleanup_test_files.sh  # Test cleanup script
│   ├── debug_metrics.py       # Debug metrics collection
│   ├── test_dmc_components.py # DMC component testing
│   └── run_two_gpus_1000.py   # Two GPU test script
├── unit_tests/                 # Unit tests (pytest)
│   ├── test_epsilon_scheduler.py
│   ├── test_learner_update.py
│   ├── test_rllib_env_and_model.py
│   └── test_rollout_multistep.py
├── logs/                       # Test logs and outputs
└── outputs/                    # Test outputs
```

## 🎮 Game Implementation

### Core Components
- **Complete Guandan Environment:** Full game logic with RLLib MultiAgentEnv integration
- **Paper-Compliant Observations:** 513-dimensional observation space following research paper specification
- **Rule-based Agents:** Multiple AI strategies (ai1-ai6) with unified interface
- **Zero Draws:** Proper Guandan rules implementation with clear winners

### State and Action Encoding
- **State τ (513 dimensions):**
  - [0–53]: Current hand cards
  - [54–107]: Remaining cards in deck
  - [108–161]: Last move to beat (zeros if leading)
  - [162–215]: Partner's last move (zeros if pass; -1 if finished)
  - [216–299]: Remaining card counts of three opponents
  - [300–461]: Played cards of the three opponents
  - [462–501]: Team levels (ours/opponent)
  - [501–513]: Wild-card flags and composability

- **Action (54 dimensions):** Card combination encoding with counts {0,1,2} per card index

## 🧠 Training Framework

### DMC Algorithm Implementation
- **Network Architecture:** MLP with 5×512 hidden layers (Tanh activation, orthogonal initialization)
- **Input:** Concatenated state-action vector (567 dimensions: 513 + 54)
- **Output:** Scalar Q-value for state-action pair
- **Action Selection:** ε-greedy over legal actions with decay schedule

### Distributed Training Pipeline
- **Actor:** Self-play rollout with ε-greedy action selection
- **Buffer:** Replay buffer with typed storage and minibatch sampling
- **Learner:** DMC update with mixed precision training and gradient clipping
- **Orchestration:** Single-process training loop with checkpointing

### Reward System
- **Round Rewards:** +3/+2/+1 for winning team based on partner position
- **Global Rewards:** +1/-1 for episode win/loss
- **Tribute Phase:** Heuristic rules (excluded from training data)

## ⚙️ Configuration

### Hardware Setup
- **GPUs:** 2× H100 80GB (1 for learner, CPU-only workers)
- **CPUs:** 192 cores
- **Workers:** 120 workers × 3 environments = 360 parallel environments
- **Memory:** 30GB Ray object store

### Training Parameters
- **Model:** 1.3M parameters (DanZero paper specification)
- **Learning Rate:** 1e-3
- **Batch Size:** 64
- **Epsilon:** 0.2 → 0.05 (decay over 10K steps)
- **Target Update:** Every 1000 steps
- **Checkpoint Frequency:** Every 100 iterations

### Performance Targets
- **Iteration Time:** 3.8-4.2 seconds
- **Throughput:** ~0.24 iter/s (900 iter/hour)
- **GPU Utilization:** 20-30% (environment-bound)
- **Training Duration:** ~11 hours for 10K iterations

## 📊 Current Status

### ✅ Completed Components
- **Environment** - Complete Guandan game engine with RLLib MultiAgentEnv wrapper
- **Rule-based Agents** - ai1-ai6 agents with unified interface
- **Game Rules** - Full Guandan rules with win/loss logic
- **DMC Training** - Paper-compliant DMC with GPU optimization
- **GPU Optimization** - Batched action selection, mixed precision, GPU data pipeline
- **Agent Interface** - BaseAgent unification implemented
- **Distributed Training** - Ray RLlib with actor-learner architecture
- **Checkpointing** - Automatic model saving every 100 iterations

### 🔧 Key Features
- **Scalable Architecture:** Tested up to 300 workers, ready for 8 H100s
- **Production Ready:** Comprehensive logging, monitoring, and checkpointing
- **Paper Compliant:** Follows DanZero research paper specifications
- **GPU Efficient:** Optimized for H100 with TF32 and mixed precision

## 🚀 Usage

### Production Training
```bash
# Quick test first (recommended)
bash testing/scripts/train_test.sh

# Full production training
bash train_production.sh

# Analyze results
python analyze_results.py results/danzero_production_*/training_stats.json
```

### Testing and Development
```bash
# Run unit tests
python -m pytest testing/unit_tests/

# Run component tests
python testing/scripts/test_dmc_components.py

# Clean up test files
bash testing/scripts/cleanup_test_files.sh
```

### Environment Testing
```bash
# Test environment
python -c "from guandan.env.rllib_env import make_guandan_env; env = make_guandan_env(); print('Environment ready!')"

# Run tournament
python guandan/eval/tournament.py
```

## 📈 Scaling

### Current Setup (2 H100s)
- **Workers:** 120
- **Environments:** 360 parallel (3 per worker)
- **Expected Performance:** ~11 hours for 10K iterations

### Scaling to 8 H100s
```bash
# Update configuration in train_production.sh:
NUM_WORKERS=800              # 8× more workers
NUM_ENVS_PER_WORKER=2        # 1600 total envs
RAY_NUM_GPUS=8               # All GPUs
NUM_GPUS=2.0                 # 2 GPUs for distributed learners

# Expected: 5-10× faster training (~1-2 hours for 10K iterations)
```

## 🛠️ Development

### Architecture Overview
- **Game Engine:** Complete Guandan game logic with 4-player support
- **RLLib Integration:** MultiAgentEnv wrapper with paper-compliant observations
- **Agent Adapter:** Bridges rule-based agents with RLLib interface
- **Training Pipeline:** DMC with replay buffer and learner (baseline)

### Key Components
- **`guandan/env/`** - Game environment and RLLib integration
- **`guandan/training/`** - Training pipeline components
- **`guandan/rllib/`** - RLlib algorithms and models
- **`guandan/eval/`** - Evaluation and tournament systems

## 📚 Additional Documentation

- **[User Guide](user/USER_GUIDE.md)** - Complete user guide with examples
- **[Developer Guide](developer/DEVELOPER_GUIDE.md)** - Development and contribution guide
- **[API Reference](api/API_REFERENCE.md)** - Complete API documentation
- **[Logging Improvements](LOGGING_IMPROVEMENTS.md)** - Centralized logging system
- **[Reference Documentation](reference/)** - Technical specifications

## 🤝 Contributing

See [Developer Guide](developer/DEVELOPER_GUIDE.md) for detailed contribution guidelines.

## 📄 License

Apache 2.0 - See [LICENSE](../LICENSE) for details.

---
*Last Updated: October 9, 2025*  
*Status: Production Ready*
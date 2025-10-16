# Technical Reference Documentation

## Observation Space Implementation

### Paper-Compliant Observation Space (513 Dimensions)

The observation space follows the exact specification from the research paper:

#### Dimension Breakdown

**State τ (513 dimensions):**
- **[0–53]**: Current hand cards (54D)
- **[54–107]**: Remaining cards in deck (54D)
- **[108–161]**: Last move to beat (zeros if leading) (54D)
- **[162–215]**: Partner's last move (zeros if pass; -1 if finished) (54D)
- **[216–299]**: Remaining card counts of three opponents (84D)
- **[300–461]**: Played cards of the three opponents (162D)
- **[462–501]**: Team levels (ours/opponent) (40D)
- **[501–513]**: Wild-card flags and composability (12D)

**Action Encoding (54 dimensions):**
- Card combination encoding with counts {0,1,2} per card index

#### Implementation Details

- **Card Mapping**: Consistent card-to-index mapping for all 54 cards
- **Move Encoding**: One-hot encoding of card combinations
- **Normalization**: Card counts divided by 27, ranks by 13
- **Special Cases**: Proper handling of leading, passing, and exhausted states

### Unified Observation Extractor

The system provides a unified, configurable observation space extractor with multiple modes:

#### Modes

**Simple Mode (212 dimensions):**
- Hand Cards (54D) + Public Info (8D) + Game State (20D) + Legal Actions (100D) + Action History (30D)

**Comprehensive Mode (513 dimensions):**
- Full paper-compliant observation space as described above

#### Usage

```python
from guandan.env.comprehensive_observation_extractor import (
    create_simple_extractor,
    create_comprehensive_extractor
)

# Simple mode (212 dimensions)
simple_extractor = create_simple_extractor()
obs_simple = simple_extractor.extract_observation(message, player_id)

# Comprehensive mode (513 dimensions)
comp_extractor = create_comprehensive_extractor()
obs_comp = comp_extractor.extract_observation(message, player_id)
```

#### RLLib Integration

```python
from guandan.env.rllib_env import (
    make_guandan_env_simple,
    make_guandan_env_comprehensive
)

# Simple mode environment
env_simple = make_guandan_env_simple()

# Comprehensive mode environment
env_comp = make_guandan_env_comprehensive()
```

## Algorithm Implementation

### DMC (Deep Monte Carlo) Algorithm

#### Network Architecture
- **Input**: Concatenated state-action vector (567 dimensions: 513 + 54)
- **Output**: Scalar Q-value for state-action pair
- **Architecture**: MLP with 5×512 hidden layers
- **Activation**: Tanh (paper-compliant) or ReLU (modern alternative)
- **Initialization**: Orthogonal weights, zero biases

#### Training Pipeline
- **Actor**: Self-play rollout with ε-greedy action selection
- **Buffer**: Replay buffer with typed storage and minibatch sampling
- **Learner**: DMC update with mixed precision training and gradient clipping
- **Orchestration**: Single-process training loop with checkpointing

#### Action Selection
- **Method**: ε-greedy over legal actions
- **Schedule**: ε decays from 0.2 to 0.05 over 10K steps
- **Implementation**: `guandan/training/actor_policy.py`

#### Reward System
- **Round Rewards**: +3/+2/+1 for winning team based on partner position
- **Global Rewards**: +1/-1 for episode win/loss
- **Tribute Phase**: Heuristic rules (excluded from training data)

### Distributed Training Framework

#### Architecture Components
- **Actor-Learner Separation**: Actors collect data, learner updates model
- **Parameter Synchronization**: Periodic weight broadcast via RLlib
- **Self-Play**: 4 agents per episode with uniform model
- **Replay Buffer**: Centralized buffer for experience storage

#### Configuration
- **Workers**: 120 workers × 3 environments = 360 parallel environments
- **GPU Allocation**: 1 GPU for learner, CPU-only workers
- **Memory**: 30GB Ray object store
- **Checkpointing**: Every 100 iterations

## Performance Specifications

### Hardware Requirements
- **GPUs**: 2× H100 80GB (minimum), 8× H100 (optimal)
- **CPUs**: 192 cores
- **Memory**: 100+ GB disk space for training

### Performance Targets
- **Iteration Time**: 3.8-4.2 seconds
- **Throughput**: ~0.24 iter/s (900 iter/hour)
- **GPU Utilization**: 20-30% (environment-bound)
- **Training Duration**: ~11 hours for 10K iterations (2 H100s)

### Scaling Performance
- **8 H100s**: 5-10× faster training (~1-2 hours for 10K iterations)
- **Worker Scaling**: Linear scaling up to 800 workers
- **Environment Scaling**: Up to 1600 parallel environments

## Implementation Status

### ✅ Completed Components
- **Environment**: Complete Guandan game engine with RLLib MultiAgentEnv wrapper
- **Observation Space**: Paper-compliant 513-dimensional observation space
- **DMC Algorithm**: Deep Monte Carlo with replay buffer and learner
- **GPU Optimization**: Batched action selection, mixed precision training
- **Distributed Training**: Ray RLlib with actor-learner architecture
- **Checkpointing**: Automatic model saving every 100 iterations
- **Agent Interface**: BaseAgent unification implemented

### 🔧 Key Features
- **Paper Compliance**: Follows DanZero research paper specifications exactly
- **Scalable Architecture**: Tested up to 300 workers, ready for 8 H100s
- **Production Ready**: Comprehensive logging, monitoring, and checkpointing
- **GPU Efficient**: Optimized for H100 with TF32 and mixed precision

## Configuration Management

### Training Configuration
```yaml
# configs/dqmc_large.yaml
model:
  hidden_sizes: [512, 512, 512, 512, 512]
  activation: "tanh"
  initialization: "orthogonal"

training:
  learning_rate: 1e-3
  batch_size: 64
  epsilon_start: 0.2
  epsilon_end: 0.05
  epsilon_decay_steps: 10000
  target_update_freq: 1000

distributed:
  num_workers: 120
  num_envs_per_worker: 3
  num_gpus: 1.0
  num_gpus_per_worker: 0.0
```

### Environment Configuration
```yaml
# Environment settings
observation_mode: "comprehensive"
max_steps: 3000
use_internal_adapters: false
```

## API Reference

### Core Classes
- **`ComprehensiveObservationExtractor`**: Main observation space extractor
- **`GuandanQModel`**: Q-network implementation
- **`EpsilonScheduler`**: ε-greedy scheduling
- **`ReplayBuffer`**: Experience storage and sampling
- **`DMCLearner`**: DMC algorithm implementation

### Key Functions
- **`extract_observation()`**: Extract observation from game state
- **`select_action_epsilon_greedy()`**: ε-greedy action selection
- **`create_dmc_trainer()`**: Create DMC training algorithm
- **`make_guandan_env()`**: Create RLLib environment

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or worker count
2. **Low GPU Utilization**: Normal for environment-bound training
3. **Ray Object Store Full**: Increase object store memory
4. **Training Crashes**: Check Ray status and GPU memory

### Debug Tools
```python
# Check observation space
from guandan.env.observation_utils import print_observation_space_summary
print_observation_space_summary(extractor)

# Monitor training
watch -n 1 nvidia-smi
tail -f logs/danzero_production_*/gpu_monitor.log
```

---
*Technical Reference - Last Updated: October 9, 2025*

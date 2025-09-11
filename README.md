# DanZero (Guandan Reinforcement Learning & Rule-based Baseline Refactoring)

> This repository is based on the DouZero codebase and has been evolved to migrate from the original "Doudizhu" (Fight the Landlord) environment to "Guandan" and build a scalable distributed reinforcement learning training and evaluation framework.

## 1. Project Vision

- Provide a standardized, vectorizable Guandan environment (unified state/action encoding, parallelizable rollout).
- Integrate Ray distributed components for high-concurrency self-play sampling and centralized Learner updates.
- Compatible unified evaluation interface for rule-based baselines and learning policies (RL policy).
- Progressive cleanup of historical Doudizhu legacy code and duplicate utility functions to ensure code cleanliness and maintainability.

## 2. Current Progress Summary

| Module | Status | Description |
|--------|--------|-------------|
| env/ | ✅ Complete | Guandan core logic fully implemented with comprehensive game engine, state management, and action validation. |
| env/observation_extractor.py | ✅ Complete | JSON to numpy observation conversion system with 212-dimensional observation space. |
| env/rllib_env*.py | ✅ Complete | RLLib MultiAgentEnv wrappers for distributed training integration. |
| agent/ai1-ai6 | ✅ Migrated | Rule-based AIs restructured and organized; null bytes issue resolved. |
| agent/baselines/legacy/mc | Archived | Old MC/Q baselines, archived only, no longer registered. |
| agent/torch | Retained | Legacy neural network examples (feature encoding for reference, will be refactored/decommissioned). |
| training/ (Ray) | ✅ Complete | Full training pipeline with parameter server, rollout workers, and learner components. |
| RLLib Integration | ✅ Complete | Working environment with test suites and comprehensive integration guide. |
| Documentation | ✅ Updated | README, PROJECT_STRUCTURE, and RLLIB_INTEGRATION_GUIDE maintained. |

## 3. Directory Structure (Current View)

```text
guandan/
  env/                # Guandan game state, rules and action generation logic
  ├── game.py         # Main game environment and self-play driver
  ├── engine.py       # Game rules, stage flow, and state transitions
  ├── card_deck.py    # Two-deck card generation and dealing logic
  ├── player.py       # Player state and data structures
  ├── context.py      # Game context and shared state
  ├── table.py        # Table state management
  ├── utils.py        # Card pattern analysis and legal action generation
  ├── move_detector.py    # Move validation and detection
  ├── move_generator.py   # Legal move generation
  ├── move_selector.py    # Move selection utilities
  ├── observation_extractor.py  # JSON to numpy observation conversion
  ├── rllib_env.py    # RLLib MultiAgentEnv wrapper
  └── rllib_env_simple.py  # Simplified RLLib environment
  agent/
    agents.py         # Agent registry and factory
    random_agent.py   # Random baseline
    ai1/              # Complex rule-based strategy
    ai2/              # Phased decision-making strategy
    ai3/              # Experimental/adversarial strategy
    ai4/              # Alternative heuristic approach
    ai6/              # Additional heuristic strategy
    baselines/
      rule/           # Rule-based AI implementations
      legacy/mc/      # Archived MC/Q baselines
    torch/            # Neural network-based agents
  training/           # Ray training pipeline components
  ├── ray_app.py      # Ray-based training orchestrator
  ├── parameter_server.py  # Distributed parameter server
  ├── rollout_worker.py    # Self-play data collection worker
  ├── learner.py      # Model training and optimization
  ├── checkpoint.py   # Model checkpoint management
  └── logger.py       # Training metrics and logging
  dmc/                # DeepMind Control integration (empty)
  config.py           # Centralized configuration
archive/
  doudizhu/           # Old Doudizhu scripts and original README archive
README.md             # This file
PROJECT_STRUCTURE.md  # Detailed project documentation
RLLIB_INTEGRATION_GUIDE.md  # RLLib integration documentation
```

## 4. Rule-based Baseline Strategies (rule/) Status

| Baseline | Positioning | Next Steps |
|----------|-------------|------------|
| ai1 | Main complex rules | Adapt to BaseAgent, extract card type parsing modules |
| ai2 | Phased decision making | Align action indices with ai1; merge duplicate features |
| ai3 | Experimental/adversarial | Evaluate for removal or conversion to profile parameterization |
| ai4 | Overlaps with ai1 | Planned for removal / merge |
| ai6 | Another heuristic approach | Eliminate if no unique value |
| random | Random baseline | Keep as minimal sanity check |

See baselines/README.md for details.

## 5. Upcoming Refactoring Milestones

| Milestone | Goal | Key Deliverables |
|-----------|------|------------------|
| M1: Observation Unification | Standardize obs dict(tensor) + legal_action_mask | env/observation_builder.py (new) |
| M2: Action Encoding Table | Fixed global action enumeration + dynamic legal mask generation | action_space.json / builder |
| M3: BaseAgent Abstraction | Unified act()/reset() interface | agent/base.py + rule adapters |
| M4: Ray Integration | rollout → buffer → learner closed loop | Training scripts & metrics output |
| M5: Redundancy Cleanup | Remove ai3/ai4/ai6 & torch (optional) | Streamlined agents.py / documentation updates |

## 6. Technical Design Points (Planned)

### 6.1 Observation Composition (Draft)

| Component | Description | Shape (Example) |
|-----------|-------------|-----------------|
| hand_cards | Own current hand cards one-hot/multi-channel | (C1,) or (Suits,Ranks) |
| public_info | Table recent N rounds of played cards structured encoding | (N, action_feat) |
| teammate_hint | Teammate remaining cards/category estimation | (vector) |
| phase_meta | Current level / tribute phase markers | (small vector) |
| legal_mask | Legal action mask (lazy construction) | (A,) |

### 6.2 Action Encoding Strategy

- Establish global "normalized action set": expand by card type + main value + auxiliary card slots.
- Provide bidirectional mapping: encode(move)->index and decode(index)->move.
- For "dynamic length" card types (straights / triple pairs), use length upper bound + padding; or split into (card_type, start_rank, length) triple combination encoding.

### 6.3 BaseAgent Adapter

Legacy rule baselines retain original `received_message` flow → adapter parses message → temporarily construct obs + mask → call internal decision → output unified action index.

## 7. Quick Start

### Traditional Agent Usage (Current)

```python
from guandan.agent.agents import agent_cls
agent = agent_cls['ai1'](id=0)
# Assuming server-pushed JSON message msg exists
action_index = agent.received_message(msg)
```

### RLLib Environment Usage (New)

```python
from guandan.env.rllib_env_simple import GuandanRLLibEnv

# Create environment
env = GuandanRLLibEnv()

# Reset and get observations
obs = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Step through environment
action = env.action_space.sample()  # Random action
obs, rewards, dones, infos = env.step(action)
```

### Future Unified Interface

```python
from guandan.agent import make_agent
agent = make_agent('rule_ai1')
obs, mask = env.reset()
while not done:
    a = agent.act(obs, mask)
    obs, reward, done, info = env.step(a)
```

## 7.1 RLLib Integration

The project now includes comprehensive RLLib integration for distributed reinforcement learning:

- **MultiAgentEnv Wrapper**: `guandan/env/rllib_env_simple.py` provides a working RLLib-compatible environment
- **Observation Extraction**: `guandan/env/observation_extractor.py` converts JSON game state to 212-dimensional numpy arrays
- **Test Suites**: Complete validation with `test_rllib_simple.py` and `test_rllib_env_simple.py`
- **Integration Guide**: Detailed documentation in `RLLIB_INTEGRATION_GUIDE.md`

See `RLLIB_INTEGRATION_GUIDE.md` for complete integration details and usage examples.

## 8. Contributing & Collaboration

Welcome through Issues / PRs:

- Report Guandan rule/judgment differences
- Provide more efficient action space compression solutions
- Optimize feature extraction and network architecture

## 9. Change Log (Simplified)

| Date | Change | Summary |
|------|--------|---------|
| 2025-01-XX | RLLib Integration | Complete RLLib MultiAgentEnv wrapper with observation extraction |
| 2025-01-XX | Environment Enhancement | Added observation_extractor.py and comprehensive test suites |
| 2025-01-XX | Documentation Update | Updated README and PROJECT_STRUCTURE with current implementation status |
| 2025-09-02 | README rewrite | Archived old DouZero README, added Guandan direction description |
| 2025-09-02 | baselines reorganization | Migrated rule ai1~ai6, archived mc |

> Complete history available in Git commit records.

## 10. Future Discussion Topics (Open Questions)

- Optimal encoding strategy for variable-length actions (straights / consecutive pairs) (fixed slots vs. segmented actions)
- Team information sharing modeling (feature fusion for teammate card information estimation)
- Stage-aware value normalization (stage-aware value normalization)
- Experience replay vs. DMC asynchronous advantage comparison

---
If you're looking for the old DouZero (Doudizhu) documentation, please check: `archive/doudizhu/README_doudizhu_original.md`.

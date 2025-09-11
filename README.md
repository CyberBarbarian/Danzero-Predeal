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
| env/ | Complete | Guandan core logic with game engine, state management, and action validation. |
| agent/ai1-ai6 | Migrated | Rule-based AIs restructured and organized. |
| agent/baselines/legacy/mc | Archived | Old MC/Q baselines, archived only, no longer registered. |
| agent/torch | Retained | Legacy neural network examples (feature encoding for reference, will be refactored/decommissioned). |
| training/ (Ray) | Initial | Contains parameter_server / rollout_worker / learner basic skeleton, needs alignment with new env.obs interface. |
| RLLib Integration | In Progress | MultiAgentEnv wrapper and observation extraction system under development. |

## 3. Directory Structure (Simplified View)

```text
guandan/
  env/                # Guandan game state, rules and action generation logic
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
archive/
  doudizhu/           # Old Doudizhu scripts and original README archive
README.md             # This file
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

## 7. Quick Start (Current Method)

```python
from guandan.agent.agents import agent_cls
agent = agent_cls['ai1'](id=0)
# Assuming server-pushed JSON message msg exists
action_index = agent.received_message(msg)
```

Future version will be:

```python
from guandan.agent import make_agent
agent = make_agent('rule_ai1')
obs, mask = env.reset()
while not done:
    a = agent.act(obs, mask)
    obs, reward, done, info = env.step(a)
```

## 8. Contributing & Collaboration

Welcome through Issues / PRs:

- Report Guandan rule/judgment differences
- Provide more efficient action space compression solutions
- Optimize feature extraction and network architecture

## 9. Future Discussion Topics (Open Questions)

- Optimal encoding strategy for variable-length actions (straights / consecutive pairs) (fixed slots vs. segmented actions)
- Team information sharing modeling (feature fusion for teammate card information estimation)
- Stage-aware value normalization (stage-aware value normalization)
- Experience replay vs. DMC asynchronous advantage comparison

---
If you're looking for the old DouZero (Doudizhu) documentation, please check: `archive/doudizhu/README_doudizhu_original.md`.

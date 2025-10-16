# Baselines Documentation

This directory contains the current baseline implementations for the Guandan project, mainly divided into two categories:

```text
baselines/
  README.md               # This file
  rule/                   # Rule-based (heuristic/hand-written logic) legacy client implementations
    ai1/ ai2/ ai3/ ai4/ ai6/
  legacy/                 # Archived implementations, not maintained but kept for reference value
    mc/                   # Legacy Monte Carlo / Simple Q-network baseline (interface no longer exposed)
```

## 1. rule/ Rule-based Baselines

| Directory | Role | Status | Notes |
|-----------|------|--------|-------|
| ai1  | Complex rule set | ✅ Active | Main rule baseline (recommended) |
| ai2  | Phased strategy | ✅ Active | Second reference rule baseline |
| ai3  | Adversarial/experimental | ✅ Active | Experimental strategy for testing |
| ai4  | Alternative heuristic | ✅ Active | Complementary approach to ai1 |
| ai6  | Specialized strategy | ✅ Active | Additional heuristic approach |

Common features:

- Entry method remains `received_message(message)`, using legacy communication protocol (non-standard Gym/RL interface).
- Large amounts of duplicate card type splitting, evaluation, and sorting logic (will be extracted into common feature modules later).
- Added `TODO` comments marking locations that need migration/abstraction.

## 2. legacy/

Currently only contains:

- `mc/`: Legacy Monte Carlo / Simple reinforcement learning baseline, code structure incompatible with new Ray training pipeline, removed from `agent_cls` registry.

Retention reasons:

- For reference of early network structure / data flow (may be used for quick comparison).
- Convenient for tracing back old experiments.

Future handling: If not referenced after 2 versions, will be completely deleted or packaged for archiving.

## 3. Naming and Plans

Will gradually rename `aiX` format to unified prefix:

- `rule_ai1`, `rule_ai2` ... to reduce semantic ambiguity.
- Or abstract into single `rule_agent` + configurable strategy profile (if differences are only in parameters).

## 4. Interface Unification Plan (BaseAgent)

Planned unified interface (draft):

```python
class BaseAgent:
    def reset(self, env_info: dict):
        ...
    def act(self, obs, legal_action_mask=None) -> int:
        ...
    def observe(self, transition):  # Optional (for training)
        ...
```

Adaptation steps:

1. Add wrapper in `rule/adapter.py` to adapt old `received_message` → `act`.
2. Extract card type enumeration/splitting functions to `guandan/env/utils.py` or create new `guandan/feature/combination.py`.
3. Action space standardization: build unified action index table + legal mask generation.
4. Gradually remove duplicate utils from each `aiX`.

## 5. Current Registry (agent_cls)

`guandan/agent/agents.py` retains: `ai1, ai2, ai3, ai4, ai6, torch, random`.

- `mc` removed (legacy).
- Before new training/evaluation pipeline integration, temporarily continue using old key names.

## 6. Recommended Usage

Temporary direct reference:

```python
from guandan.agent.agents import agent_cls
agent = agent_cls['ai1'](id=0)
```

Future versions will encourage:

```python
from guandan.agent import make_agent  # Planned new factory function
```

## 7. TODO Summary

- [x] Establish `BaseAgent` abstraction and adapter layer
- [x] Extract common card type/feature logic
- [x] Unify action index + legal mask
- [ ] Rename aiX → rule_aiX or merge strategy configuration
- [ ] Clean up redundancy (ai3 / ai4 / ai6) and log noise output
- [x] Add minimal smoke test for rule baselines (import + simple fake message driving)

## 8. Current Status (Updated)

### Recent Improvements

- ✅ **Win/Loss Logic**: Integrated proper Guandan win/loss detection in game core
- ✅ **Tribute System**: Fixed tribute back limit (8 → 10 points) and priority rules
- ✅ **Rank Progression**: Corrected rank upgrade mechanics based on partner position
- ✅ **Tournament System**: Implemented proper agent vs agent testing with zero draws
- ✅ **Documentation**: Comprehensive API and development documentation

### Agent Performance

All rule-based agents (ai1-ai6) are now fully functional and tested:

- **ai1**: Most sophisticated strategy with complex decision making
- **ai2**: Phased approach with strategic planning
- **ai3**: Experimental/adversarial tactics
- **ai4**: Alternative heuristic approach
- **ai6**: Specialized strategy with unique tactics

### Integration Status

- ✅ **RLLib Integration**: All agents work seamlessly with RLLib MultiAgentEnv
- ✅ **Tournament Testing**: Agents can be tested against each other
- ✅ **Observation Space**: Both simple (212D) and comprehensive (513D) modes supported
- ✅ **Game Rules**: Full Guandan rules implementation with proper win detection

---
If you need to add more internal metrics or refactoring progress to the README, please add an Issue or update this file in a PR.

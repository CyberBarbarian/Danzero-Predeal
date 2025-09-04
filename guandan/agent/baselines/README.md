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
| ai1  | Complex rule set | Retained | As main rule baseline (recommended) |
| ai2  | Phased strategy | Retained | As second reference rule baseline |
| ai3  | Adversarial/experimental | To be pruned | Logic redundancy, mainly for early adversarial testing |
| ai4  | Highly overlaps with ai1 | To be archived | Planned for future simplification or deletion |
| ai6  | Another heuristic approach | To be archived | Kept only for specific comparisons |

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

- [ ] Establish `BaseAgent` abstraction and adapter layer
- [ ] Extract common card type/feature logic
- [ ] Unify action index + legal mask
- [ ] Rename aiX → rule_aiX or merge strategy configuration
- [ ] Clean up redundancy (ai3 / ai4 / ai6) and log noise output
- [ ] Add minimal smoke test for rule baselines (import + simple fake message driving)

## 8. Deletion Strategy Recommendations

| Condition | Action |
|-----------|--------|
| BaseAgent adaptation complete and rule_ai1 & rule_ai2 stable | Remove ai3/ai4/ai6 |
| No one references legacy/mc and new RL runs successfully | Delete legacy/mc |

---
If you need to add more internal metrics or refactoring progress to the README, please add an Issue or update this file in a PR.

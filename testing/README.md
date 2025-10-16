# Testing Directory

This directory contains all test scripts, unit tests, logs, and outputs for the DanZero project.

## 📁 Structure

### `scripts/`
- **`train_test.sh`** - Quick test script (20 iterations, ~2 minutes)
- **`cleanup_test_files.sh`** - Cleanup script for test files
- **`debug_metrics.py`** - Debug metrics collection script
- **`test_dmc_components.py`** - DMC component testing
- **`run_two_gpus_1000.py`** - Two GPU test script

### `unit_tests/`
- **`test_epsilon_scheduler.py`** - Epsilon scheduler unit tests
- **`test_learner_update.py`** - Learner update unit tests
- **`test_rllib_env_and_model.py`** - RLlib environment and model tests
- **`test_rollout_multistep.py`** - Rollout multistep unit tests

### `logs/`
- Test run logs and output files
- Debug logs and temporary files

### `outputs/`
- Test checkpoints (`checkpoints_test/`)
- Test logs (`logs_test/`)
- Test results (`results_test/`)

## 🚀 Usage

### Quick Test
```bash
cd testing/scripts
bash train_test.sh
```

### Unit Tests
```bash
# Run all unit tests
python -m pytest testing/unit_tests/

# Run specific test
python -m pytest testing/unit_tests/test_rllib_env_and_model.py
```

### Component Tests
```bash
cd testing/scripts
python test_dmc_components.py
```

### Cleanup Test Files
```bash
cd testing/scripts
bash cleanup_test_files.sh
```

### Debug Metrics
```bash
cd testing/scripts
python debug_metrics.py
```

## 📋 Notes
- All test-related files are now consolidated in this directory
- Unit tests use pytest framework
- Test scripts are separate from production scripts
- Test outputs are isolated in this directory
- Use `cleanup_test_files.sh` to remove old test files
- Production training uses `train_production.sh` in root directory

---
*Organized: October 9, 2025*

# Testing and Profiling Scripts

This directory contains scripts for testing and profiling the DanZero training system.

## Quick Testing (No Profiling)

For rapid validation without resource monitoring:

```bash
cd /mnt/project_modelware/lizikang/DanZero
source /mnt/project_modelware/lizikang/Danvenv/bin/activate
python testing/scripts/test_cpu_optimization.py
```

**Duration:** ~30 seconds (5 iterations)

**Output:** Console only, validates configuration stability

## Profiling Test (With CPU/GPU Monitoring)

For detailed resource utilization analysis:

```bash
cd /mnt/project_modelware/lizikang/DanZero
source /mnt/project_modelware/lizikang/Danvenv/bin/activate
python testing/scripts/test_with_profiling.py
```

**Duration:** ~2-3 minutes (20 iterations)

**Output:** Saved to `testing/outputs/profiling_[timestamp]/`

### Generated Files

1. **`resource_metrics.json`**
   - Raw CPU utilization data (sampled every second)
   - Raw GPU utilization data (both GPUs, every second)
   - GPU memory usage over time
   - Timestamps for all measurements

2. **`training_metrics.json`**
   - Iteration times
   - Episodes completed
   - Timesteps sampled
   - Reward progression
   - Episode lengths

3. **`profiling_results.png`**
   - 6 subplots showing:
     - CPU utilization curve
     - GPU 0 utilization curve
     - GPU memory usage
     - Iteration time progression
     - Episode completion progress
     - Reward progression

## What to Look For

### CPU Utilization (Target: 70-80%)
- **Mean CPU usage** should be 70-80% during training
- If < 60%: Consider increasing workers or envs per worker
- If > 90%: May cause context switching overhead

### GPU Utilization (Target: 20-40%)
- **Mean GPU usage** depends on CPU bottleneck
- Higher is better (means GPU is being fed data)
- If < 10%: CPU is severely bottlenecked
- If > 80%: GPU is bottleneck (rare with current config)

### Training Performance
- **Iteration time** should stabilize after 5-10 iterations
- First iteration is slower (worker initialization)
- Consistent iteration times = stable configuration
- Episodes per iteration should be > 0

### Memory Usage
- GPU memory should be stable (not growing)
- If growing: potential memory leak
- Should use ~10-15 GB per GPU with current config

## Customizing Test Duration

Edit `test_with_profiling.py` to change number of iterations:

```python
# Line 228
NUM_TEST_ITERATIONS = 20  # Change this value
```

Recommended values:
- Quick test: 10 iterations (~1 minute)
- Standard test: 20 iterations (~2-3 minutes)
- Thorough test: 50 iterations (~5-7 minutes)

## Comparing Configurations

To compare different configurations:

1. Run profiling test with current config (150 workers × 4 envs)
2. Note the output directory
3. Modify `train_production.sh` to try different config
4. Run profiling test again
5. Compare the `profiling_results.png` plots

Example configurations to test:
- 120 workers × 5 envs = 600 environments
- 160 workers × 4 envs = 640 environments
- 180 workers × 3 envs = 540 environments

## Integration with Production Training

**Important:** The profiling script is for TEST MODE ONLY.

The production training script (`train_production.sh`) does NOT include this profiling overhead:
- No per-second CPU/GPU sampling
- No matplotlib plot generation
- Focused on maximum training throughput

To monitor production training:
- Use the built-in monitoring: `logs/[experiment]/gpu_monitor.log`
- Use system tools: `htop`, `nvidia-smi dmon`
- Check training metrics: `results/[experiment]/training_stats.json`

## Troubleshooting

### matplotlib ImportError
If you see "matplotlib not available", install it:
```bash
pip install matplotlib
```

The script will still run and save JSON data, just won't generate plots.

### GPU Monitoring Fails
If `nvidia-smi` errors occur:
- Check GPU drivers: `nvidia-smi`
- Ensure CUDA_VISIBLE_DEVICES is set correctly
- Script will continue with 0 values for GPU metrics

### CPU Monitoring Fails
If `top` command errors occur:
- Ensure `top` is installed (should be on all Linux systems)
- Script will continue with 0 values for CPU metrics

### Ray Initialization Errors
If Ray fails to initialize:
- Check available resources: `ray status`
- Ensure no other Ray sessions: `ray stop`
- Reduce worker count in test temporarily

## Example Output

After running the profiling test:

```
================================================================================
📊 PROFILING SUMMARY
================================================================================

CPU Utilization:
  Mean: 76.3%
  Max:  89.2%
  Min:  45.1%

GPU 0 Utilization:
  Mean: 28.4%
  Max:  65.3%
  Min:  5.2%

Training Performance:
  Iterations: 20
  Avg iteration time: 4.23s
  Total training time: 84.6s
  Throughput: 0.236 iter/s

================================================================================
📁 All results saved to: testing/outputs/profiling_20251009_193045
================================================================================

Files generated:
  • testing/outputs/profiling_20251009_193045/resource_metrics.json
  • testing/outputs/profiling_20251009_193045/training_metrics.json
  • testing/outputs/profiling_20251009_193045/profiling_results.png
```

## Next Steps After Profiling

1. **Review the plots** - Check if utilization matches expectations
2. **Analyze bottlenecks** - Is CPU or GPU the limiting factor?
3. **Adjust configuration** - Modify workers/envs based on findings
4. **Re-test** - Validate improvements
5. **Run production training** - Once satisfied with configuration

## Questions?

Check the main project documentation: `docs/`


#!/usr/bin/env python3
"""
End-to-end training runner: runs for a specified number of epochs and then
performs full analysis automatically.

Usage:
  python testing/scripts/train_and_analyze_rounds.py --epochs 10

Options:
  --epochs/-e <int>        Number of epochs to train (overrides script default)
  --results-dir <path>     Optional fixed results dir name prefix
  --ray-dir <path>         Optional Ray results dir for re-analysis
  --no-monitor             Disable background live monitor
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_SCRIPT = os.path.join(ROOT, "testing", "scripts", "test_long_training_monitoring.py")
ANALYZE_SCRIPT = os.path.join(ROOT, "testing", "scripts", "analyze_guandan_metrics.py")


def run_training(epochs: int, results_dir_hint: str | None, enable_monitor: bool) -> str:
    env = os.environ.copy()
    env["DANZERO_TOTAL_EPOCHS"] = str(epochs)
    if not enable_monitor:
        env["DANZERO_DISABLE_MONITOR"] = "1"

    # Optional results dir hint propagated for consistency in naming (prefixed)
    if results_dir_hint:
        env["DANZERO_RESULTS_PREFIX"] = results_dir_hint

    # Run training synchronously so we can do analysis right after
    subprocess.check_call([sys.executable, TRAIN_SCRIPT], env=env)

    # Discover latest training_results_<timestamp>
    # Find latest under project results/ instead of project root
    proj_results = os.path.join(ROOT, "results")
    if not os.path.isdir(proj_results):
        proj_results = ROOT
    entries = [d for d in os.listdir(proj_results) if d.startswith("training_results_") and os.path.isdir(os.path.join(proj_results, d))]
    if not entries:
        raise RuntimeError("No training_results_* directory found after training.")
    latest = max(entries, key=lambda d: os.path.getmtime(os.path.join(proj_results, d)))
    return os.path.join(proj_results, latest)


def run_analysis(results_dir: str, ray_dir: str | None):
    # Primary path: analyze from Ray results (auto-detected if not provided)
    cmd = [sys.executable, ANALYZE_SCRIPT]
    if ray_dir:
        cmd += ["--ray-dir", ray_dir]
    output_path = os.path.join(results_dir, "guandan_analysis_full.json")
    cmd += ["--output", output_path]

    try:
        subprocess.check_call(cmd)
        print(f"✅ Full analysis saved to: {output_path}")
    except subprocess.CalledProcessError:
        print("⚠️ Analysis from Ray results failed; skipping.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=3)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--ray-dir", type=str, default=None)
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 Train-and-Analyze Runner")
    print("=" * 80)
    print(f"🎯 Epochs: {args.epochs}")
    print(f"📟 Monitor: {'Disabled' if args.no_monitor else 'Enabled'}")
    if args.results_dir:
        print(f"📁 Results prefix: {args.results_dir}")
    print()

    results_dir = run_training(args.epochs, args.results_dir, not args.no_monitor)
    print(f"📁 Training results: {results_dir}")

    run_analysis(results_dir, args.ray_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()



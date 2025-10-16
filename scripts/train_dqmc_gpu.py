#!/usr/bin/env python3
"""
GPU-optimized DMC Training Script

This script uses batched action selection and GPU data pipeline for better GPU utilization.
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from guandan.training.loop_gpu import run_gpu_training_iteration


def main():
    parser = argparse.ArgumentParser(description="GPU-optimized DMC training")
    parser.add_argument("--config", type=str, default="configs/dqmc_large.yaml", 
                       help="Path to config file")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to train")
    parser.add_argument("--batch_size", type=int, default=4096,
                       help="Training batch size")
    parser.add_argument("--action_batch_size", type=int, default=32,
                       help="Batch size for action selection")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Log directory")
    
    args = parser.parse_args()
    
    print(f"Starting GPU-optimized DMC training...")
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    print(f"Training batch size: {args.batch_size}")
    print(f"Action batch size: {args.action_batch_size}")
    
    result = run_gpu_training_iteration(
        log_dir=args.log_dir,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        config_path=args.config,
        action_batch_size=args.action_batch_size,
    )
    
    print(f"Training completed!")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()

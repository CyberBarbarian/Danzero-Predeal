#!/usr/bin/env python3
"""
RLLib DMC Training Script

Trains DMC algorithm using RLLib's distributed framework.
"""
import argparse
import json

from guandan.rllib.trainers import create_dmc_trainer


def main():
    parser = argparse.ArgumentParser(description="Train DMC using RLLib")
    parser.add_argument("--env_config", type=str, default='{"observation_mode": "comprehensive", "use_internal_adapters": false}', 
                       help="JSON string for environment config")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of rollout workers")
    parser.add_argument("--num_envs_per_worker", type=int, default=1, help="Environments per rollout worker")
    parser.add_argument("--num_gpus", type=int, default=None, help="Total GPUs for learner (None=auto)")
    parser.add_argument("--num_gpus_per_worker", type=float, default=0.0, help="GPU fraction per rollout worker")
    parser.add_argument("--disable_gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epsilon_start", type=float, default=0.2, help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=10000, help="Epsilon decay steps")
    
    args = parser.parse_args()
    
    # Parse environment config
    env_config = json.loads(args.env_config)
    
    print(f"Starting DMC training with RLLib...")
    print(f"Environment config: {env_config}")
    print(f"Workers: {args.num_workers}")
    print(f"Iterations: {args.iterations}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} over {args.epsilon_decay_steps} steps")
    
    # Create DMC trainer
    algo = create_dmc_trainer(
        env_config=env_config,
        num_workers=args.num_workers,
        num_envs_per_worker=args.num_envs_per_worker,
        num_gpus=0 if args.disable_gpu else args.num_gpus,
        num_gpus_per_worker=args.num_gpus_per_worker,
        lr=args.lr,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    
    # Training loop
    for i in range(args.iterations):
        result = algo.train()
        
        print(f"Iteration {i+1}/{args.iterations}:")
        print(f"  Episode reward mean: {result.get('episode_reward_mean', 0.0):.3f}")
        print(f"  Episodes this iter: {result.get('episodes_this_iter', 0)}")
        print(f"  Update count: {result.get('update_count', 0)}")
        print(f"  Timesteps total: {result.get('timesteps_total', 0)}")
        
        # Save checkpoint every 10 iterations
        if (i + 1) % 10 == 0:
            checkpoint_path = algo.save(f"checkpoints/dmc_rllib_checkpoint_{i+1}")
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    print("Training completed!")
    
    # Save final checkpoint
    final_checkpoint = algo.save("checkpoints/dmc_rllib_final")
    print(f"Final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()

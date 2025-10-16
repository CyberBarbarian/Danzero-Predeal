#!/usr/bin/env python3
import argparse

from guandan.training.loop import run_training_iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/dqmc_paper.yaml')
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    out = run_training_iteration(
        config_path=args.config,
        num_episodes=args.episodes if args.episodes is not None else 1,
        batch_size=args.batch_size if args.batch_size is not None else 1,
    )
    print(out)


if __name__ == '__main__':
    main()



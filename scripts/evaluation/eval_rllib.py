#!/usr/bin/env python3
import argparse
import json

from ray.rllib.algorithms.algorithm import Algorithm

from guandan.rllib.builders import build_ppo_config, init_ray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", type=str, default="{}", help="JSON for env config")
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env_config = json.loads(args.env_config)
    init_ray()
    algo: Algorithm = build_ppo_config(env_config).build()
    if args.checkpoint:
        algo.restore(args.checkpoint)

    for _ in range(args.episodes):
        env = algo.workers.local_worker().env
        obs, _ = env.reset()
        done = {aid: False for aid in env.agent_ids}
        total_reward = {aid: 0.0 for aid in env.agent_ids}
        while not all(done.values()):
            actions = {}
            for aid, o in obs.items():
                actions[aid] = algo.compute_single_action(o, policy_id=None)
            obs, rewards, terminated, truncated, _ = env.step(actions)
            done = {aid: terminated.get(aid, False) or truncated.get(aid, False) for aid in env.agent_ids}
            for aid, r in rewards.items():
                total_reward[aid] += r
        print("episode_reward_sum", total_reward)


if __name__ == "__main__":
    main()



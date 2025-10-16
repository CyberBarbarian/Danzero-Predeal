#!/usr/bin/env python3
"""
Evaluate a trained checkpoint to check training quality
"""

import ray
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from guandan.env.rllib_env import GuandanMultiAgentEnv

def evaluate_checkpoint(checkpoint_path, num_episodes=10):
    """
    Evaluate a trained checkpoint by playing episodes and measuring performance.
    
    Returns:
        dict: Statistics about the trained policy's performance
    """
    print("="*80)
    print("🎯 CHECKPOINT EVALUATION")
    print("="*80)
    print()
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Convert to absolute path if needed
    import os
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"Absolute path: {checkpoint_path}")
    
    # Initialize Ray
    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_gpus=1)
    
    # Load the trained algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print("✅ Checkpoint loaded successfully")
    print()
    
    # Create evaluation environment
    env = GuandanMultiAgentEnv({
        "observation_mode": "comprehensive",
        "use_internal_adapters": False,
        "max_steps": 3000,
    })
    
    print(f"Running {num_episodes} evaluation episodes...")
    print("-"*80)
    
    episode_rewards = {agent_id: [] for agent_id in env.agent_ids}
    episode_lengths = []
    wins_by_team = {"team1": 0, "team2": 0}
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = {"__all__": False}
        episode_reward = {agent_id: 0.0 for agent_id in env.agent_ids}
        episode_length = 0
        
        while not done["__all__"]:
            # Get actions from trained policy
            actions = {}
            for agent_id in obs.keys():
                action = algo.compute_single_action(obs[agent_id], policy_id=agent_id)
                actions[agent_id] = action
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = {k: terminated.get(k, False) or truncated.get(k, False) 
                   for k in terminated.keys()}
            done["__all__"] = all(done.values()) or terminated.get("__all__", False) or truncated.get("__all__", False)
            
            # Track rewards
            for agent_id in rewards.keys():
                episode_reward[agent_id] += rewards[agent_id]
            
            episode_length += 1
        
        # Record episode statistics
        for agent_id in env.agent_ids:
            episode_rewards[agent_id].append(episode_reward.get(agent_id, 0.0))
        episode_lengths.append(episode_length)
        
        # Determine winner
        team1_reward = episode_reward.get('agent_0', 0) + episode_reward.get('agent_2', 0)
        team2_reward = episode_reward.get('agent_1', 0) + episode_reward.get('agent_3', 0)
        
        if team1_reward > team2_reward:
            wins_by_team["team1"] += 1
            winner = "Team 1"
        elif team2_reward > team1_reward:
            wins_by_team["team2"] += 1
            winner = "Team 2"
        else:
            winner = "Draw"
        
        print(f"Episode {ep+1:2d}: Length={episode_length:4d} | "
              f"Team1={team1_reward:+6.2f} | Team2={team2_reward:+6.2f} | "
              f"Winner: {winner}")
    
    print()
    print("="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print()
    
    # Calculate statistics
    print("Episode Statistics:")
    print(f"  Average length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Min length: {np.min(episode_lengths)} steps")
    print(f"  Max length: {np.max(episode_lengths)} steps")
    print()
    
    print("Reward Statistics (per agent):")
    for agent_id in env.agent_ids:
        rewards = episode_rewards[agent_id]
        print(f"  {agent_id}: {np.mean(rewards):+6.2f} ± {np.std(rewards):5.2f} "
              f"(min={np.min(rewards):+6.2f}, max={np.max(rewards):+6.2f})")
    print()
    
    print("Team Performance:")
    print(f"  Team 1 wins: {wins_by_team['team1']}/{num_episodes} ({100*wins_by_team['team1']/num_episodes:.1f}%)")
    print(f"  Team 2 wins: {wins_by_team['team2']}/{num_episodes} ({100*wins_by_team['team2']/num_episodes:.1f}%)")
    print()
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'wins_by_team': wins_by_team,
        'num_episodes': num_episodes
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_checkpoint.py <checkpoint_path> [num_episodes]")
        print()
        print("Example:")
        print("  python evaluate_checkpoint.py checkpoints/danzero_production_20251009_164454 10")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    results = evaluate_checkpoint(checkpoint_path, num_episodes)
    
    print("="*80)
    print("✅ Evaluation complete!")
    print("="*80)


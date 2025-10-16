#!/usr/bin/env python3
"""
Simple DMC Training Script

A simplified version that demonstrates DMC training without complex RLLib integration.
"""
import argparse
import json
import time
import torch
import numpy as np
from typing import Dict, Any

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guandan.rllib.models.q_model import GuandanQModel
from guandan.training.logger import Logger
from guandan.training.epsilon import EpsilonScheduler
from guandan.env.rllib_env import GuandanMultiAgentEnv


class SimpleDMCTrainer:
    """Simple DMC trainer for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize model
        self.model = GuandanQModel(
            hidden=config.get("model_hidden", (512, 512, 512, 512, 512)),
            activation=config.get("model_activation", "tanh"),
            orthogonal_init=config.get("model_orthogonal_init", True)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get("lr", 1e-3)
        )
        
        # Initialize epsilon scheduler
        self.epsilon_scheduler = EpsilonScheduler(
            start=config.get("epsilon_start", 0.2),
            end=config.get("epsilon_end", 0.05),
            decay_steps=config.get("epsilon_decay_steps", 10000)
        )
        
        # Initialize environment
        self.env = GuandanMultiAgentEnv(config.get("env_config", {}))
        
        # Training state
        self.update_count = 0
        self.episode_count = 0
        # Logger
        log_dir = config.get("log_dir", "results")
        run_name = config.get("run_name", f"dmc_simple_{int(time.time())}")
        self.logger = Logger(log_dir=log_dir, xpid=run_name)
        self.logger.save_meta({"config": config})
        
    def select_action(self, obs: np.ndarray, legal_actions: list, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon_scheduler.value():
            # Random action
            return np.random.randint(len(legal_actions))
        else:
            # Greedy action
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            legal_actions_tensor = torch.tensor(legal_actions, dtype=torch.float32)
            
            with torch.no_grad():
                q_values = self.model.evaluate_legal_actions(obs_tensor, legal_actions_tensor)
                return q_values.argmax().item()
    
    def run_episode(self) -> Dict[str, Any]:
        """Run a single episode."""
        obs, infos = self.env.reset()
        total_reward = {aid: 0.0 for aid in self.env.agent_ids}
        steps = 0
        samples = []
        
        terminated = {aid: False for aid in self.env.agent_ids}
        truncated = {aid: False for aid in self.env.agent_ids}
        
        while not any(terminated.values()) and steps < 1000:  # Max steps limit
            current = infos['agent_0']['current_player']
            current_id = f'agent_{current}'
            legal_encoded = infos[current_id].get('legal_actions_encoded_valid') or infos[current_id]['legal_actions_encoded']
            
            if len(legal_encoded) == 0:
                break
                
            tau = obs[current_id]
            action_idx = self.select_action(tau, legal_encoded, explore=True)
            
            actions = {aid: 0 for aid in self.env.agent_ids}
            actions[current_id] = action_idx
            
            obs, rewards, terminated, truncated, infos = self.env.step(actions)
            steps += 1
            
            for aid, r in rewards.items():
                total_reward[aid] += r
            
            # Store sample
            action_vec = np.asarray(legal_encoded[action_idx], dtype=np.float32)
            tau_t = np.asarray(tau, dtype=np.float32)
            samples.append({
                'tau': tau_t,
                'action': action_vec,
                'reward': r,
                'player_id': current
            })
        
        # Update epsilon
        self.epsilon_scheduler.step(steps)
        self.episode_count += 1
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'samples': samples
        }
    
    def train_step(self, samples: list):
        """Perform one training step."""
        if len(samples) < self.config.get("batch_size", 32):
            return 0.0
        
        # Sample batch
        batch_size = min(len(samples), self.config.get("batch_size", 32))
        batch_indices = np.random.choice(len(samples), size=batch_size, replace=False)
        
        # Prepare batch
        tau_batch = torch.tensor([samples[i]['tau'] for i in batch_indices], dtype=torch.float32)
        action_batch = torch.tensor([samples[i]['action'] for i in batch_indices], dtype=torch.float32)
        reward_batch = torch.tensor([samples[i]['reward'] for i in batch_indices], dtype=torch.float32)
        
        # Forward pass
        q_values = self.model(tau_batch, action_batch)
        
        # Compute loss (simple MSE with immediate rewards)
        loss = torch.mean((q_values - reward_batch) ** 2)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_count += 1
        # Log training loss
        self.logger.log({"train/loss": float(loss.item()), "train/updates": self.update_count})
        return float(loss.item())
    
    def train(self, num_episodes: int):
        """Train for specified number of episodes."""
        print(f"Starting training for {num_episodes} episodes...")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Run episode
            episode_data = self.run_episode()
            samples = episode_data['samples']
            total_reward = episode_data['total_reward']
            steps = episode_data['steps']
            
            # Train on samples
            loss = self.train_step(samples)
            
            # Track rewards
            mean_reward = np.mean(list(total_reward.values()))
            episode_rewards.append(mean_reward)
            # Per-episode logging
            self.logger.log({
                "reward/episode_mean": float(mean_reward),
                "episode/steps": int(steps),
                "epsilon/value": float(self.epsilon_scheduler.value()),
            })
            
            # Log progress
            if episode % 10 == 0 or episode == num_episodes - 1:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Reward={mean_reward:.3f}, "
                      f"Avg10={avg_reward:.3f}, "
                      f"Steps={steps}, "
                      f"Loss={loss:.6f}, "
                      f"Epsilon={self.epsilon_scheduler.value():.3f}, "
                      f"Updates={self.update_count}")
        
        print("Training completed!")
        self.logger.close()
        return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Simple DMC Training")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epsilon_start", type=float, default=0.2, help="Initial epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=1000, help="Epsilon decay steps")
    parser.add_argument("--log_dir", type=str, default="results", help="Directory for logs/TensorBoard")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name/xpid")
    
    args = parser.parse_args()
    
    # Create config
    config = {
        "env_config": {"observation_mode": "comprehensive", "use_internal_adapters": False},
        "model_hidden": (512, 512, 512, 512, 512),
        "model_activation": "tanh",
        "model_orthogonal_init": True,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "log_dir": args.log_dir,
        "run_name": args.run_name or f"dmc_simple_{int(time.time())}",
    }
    
    print(f"Training config: {config}")
    
    # Create trainer
    trainer = SimpleDMCTrainer(config)
    
    # Train
    rewards = trainer.train(args.episodes)
    
    # Save model
    torch.save(trainer.model.state_dict(), "checkpoints/dmc_simple_model.pt")
    print("Model saved to checkpoints/dmc_simple_model.pt")


if __name__ == "__main__":
    main()

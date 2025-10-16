"""
GPU-optimized rollout worker with batched action selection
"""
import numpy as np
import torch
from typing import Dict, Any, List
from collections import deque

from guandan.env.rllib_env import make_guandan_env
from guandan.rllib.models import GuandanQModel
from .epsilon import EpsilonScheduler
from .replay_buffer import Sample


def batch_select_actions_gpu(
    states: List[np.ndarray],
    legal_actions_list: List[List],
    model: GuandanQModel,
    epsilon: float,
    device: torch.device
) -> List[int]:
    """
    Batch action selection on GPU for multiple states simultaneously.
    
    Args:
        states: List of state vectors (513-D each)
        legal_actions_list: List of legal action lists for each state
        model: DMC Q-model
        epsilon: Exploration rate
        device: GPU device
    
    Returns:
        List of selected action indices
    """
    batch_size = len(states)
    if batch_size == 0:
        return []
    
    # Convert states to tensor
    states_tensor = torch.tensor(np.stack(states), dtype=torch.float32, device=device)  # (B, 513)
    
    # For each state, evaluate all legal actions
    selected_indices = []
    
    for i, (state, legal_actions) in enumerate(zip(states, legal_actions_list)):
        if len(legal_actions) == 0:
            selected_indices.append(0)
            continue
            
        # Convert legal actions to tensor
        legal_actions_tensor = torch.tensor(legal_actions, dtype=torch.float32, device=device)  # (L, 54)

        # Evaluate legal actions directly via model API
        with torch.no_grad():
            q_values = model.evaluate_legal_actions(states_tensor[i], legal_actions_tensor)
        
        # Epsilon-greedy selection
        if np.random.random() < epsilon:
            # Random action
            selected_idx = np.random.randint(len(legal_actions))
        else:
            # Greedy action
            selected_idx = q_values.argmax().item()
        
        selected_indices.append(selected_idx)
    
    return selected_indices


class GPURolloutWorker:
    """GPU-optimized rollout worker with batched action selection"""
    
    def __init__(
        self,
        wid: int,
        epsilon: float = 0.2,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        max_steps_per_episode: int = 2000,
        model_hidden: tuple = (512, 512, 512, 512, 512),
        model_activation: str = "tanh",
        model_orthogonal_init: bool = True,
        batch_size: int = 32,  # Batch size for action selection
    ):
        self.wid = wid
        self.epsilon = float(epsilon)
        self.eps_sched = EpsilonScheduler(start=epsilon, end=epsilon_end, decay_steps=epsilon_decay_steps)
        self.env = make_guandan_env({'observation_mode': 'comprehensive', 'use_internal_adapters': False})
        self.model = GuandanQModel(hidden=model_hidden, activation=model_activation, orthogonal_init=model_orthogonal_init)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.batch_size = batch_size
        
        # Batch collection for action selection
        self.state_batch = []
        self.legal_actions_batch = []
        self.batch_indices = []
    
    def run_episode(self) -> Dict[str, Any]:
        """Run a single episode with batched GPU action selection"""
        obs, infos = self.env.reset()
        total_reward = {aid: 0.0 for aid in self.env.agent_ids}
        samples = []
        steps = 0
        
        while steps < self.max_steps_per_episode:
            current = infos['agent_0']['current_player']
            current_id = f'agent_{current}'
            legal_encoded = infos[current_id].get('legal_actions_encoded_valid') or infos[current_id]['legal_actions_encoded']
            
            tau = obs[current_id]
            
            # Add to batch
            self.state_batch.append(tau)
            self.legal_actions_batch.append(legal_encoded)
            self.batch_indices.append(len(samples))
            
            # Process batch when full or at episode end
            if len(self.state_batch) >= self.batch_size:
                self._process_batch(samples)
            
            # Store sample placeholder
            if len(legal_encoded) > 0:
                action_vec = np.asarray(legal_encoded[0], dtype=np.float32)  # Placeholder
            else:
                action_vec = np.zeros((54,), dtype=np.float32)
            tau_t = np.asarray(tau, dtype=np.float32)
            samples.append(Sample(tau=tau_t, action54=action_vec, g=0.0, player_id=current))
            
            # Get action (will be updated by batch processing)
            if len(legal_encoded) > 0:
                idx = 0  # Placeholder, will be updated
            else:
                idx = 0
            
            actions = {aid: 0 for aid in self.env.agent_ids}
            actions[current_id] = idx
            obs, rewards, terminated, truncated, infos = self.env.step(actions)
            steps += 1
            
            for aid, r in rewards.items():
                total_reward[aid] += r
            
            if any(terminated.values()):
                break
        
        # Process remaining batch
        if len(self.state_batch) > 0:
            self._process_batch(samples)
        
        # Calculate final rewards and update samples
        if any(terminated.values()):
            final_rewards = rewards
            team1_total_reward = final_rewards.get('agent_0', 0.0) + final_rewards.get('agent_2', 0.0)
            team2_total_reward = final_rewards.get('agent_1', 0.0) + final_rewards.get('agent_3', 0.0)
            
            global_bonus_team1 = 1.0 if team1_total_reward > team2_total_reward else -1.0
            global_bonus_team2 = 1.0 if team2_total_reward > team1_total_reward else -1.0

            for i in range(len(samples)):
                pid = samples[i].player_id
                aid = f'agent_{pid}'
                
                cumulative_round_value = float(final_rewards.get(aid, 0.0))
                
                if pid == 0 or pid == 2:  # Team 1
                    g = cumulative_round_value + global_bonus_team1
                else:  # Team 2
                    g = cumulative_round_value + global_bonus_team2
                
                samples[i] = Sample(tau=samples[i].tau, action54=samples[i].action54, g=g, player_id=pid)
        
        return {
            'samples': samples,
            'reward': total_reward,
            'steps': steps,
        }
    
    def _process_batch(self, samples: List[Sample]):
        """Process a batch of states for action selection"""
        if len(self.state_batch) == 0:
            return
        
        # Batch action selection on GPU
        selected_indices = batch_select_actions_gpu(
            self.state_batch,
            self.legal_actions_batch,
            self.model,
            self.eps_sched.value(),
            self.device
        )
        
        # Update samples with selected actions
        for i, (batch_idx, selected_idx) in enumerate(zip(self.batch_indices, selected_indices)):
            if batch_idx < len(samples):
                legal_actions = self.legal_actions_batch[i]
                if len(legal_actions) > 0 and selected_idx < len(legal_actions):
                    action_vec = np.asarray(legal_actions[selected_idx], dtype=np.float32)
                    samples[batch_idx] = Sample(
                        tau=samples[batch_idx].tau,
                        action54=action_vec,
                        g=samples[batch_idx].g,
                        player_id=samples[batch_idx].player_id
                    )
        
        # Clear batch
        self.state_batch.clear()
        self.legal_actions_batch.clear()
        self.batch_indices.clear()

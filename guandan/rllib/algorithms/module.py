"""
Torch RLModule implementation for DanZero Deep Monte Carlo (DMC).

This module exposes the Guandan Q-network inside RLlib's new API stack. The
network takes 513-D state encodings and 54-D action encodings, concatenates
them, and predicts scalar Q-values. The module implements the required forward
APIs for inference/exploration/training.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override

from guandan.rllib.models.q_model import GuandanQModel


class DMCTorchRLModule(TorchRLModule):
    """RLModule exposing the Guandan Q network for DMC."""

    framework: str = "torch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau_dim = kwargs.get("tau_dim", 513)
        self.action_dim = kwargs.get("action_dim", 54)
        model_hidden = kwargs.get("model_hidden", (512, 512, 512, 512, 512))
        model_activation = kwargs.get("model_activation", "tanh")
        orthogonal_init = kwargs.get("model_orthogonal_init", True)

        self.q_model = GuandanQModel(
            tau_dim=self.tau_dim,
            action_dim=self.action_dim,
            hidden=model_hidden,
            activation=model_activation,
            orthogonal_init=orthogonal_init,
        )
        
    @property
    def device(self):
        """Get the device of the Q-model parameters."""
        return next(self.q_model.parameters()).device

    @override(RLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # During inference, compute Q-values for all legal actions to select the best one
        return self._forward_all_legal_actions(batch, **kwargs)

    @override(RLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # During exploration, compute Q-values for all legal actions for epsilon-greedy selection
        return self._forward_all_legal_actions(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # During training, only compute Q-value for the action that was taken (efficient!)
        return self._forward_taken_action(batch, **kwargs)

    def _forward_taken_action(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Efficient forward pass for training: only compute Q for the action that was taken.
        
        This is much more memory-efficient than computing Q for all actions.
        """
        # Get tau (state encoding)
        tau = batch.get("tau")
        if tau is None:
            tau = batch.get(Columns.OBS)
        
        # Get the action that was taken
        action = batch.get(Columns.ACTIONS)
        
        if tau is None or action is None:
            raise ValueError("Batch must contain 'tau' and 'actions' for training.")
        
        # Convert to tensors if needed
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
        
        # Move to device
        tau = tau.to(self.device)
        action = action.to(self.device)
        
        # Ensure proper shapes
        if len(tau.shape) == 1:
            tau = tau.unsqueeze(0)
        if len(action.shape) == 0:
            action = action.unsqueeze(0)
        
        batch_size = tau.shape[0]
        
        # Create one-hot encoding for the taken actions
        action_encoding = torch.zeros((batch_size, 54), device=self.device)
        action_encoding[torch.arange(batch_size), action] = 1.0
        
        # Compute Q-value only for the taken action
        q_value = self.q_model(tau, action_encoding)  # (batch_size,)
        
        # For training, we return a simplified structure
        # The learner will extract the Q-value directly
        return {
            "q_values": q_value.unsqueeze(-1),  # (batch_size, 1) for consistency
            Columns.ACTION_DIST_INPUTS: q_value.unsqueeze(-1)
        }
    
    def _forward_all_legal_actions(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Forward pass that evaluates Q-values for all legal actions (for action selection).
        
        For DMC, we need to compute Q(s, a) for each legal action a in state s.
        Returns Q-values for all 54 actions (batch_size, 54) with -inf for illegal actions.
        """
        # Try to get tau from various possible keys
        tau = batch.get("tau")
        if tau is None:
            tau = batch.get(Columns.OBS)
        
        # Try to get legal actions mask from various possible keys
        legal_actions_mask = batch.get("legal_actions")
        if legal_actions_mask is None:
            # If no legal actions provided, assume all actions are legal
            legal_actions_mask = torch.ones((1, 54), dtype=torch.float32, device=self.device)

        if tau is None:
            raise ValueError("Batch must contain 'tau' or 'obs' tensors.")

        # Convert to tensors if needed and ensure proper device placement
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32)
        if not isinstance(legal_actions_mask, torch.Tensor):
            legal_actions_mask = torch.tensor(legal_actions_mask, dtype=torch.float32)
        
        # Move to correct device
        tau = tau.to(self.device)
        legal_actions_mask = legal_actions_mask.to(self.device)

        # Ensure tau has batch dimension: (batch_size, tau_dim)
        if len(tau.shape) == 1:
            tau = tau.unsqueeze(0)
        batch_size = tau.shape[0]
        
        # Ensure legal_actions_mask matches batch size: (batch_size, 54)
        if len(legal_actions_mask.shape) == 1:
            legal_actions_mask = legal_actions_mask.unsqueeze(0)
        if legal_actions_mask.shape[0] != batch_size:
            if legal_actions_mask.shape[0] == 1:
                legal_actions_mask = legal_actions_mask.expand(batch_size, -1)
            else:
                legal_actions_mask = legal_actions_mask[:batch_size]
        
        # Efficient batched Q-value computation for legal actions only
        # Process in mini-batches to avoid OOM
        num_actions = 54
        max_chunk_size = 128  # Process at most 128 (state, action) pairs at once to conserve GPU memory
        
        # Build lists of batch indices, tau values, and action indices for all legal actions
        batch_indices = []
        tau_list = []
        action_indices_list = []
        
        for i in range(batch_size):
            legal_mask = legal_actions_mask[i] > 0
            legal_indices = torch.where(legal_mask)[0]
            num_legal = len(legal_indices)
            
            if num_legal > 0:
                # Add this state repeated for each legal action
                batch_indices.extend([i] * num_legal)
                tau_list.append(tau[i].unsqueeze(0).expand(num_legal, -1))
                action_indices_list.append(legal_indices)
        
        # Initialize Q-values with -inf
        q_values = torch.full((batch_size, num_actions), float('-inf'), device=self.device)
        
        if len(tau_list) > 0:
            # Stack all tau and action encodings
            tau_batch = torch.cat(tau_list, dim=0)  # (total_legal_actions, 513)
            action_indices_batch = torch.cat(action_indices_list)  # (total_legal_actions,)
            
            # Process in chunks to avoid OOM
            total_legal = len(action_indices_batch)
            q_values_list = []
            
            for chunk_start in range(0, total_legal, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, total_legal)
                
                # Get chunk of data
                tau_chunk = tau_batch[chunk_start:chunk_end]
                action_indices_chunk = action_indices_batch[chunk_start:chunk_end]
                
                # Create one-hot action encodings for this chunk
                action_encodings_chunk = torch.zeros((len(action_indices_chunk), num_actions), device=self.device)
                action_encodings_chunk[torch.arange(len(action_indices_chunk)), action_indices_chunk] = 1.0
                
                # Evaluate Q-values for this chunk
                q_chunk = self.q_model(tau_chunk, action_encodings_chunk)
                q_values_list.append(q_chunk)
            
            # Concatenate all chunks
            q_values_legal = torch.cat(q_values_list, dim=0)
            
            # Scatter results back to the output tensor
            for idx, (b_idx, a_idx) in enumerate(zip(batch_indices, action_indices_batch)):
                q_values[b_idx, a_idx] = q_values_legal[idx]
        
        return {
            "q_values": q_values,  # (batch_size, 54)
            "legal_actions": legal_actions_mask,  # (batch_size, 54)
            Columns.ACTION_DIST_INPUTS: q_values  # (batch_size, 54)
        }

    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Legacy forward pass for compatibility."""
        tau = batch.get("tau")
        action = batch.get("action")

        if tau is None or action is None:
            raise ValueError("Batch must contain 'tau' and 'action' tensors.")

        q_values = self.q_model(tau, action)
        return {"q_values": q_values, Columns.ACTION_DIST_INPUTS: q_values.unsqueeze(-1)}

    @override(RLModule)
    def get_initial_state(self) -> Dict[str, torch.Tensor]:
        return {}

    @override(RLModule)
    def is_stateful(self) -> bool:
        return False



"""
Learner implementation for DanZero Deep Monte Carlo (DMC).

This builds on RLlib's TorchLearner to apply MSE regression between predicted
Q-values and paper-style targets (g) coming from rollouts.
"""

from __future__ import annotations

from typing import Dict, Any

import torch
import numpy as np

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType


class DMCLearner(TorchLearner):
    """TorchLearner for DMC MSE loss."""

    @override(TorchLearner)
    def configure_optimizers_for_module(
        self, module_id: ModuleID, config: AlgorithmConfig
    ) -> None:
        module = self.module[module_id]
        optimizer = torch.optim.Adam(
            module.q_model.parameters(), lr=float(config.get("lr", 1e-3))
        )
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="adam",
            optimizer=optimizer,
            params=list(module.q_model.parameters()),
        )

    @override(TorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: AlgorithmConfig,
        batch: Dict[str, TensorType],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        # Extract Q-values from forward pass
        # During training, q_values is (batch_size, 1) - just the Q for the taken action
        q_values = fwd_out.get("q_values")
        
        if q_values is None:
            # If no q_values, return zero loss (this can happen during initialization)
            # Get device from module
            module = self.module[module_id]
            device = next(module.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Squeeze to (batch_size,)
        if q_values.ndim > 1:
            q_values = q_values.squeeze(-1)
        
        # Extract targets - could be in 'g' or 'rewards' column
        targets = batch.get("g")
        if targets is None:
            targets = batch.get(Columns.REWARDS)
        if targets is None:
            # If no targets available yet (e.g., during first iteration), return zero loss
            return torch.tensor(0.0, device=q_values.device, requires_grad=True)
        
        # Ensure targets are on the same device as q_values
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32, device=q_values.device)
        else:
            targets = targets.to(q_values.device)
        
        # Flatten targets if needed
        if targets.ndim > 1:
            targets = targets.squeeze()
        
        # Compute MSE loss
        loss = torch.mean((q_values - targets) ** 2)
        
        # Log metrics
        self.metrics.log_value((module_id, "loss"), loss.detach().item(), window=1)
        self.metrics.log_value((module_id, "q_values_mean"), torch.mean(q_values).detach().item(), window=1)
        self.metrics.log_value((module_id, "targets_mean"), torch.mean(targets).detach().item(), window=1)
        
        # Log Guandan-specific metrics for analysis
        # Track individual agent rewards (from targets which represent returns)
        self.metrics.log_value((module_id, "episode_return_mean"), torch.mean(targets).detach().item(), window=1)
        self.metrics.log_value((module_id, "episode_return_std"), torch.std(targets).detach().item(), window=1)
        self.metrics.log_value((module_id, "q_value_std"), torch.std(q_values).detach().item(), window=1)
        
        return loss

    def _convert_batch_type(
        self,
        batch: Any,
        to_device: bool = False,
        pin_memory: bool = False,
        use_stream: bool = False,
    ) -> Any:
        """Convert batch to appropriate tensor types for training."""
        # For now, assume the batch is already in the correct format
        # This method can be extended if needed for specific tensor conversions
        return batch



from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from guandan.rllib.models import GuandanQModel

class Learner:
    """梯度更新占位类.
    日后负责: 接收批次 -> 前向 -> 计算损失 -> 反向 -> 更新参数, 并推送到参数服务器.
    """
    def __init__(self, model: Any = None, lr: float = 1e-3, lambda_clip: float = 0.2, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model: GuandanQModel = model if model is not None else GuandanQModel()
        self.device = device
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lambda_clip = float(lambda_clip)
        self.global_step = 0
        # Mixed precision training for better GPU utilization
        self.scaler = GradScaler() if self.device == "cuda" else None
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device == "cuda":
            self.model = torch.compile(self.model)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Old-code DMC loss: mean((Q(s,a) - g) ** 2)
        batch keys expected: tau (B,513), action (B,54), g (B,)  [temporary: accepts 'reward' as g]
        """
        tau = torch.tensor(batch['tau'], dtype=torch.float32, device=self.device)
        action = torch.tensor(batch['action'], dtype=torch.float32, device=self.device)
        # accept 'g' or fallback to 'reward' for compatibility
        g_arr = batch.get('g', batch.get('reward'))
        g = torch.tensor(g_arr, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            # Mixed precision training
            with autocast():
                q_learner = self.model(tau, action)  # (B,)
                loss = torch.mean((q_learner - g) ** 2)
            
            self.scaler.scale(loss).backward()
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            q_learner = self.model(tau, action)  # (B,)
            loss = torch.mean((q_learner - g) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.global_step += 1
        return {'loss': float(loss.item()), 'step': float(self.global_step)}

    def update_gpu_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update with GPU tensors (no CPU-GPU transfer needed).
        batch keys: tau (B,513), action (B,54), g (B,)
        """
        tau = batch['tau']  # Already on GPU
        action = batch['action']  # Already on GPU
        g = batch['g']  # Already on GPU

        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            # Mixed precision training
            with autocast():
                q_learner = self.model(tau, action)  # (B,)
                loss = torch.mean((q_learner - g) ** 2)
            
            self.scaler.scale(loss).backward()
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            q_learner = self.model(tau, action)  # (B,)
            loss = torch.mean((q_learner - g) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.global_step += 1
        return {'loss': float(loss.item()), 'step': float(self.global_step)}

__all__ = ['Learner']


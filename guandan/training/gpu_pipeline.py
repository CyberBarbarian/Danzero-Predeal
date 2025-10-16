"""
GPU-accelerated data pipeline for DMC training
"""
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque

from .replay_buffer import Sample


class GPUTensorBuffer:
    """GPU-based replay buffer that keeps tensors on GPU"""
    
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.gpu_tensors = {
            'tau': None,
            'action': None,
            'g': None,
            'player_id': None
        }
        self.size = 0
    
    def add(self, sample: Sample):
        """Add a sample to the buffer"""
        self.buffer.append(sample)
        self.size = len(self.buffer)
    
    def _ensure_gpu_tensors(self):
        """Ensure GPU tensors are allocated and up-to-date"""
        if self.size == 0:
            return
        
        # Convert to tensors if not already done
        if self.gpu_tensors['tau'] is None or self.gpu_tensors['tau'].size(0) != self.size:
            taus = []
            actions = []
            gs = []
            player_ids = []
            
            for sample in self.buffer:
                taus.append(sample.tau)
                actions.append(sample.action54)
                gs.append(sample.g)
                player_ids.append(sample.player_id)
            
            self.gpu_tensors['tau'] = torch.tensor(np.stack(taus), dtype=torch.float32, device=self.device)
            self.gpu_tensors['action'] = torch.tensor(np.stack(actions), dtype=torch.float32, device=self.device)
            self.gpu_tensors['g'] = torch.tensor(gs, dtype=torch.float32, device=self.device)
            self.gpu_tensors['player_id'] = torch.tensor(player_ids, dtype=torch.long, device=self.device)
    
    def sample_batch_gpu(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch directly on GPU without CPU-GPU transfer"""
        if self.size < batch_size:
            raise ValueError(f"Buffer size {self.size} < batch_size {batch_size}")
        
        self._ensure_gpu_tensors()
        
        # Sample indices on GPU
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return {
            'tau': self.gpu_tensors['tau'][indices],
            'action': self.gpu_tensors['action'][indices],
            'g': self.gpu_tensors['g'][indices],
            'player_id': self.gpu_tensors['player_id'][indices]
        }
    
    def __len__(self):
        return self.size


class GPUDataLoader:
    """GPU-optimized data loader for training"""
    
    def __init__(self, buffer: GPUTensorBuffer, batch_size: int, device: torch.device):
        self.buffer = buffer
        self.batch_size = batch_size
        self.device = device
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get a batch of data directly on GPU"""
        return self.buffer.sample_batch_gpu(self.batch_size)


def create_gpu_pipeline(capacity: int, batch_size: int, device: torch.device) -> Tuple[GPUTensorBuffer, GPUDataLoader]:
    """Create a GPU-optimized data pipeline"""
    buffer = GPUTensorBuffer(capacity, device)
    loader = GPUDataLoader(buffer, batch_size, device)
    return buffer, loader

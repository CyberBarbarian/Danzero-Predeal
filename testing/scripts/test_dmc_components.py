#!/usr/bin/env python3
"""
Test DMC Components

Simple test to verify DMC components work correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from guandan.rllib.models.q_model import GuandanQModel
from guandan.training.epsilon import EpsilonScheduler


def test_q_model():
    """Test GuandanQModel."""
    print("Testing GuandanQModel...")
    
    model = GuandanQModel()
    print(f"Model created successfully")
    
    # Test forward pass
    batch_size = 4
    tau = torch.randn(batch_size, 513)
    action = torch.randn(batch_size, 54)
    
    q_values = model(tau, action)
    print(f"Forward pass successful: {q_values.shape}")
    
    # Test legal actions evaluation
    legal_actions = torch.randn(10, 54)  # 10 legal actions
    q_legal = model.evaluate_legal_actions(tau[0], legal_actions)
    print(f"Legal actions evaluation successful: {q_legal.shape}")
    
    return True


def test_epsilon_scheduler():
    """Test EpsilonScheduler."""
    print("Testing EpsilonScheduler...")
    
    scheduler = EpsilonScheduler(start=0.2, end=0.05, decay_steps=100)
    print(f"Initial epsilon: {scheduler.value()}")
    
    # Test decay
    for i in range(10):
        scheduler.step(10)
        print(f"Step {i*10}: epsilon = {scheduler.value():.3f}")
    
    return True


def main():
    """Run all tests."""
    print("Running DMC component tests...\n")
    
    try:
        test_q_model()
        print("✅ Q Model test passed\n")
        
        test_epsilon_scheduler()
        print("✅ Epsilon Scheduler test passed\n")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

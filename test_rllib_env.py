#!/usr/bin/env python3
"""
Test script for Guandan RLLib MultiAgentEnv wrapper.

This script tests the basic functionality of the RLLib environment wrapper
to ensure it works correctly before integrating with Ray training.

Usage:
    python test_rllib_env.py
"""

import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'guandan'))

from guandan.env.rllib_env import make_guandan_env
from guandan.env.observation_extractor import extract_observation


def test_environment_creation():
    """Test basic environment creation and initialization."""
    print("Testing environment creation...")
    
    try:
        env = make_guandan_env()
        print(f"✓ Environment created successfully")
        print(f"  - Agent IDs: {env.agent_ids}")
        print(f"  - Observation spaces: {[space.shape for space in env.observation_spaces.values()]}")
        print(f"  - Action spaces: {[space.n for space in env.action_spaces.values()]}")
        return env
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return None


def test_reset_functionality(env):
    """Test environment reset functionality."""
    print("\nTesting reset functionality...")
    
    try:
        observations, infos = env.reset()
        print(f"✓ Reset successful")
        print(f"  - Number of agents: {len(observations)}")
        print(f"  - Observation shapes: {[obs.shape for obs in observations.values()]}")
        print(f"  - Info keys: {list(infos.keys())}")
        return True
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False


def test_step_functionality(env):
    """Test environment step functionality."""
    print("\nTesting step functionality...")
    
    try:
        # Reset first
        observations, infos = env.reset()
        
        # Create random actions for all agents
        actions = {agent_id: 0 for agent_id in env.agent_ids}
        
        # Take a step
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        print(f"✓ Step successful")
        print(f"  - Observation shapes: {[obs.shape for obs in obs.values()]}")
        print(f"  - Rewards: {rewards}")
        print(f"  - Terminated: {terminateds}")
        print(f"  - Truncated: {truncateds}")
        return True
    except Exception as e:
        print(f"✗ Step failed: {e}")
        return False


def test_observation_extractor():
    """Test observation extraction from JSON messages."""
    print("\nTesting observation extractor...")
    
    try:
        # Test with a sample JSON message
        sample_message = '''
        {
            "type": "act",
            "stage": "play",
            "handCards": ["H2", "H3", "S4", "C5"],
            "myPos": 0,
            "selfRank": 1,
            "oppoRank": 1,
            "curRank": 1,
            "publicInfo": [{"rest": 27}, {"rest": 27}, {"rest": 27}, {"rest": 27}],
            "actionList": [["PASS", "PASS", "PASS"], ["Single", "2", ["H2"]]],
            "curAction": ["PASS", "PASS", "PASS"],
            "greaterAction": ["PASS", "PASS", "PASS"]
        }
        '''
        
        obs = extract_observation(sample_message, 0)
        print(f"✓ Observation extraction successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  - Non-zero elements: {np.count_nonzero(obs)}")
        return True
    except Exception as e:
        print(f"✗ Observation extraction failed: {e}")
        return False


def test_multiple_episodes(env, num_episodes=3):
    """Test running multiple episodes."""
    print(f"\nTesting multiple episodes ({num_episodes} episodes)...")
    
    try:
        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}:")
            
            # Reset
            observations, infos = env.reset()
            print(f"    - Reset: {len(observations)} agents")
            
            # Run a few steps
            for step in range(5):
                actions = {agent_id: np.random.randint(0, 10) for agent_id in env.agent_ids}
                obs, rewards, terminateds, truncateds, infos = env.step(actions)
                
                if any(terminateds.values()) or any(truncateds.values()):
                    print(f"    - Episode ended at step {step + 1}")
                    break
            
            print(f"    - Episode {episode + 1} completed")
        
        print(f"✓ Multiple episodes test successful")
        return True
    except Exception as e:
        print(f"✗ Multiple episodes test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Guandan RLLib Environment Test Suite")
    print("=" * 60)
    
    # Test 1: Environment creation
    env = test_environment_creation()
    if env is None:
        print("\n❌ Environment creation failed. Stopping tests.")
        return
    
    # Test 2: Reset functionality
    reset_success = test_reset_functionality(env)
    
    # Test 3: Step functionality
    step_success = test_step_functionality(env)
    
    # Test 4: Observation extractor
    obs_success = test_observation_extractor()
    
    # Test 5: Multiple episodes
    episodes_success = test_multiple_episodes(env, 3)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Environment Creation: {'✓' if env else '✗'}")
    print(f"  Reset Functionality:  {'✓' if reset_success else '✗'}")
    print(f"  Step Functionality:   {'✓' if step_success else '✗'}")
    print(f"  Observation Extractor: {'✓' if obs_success else '✗'}")
    print(f"  Multiple Episodes:    {'✓' if episodes_success else '✗'}")
    
    all_passed = all([env, reset_success, step_success, obs_success, episodes_success])
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 The RLLib environment wrapper is working correctly!")
        print("   You can now proceed with Ray integration.")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")


if __name__ == "__main__":
    main()

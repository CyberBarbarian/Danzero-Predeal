"""
DMC (Deep Monte Carlo) Algorithm for RLlib.

Provides integration with RLlib's new API stack (RLModule + Learner)
for distributed experience collection and Q(s, a) regression onto 
Monte Carlo returns.
"""

from __future__ import annotations

from typing import Dict

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override


class DMCConfig(AlgorithmConfig):
    """Configuration for DMC algorithm."""
    
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or DMC)
        
        # DMC-specific hyperparameters (from paper)
        self.lr = 1e-3
        self.epsilon_start = 0.2
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 10000
        self.batch_size = 64
        self.replay_buffer_size = 100000
        self.target_update_freq = 1000
        self.gradient_clip_norm = 1.0
        
        # Model configuration (from paper: 5 layers × 512 hidden units with Tanh)
        self.model_hidden = (512, 512, 512, 512, 512)  # 5-layer network as in paper
        self.model_activation = "tanh"  # Tanh activation as in paper
        self.model_orthogonal_init = True
        
        # Training configuration
        self.train_batch_size = 64
        self.sgd_minibatch_size = 64
        self.num_sgd_iter = 1
        
        # New API stack configuration
        self.tau_dim = 513
        self.action_dim = 54
    
    def get_default_rl_module_spec(self):
        """Return default RL module spec for DMC."""
        # Import here to avoid circular imports
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        from guandan.rllib.algorithms.module import DMCTorchRLModule
        
        return RLModuleSpec(
            module_class=DMCTorchRLModule,
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config={
                "tau_dim": self.tau_dim,
                "action_dim": self.action_dim,
                "model_hidden": self.model_hidden,
                "model_activation": self.model_activation,
                "model_orthogonal_init": self.model_orthogonal_init,
            }
        )


class DMC(Algorithm):
    """RLlib Algorithm implementation for DMC using the new API stack."""

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return DMCConfig()
    
    @override(Algorithm)
    def training_step(self) -> Dict:
        """Custom training step that properly handles metrics initialization and learner results format."""
        # Monkey-patch the learner_group.update to fix list return value
        original_learner_update = self.learner_group.update
        
        def patched_learner_update(*args, **kwargs):
            """Wrapper that converts list results to dict."""
            results = original_learner_update(*args, **kwargs)
            
            # If results is a list with a single dict, unwrap it
            if isinstance(results, list):
                if len(results) == 1 and isinstance(results[0], dict):
                    return results[0]
                elif len(results) > 1:
                    # Multiple dicts - merge them
                    merged = {}
                    for item in results:
                        if isinstance(item, dict):
                            merged.update(item)
                    return merged if merged else {}
                else:
                    return {}
            return results
        
        self.learner_group.update = patched_learner_update
        
        # Temporarily monkey-patch metrics.log_dict to handle Stats objects
        original_log_dict = self.metrics.log_dict
        
        def convert_stats_to_values(data):
            """Recursively convert Stats objects to their values."""
            # Check if it's a Stats object by checking for peek() method
            # Stats objects have __class__.__name__ == 'Stats'
            if hasattr(data, '__class__') and data.__class__.__name__ == 'Stats':
                # Convert Stats object to its peek value
                return data.peek() if hasattr(data, 'peek') else data
            elif isinstance(data, dict):
                return {k: convert_stats_to_values(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_stats_to_values(item) for item in data]
            else:
                return data
        
        def patched_log_dict(stats_dict, *args, **kwargs):
            """Wrapper that unwraps list-formatted stats and converts Stats objects before logging."""
            try:
                # If stats_dict is a list with a single dict, unwrap it
                if isinstance(stats_dict, list):
                    if len(stats_dict) == 1 and isinstance(stats_dict[0], dict):
                        stats_dict = stats_dict[0]
                    elif len(stats_dict) == 0:
                        # Empty list, return empty dict
                        stats_dict = {}
                    else:
                        # Multiple items - merge them
                        merged = {}
                        for item in stats_dict:
                            if isinstance(item, dict):
                                merged.update(item)
                        stats_dict = merged if merged else {}
                
                # Ensure we have a dict
                if not isinstance(stats_dict, dict):
                    stats_dict = {}
                
                # Convert all Stats objects to values to avoid deprecation warning
                stats_dict = convert_stats_to_values(stats_dict)
                
                return original_log_dict(stats_dict, *args, **kwargs)
            except Exception as e:
                # Silently handle to avoid interrupting training
                return original_log_dict({}, *args, **kwargs)
        
        self.metrics.log_dict = patched_log_dict
        
        try:
            # Use the parent class implementation which handles the new API stack
            return super().training_step()
        except (KeyError, AttributeError, UnboundLocalError) as e:
            # Handle missing metrics on first training step
            error_str = str(e)
            if "num_env_steps_sampled_lifetime" in error_str:
                # Initialize the metric to 0 and try again
                from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
                self.metrics.log_value(NUM_ENV_STEPS_SAMPLED_LIFETIME, 0, reduce="sum")
                return super().training_step()
            elif "'list' object has no attribute 'keys'" in error_str:
                # Learner returned results as list - this should be caught by our patch
                # If we get here, try one more time
                print("Warning: Learner results format issue, retrying...")
                return super().training_step()
            elif isinstance(e, UnboundLocalError):
                # RLlib Learner bug where loss_per_module may be undefined when no data
                # Return an empty result to allow outer loop to continue and log
                return {}
            else:
                raise
        finally:
            # Restore original methods
            self.metrics.log_dict = original_log_dict
            self.learner_group.update = original_learner_update

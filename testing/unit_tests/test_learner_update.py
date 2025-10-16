import numpy as np

from guandan.training.learner import Learner


def test_learner_update_runs():
    learner = Learner()
    B = 8
    batch = {
        'tau': np.zeros((B, 513), dtype=np.float32),
        'action': np.zeros((B, 54), dtype=np.float32),
        'q_actor': np.ones((B,), dtype=np.float32),
        'reward': np.zeros((B,), dtype=np.float32),
    }
    out1 = learner.update(batch)
    assert 'loss' in out1 and 'step' in out1
    out2 = learner.update(batch)
    assert out2['step'] > out1['step']



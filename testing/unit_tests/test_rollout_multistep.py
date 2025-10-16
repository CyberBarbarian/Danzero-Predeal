from guandan.training.rollout_worker import RolloutWorker


def test_rollout_multistep_and_rewards_assignment():
    worker = RolloutWorker(wid=0, epsilon=1.0)
    out = worker.run_episode()
    assert out['steps'] >= 1
    # Samples should be non-empty and have reward fields set (0 if not terminal)
    assert len(out['samples']) >= 1
    assert hasattr(out['samples'][0], 'reward')



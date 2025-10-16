from guandan.training.epsilon import EpsilonScheduler


def test_epsilon_scheduler_monotonic():
    sched = EpsilonScheduler(start=0.5, end=0.1, decay_steps=10)
    prev = 1.0
    for _ in range(12):
        val = sched.value()
        assert val <= prev + 1e-6
        prev = val
        sched.step()



class EpsilonScheduler:
    def __init__(self, start: float = 0.2, end: float = 0.05, decay_steps: int = 100000):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = int(decay_steps)
        self.t = 0

    def value(self) -> float:
        if self.t >= self.decay_steps:
            return self.end
        frac = 1.0 - (self.t / max(1, self.decay_steps))
        return self.end + (self.start - self.end) * frac

    def step(self, n: int = 1) -> float:
        self.t += int(n)
        return self.value()



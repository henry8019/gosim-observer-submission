"""Small causal AR(1) estimate fitted only to consecutive received samples."""
from collections import deque


class ObservedWeather:
    def __init__(self):
        self.pairs = deque(maxlen=64)
        self.previous = None
        self.current = None
        self.origin = None

    def observe(self, origin, quality):
        if origin == self.origin:
            return
        if self.previous and quality is not None:
            old_time, old_quality = self.previous
            if origin-old_time == 900 and old_quality is not None:
                self.pairs.append((old_quality, quality))
        self.previous = origin, quality
        self.origin, self.current = origin, quality

    def predict(self, steps, blend):
        if self.current is None:
            return None
        if steps <= 0 or len(self.pairs) < 12:
            return self.current
        mx = sum(x for x, _ in self.pairs)/len(self.pairs)
        my = sum(y for _, y in self.pairs)/len(self.pairs)
        variance = sum((x-mx)**2 for x, _ in self.pairs)
        if variance < 1e-12:
            return self.current
        phi = max(0, min(1, sum((x-mx)*(y-my) for x, y in self.pairs)/variance))
        intercept = my-phi*mx
        estimate = self.current
        for _ in range(int(steps)):
            estimate = max(.05, min(3, intercept+phi*estimate))
        return self.current*(1-blend)+estimate*blend

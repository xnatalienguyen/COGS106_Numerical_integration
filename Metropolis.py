import numpy as np
from statistics import NormalDist


class Metropolis:
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.state = initialState
        self.accepted = 0
        self.proposed = 0
        self.samples = []

    def accept(self, proposal):
        acceptance_prob = min(1, np.exp(self.logTarget(
            proposal) - self.logTarget(self.state)))
        self.proposed += 1
        if acceptance_prob > np.random.uniform():
            self.state = proposal
            self.accepted += 1
            return True
        else:
            return False

    def adapt(self, blockLengths):
        for block in range(len(blockLengths)):
            for i in range(blockLengths[block]):
                proposal = np.random.normal(self.state, self.stdDev)
                self.accept(proposal)
            acceptance_rate = self.accepted / self.proposed
            if acceptance_rate < 0.1:
                self.stdDev /= 2
            elif acceptance_rate > 0.3:
                self.stdDev *= 2

    def sample(self, n):
        for i in range(n):
            proposal = np.random.normal(self.state, self.stdDev)
            if self.accept(proposal):
                self.samples.append(self.state)

    def summary(self):
        mean = np.mean(self.samples)
        ci = NormalDist().inv_cdf(0.975) * np.std(self.samples) / \
            np.sqrt(len(self.samples))
        return {"mean": mean, "95% CI": (mean - ci, mean + ci)}

import random


# -------------------------
# Greedy Policy
# -------------------------
class GreedyPolicy:
    def choose_action(self, state):
        aoi, battery, _ = state

        if battery > 0:
            return 1  # try to transmit
        return 0


# -------------------------
# Periodic Policy
# -------------------------
class PeriodicPolicy:
    def __init__(self, interval=2):
        self.interval = interval
        self.counter = 0

    def choose_action(self, state):
        self.counter += 1

        if self.counter % self.interval == 0:
            return 1
        return 0


# -------------------------
# Random Policy (optional)
# -------------------------
class RandomPolicy:
    def choose_action(self, state):
        return random.choice([0, 1])
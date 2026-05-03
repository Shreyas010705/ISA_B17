import random

# -------------------------------
# 1. Greedy Policy
# -------------------------------
class GreedyPolicy:
    """
    Always tries to transmit if possible
    (if battery > 0 and duty cycle allows)
    """

    def choose_action(self, state):
        aoi, battery, time_since_last_tx = state

        # If battery is available → always transmit
        if battery > 0:
            return 1  # transmit
        else:
            return 0  # wait


# -------------------------------
# 2. Periodic Policy
# -------------------------------
class PeriodicPolicy:
    """
    Transmits every N steps (fixed interval)
    Example: interval=2 → transmit every 2 steps
    """

    def __init__(self, interval=2):
        self.interval = interval
        self.counter = 0

    def choose_action(self, state):
        self.counter += 1

        if self.counter >= self.interval:
            self.counter = 0
            return 1  # transmit
        else:
            return 0  # wait


# -------------------------------
# 3. Random Policy (optional baseline)
# -------------------------------
class RandomPolicy:
    """
    Takes random actions
    (used as a weak baseline)
    """

    def choose_action(self, state):
        return random.choice([0, 1])
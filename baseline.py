import random


# -------------------------
# GREEDY POLICY
# -------------------------
class GreedyPolicy:
    def reset(self):
        # No internal state, but added for consistency
        pass

    def choose_action(self, state):
        aoi, battery, sleep = state

        # Cannot act during sleep
        if sleep == 1:
            return 0

        # Always send if possible
        if battery > 0:
            return 1

        return 0


# -------------------------
# PERIODIC POLICY
# -------------------------
class PeriodicPolicy:
    def __init__(self, interval=5):
        self.interval = interval
        self.reset()

    def reset(self):
        self.counter = 0

    def choose_action(self, state):
        _, battery, sleep = state

        # Cannot act during sleep
        if sleep == 1:
            return 0

        self.counter += 1

        # Send every "interval" steps if energy available
        if self.counter % self.interval == 0 and battery > 0:
            return 1

        return 0


# -------------------------
# RANDOM POLICY (OPTIONAL)
# -------------------------
class RandomPolicy:
    def reset(self):
        pass

    def choose_action(self, state):
        _, battery, sleep = state

        # Cannot act during sleep
        if sleep == 1:
            return 0

        # Only choose valid actions
        if battery > 0:
            return random.choice([0, 1])
        else:
            return 0


# -------------------------
# THRESHOLD POLICY (NEW)
# -------------------------
class ThresholdPolicy:
    def __init__(self, aoi_threshold=5, battery_threshold=2):
        self.aoi_threshold = aoi_threshold
        self.battery_threshold = battery_threshold

    def reset(self):
        # No internal state, for consistency
        pass

    def choose_action(self, state):
        aoi, battery, sleep = state

        # Cannot act during sleep
        if sleep == 1:
            return 0

        # Cannot send without energy
        if battery <= 0:
            return 0

        # Threshold-based decision
        if aoi >= self.aoi_threshold and battery >= self.battery_threshold:
            return 1  # send
        else:
            return 0  # wait
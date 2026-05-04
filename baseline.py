import random


class GreedyPolicy:
    def choose_action(self, state):
        aoi, battery, sleep = state

        if sleep == 1:
            return 0

        if battery > 0:
            return 1
        return 0


class PeriodicPolicy:
    def __init__(self, interval=2):
        self.interval = interval
        self.counter = 0

    def choose_action(self, state):
        _, _, sleep = state

        if sleep == 1:
            return 0

        self.counter += 1

        if self.counter % self.interval == 0:
            return 1
        return 0


class RandomPolicy:
    def choose_action(self, state):
        return random.choice([0, 1])

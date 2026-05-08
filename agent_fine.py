import random


class QLearningAgent:
    def __init__(self):
        self.q_table = {}

        # Learning parameters
        self.alpha = 0.1
        self.gamma = 0.9

        # Exploration
        self.epsilon = 0.2
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.actions = [0, 1]

    # -------------------------
    # STATE DISCRETIZATION
    # -------------------------
    def discretize_state(self, state):
        aoi, battery, sleep = state

        # AoI bins
        if aoi <= 2:
            aoi_bin = 0
        elif aoi <= 5:
            aoi_bin = 1
        elif aoi <= 10:
            aoi_bin = 2
        else:
            aoi_bin = 3
            
        # Battery bins
        if battery <= 1:
            battery_bin = 0
        elif battery <= 3:
            battery_bin = 1
        else:
            battery_bin = 2

        return (aoi_bin, battery_bin, sleep)

    # -------------------------
    # GET Q VALUE
    # -------------------------
    def get_q(self, state, action):
        state = self.discretize_state(state)
        return self.q_table.get((state, action), 0.0)

    # -------------------------
    # CHOOSE ACTION
    # -------------------------
    def choose_action(self, state):
        state = self.discretize_state(state)

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation with random tie-breaking
        q_vals = [self.q_table.get((state, a), 0.0) for a in self.actions]

        max_q = max(q_vals)
        best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]

        return random.choice(best_actions)

    # -------------------------
    # UPDATE Q VALUE
    # -------------------------
    def update(self, state, action, reward, next_state):
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        old_q = self.q_table.get((state, action), 0.0)

        future_q = max([
            self.q_table.get((next_state, a), 0.0)
            for a in self.actions
        ])

        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

        self.q_table[(state, action)] = new_q

    # -------------------------
    # DECAY EPSILON
    # -------------------------
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
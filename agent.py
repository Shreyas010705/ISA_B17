import random


class QLearningAgent:
    def __init__(self):
        self.q_table = {}

        # Tuned for stability
        self.alpha = 0.1      # learning rate
        self.gamma = 0.9      # future reward importance
        self.epsilon = 0.1    # exploration

        self.actions = [0, 1]  # 0 = wait, 1 = transmit

    # -------------------------
    # Get Q value
    # -------------------------
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    # -------------------------
    # Choose action
    # -------------------------
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = [self.get_q(state, a) for a in self.actions]
        return self.actions[q_values.index(max(q_values))]

    # -------------------------
    # Update Q value
    # -------------------------
    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)

        future_q = max([self.get_q(next_state, a) for a in self.actions])

        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

        self.q_table[(state, action)] = new_q
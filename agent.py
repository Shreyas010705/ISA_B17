import random


class QLearningAgent:
    def __init__(self):
        self.q_table = {}

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.05   # 🔥 reduced (was 0.1)

        self.actions = [0, 1]

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_vals = [self.get_q(state, a) for a in self.actions]
        return self.actions[q_vals.index(max(q_vals))]

    def update(self, state, action, reward, next_state):
        old_q = self.get_q(state, action)
        future_q = max([self.get_q(next_state, a) for a in self.actions])

        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q_table[(state, action)] = new_q

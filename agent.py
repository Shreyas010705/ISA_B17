import random

class QLearningAgent:
    def __init__(self, actions=[0, 1], alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        actions = [0, 1]
            0 → wait
            1 → transmit
        
        alpha → learning rate (how fast it learns)
        gamma → discount factor (importance of future reward)
        epsilon → exploration rate (random actions)
        """
        
        self.q_table = {}  # stores Q-values
        self.actions = actions
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        """
        Get Q-value for (state, action)
        If not present → return 0
        """
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """
        ε-greedy policy:
        - with probability epsilon → random action (explore)
        - otherwise → best action (exploit)
        """

        if random.random() < self.epsilon:
            return random.choice(self.actions)  # explore
        else:
            # choose action with highest Q-value
            q_values = [self.get_q(state, a) for a in self.actions]
            max_q = max(q_values)
            
            # if tie → choose randomly among best
            best_actions = [a for a in self.actions if self.get_q(state, a) == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """
        Q-learning update rule:
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """

        current_q = self.get_q(state, action)

        # max Q for next state
        next_q_values = [self.get_q(next_state, a) for a in self.actions]
        max_next_q = max(next_q_values)

        # Q-learning formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        # update table
        self.q_table[(state, action)] = new_q
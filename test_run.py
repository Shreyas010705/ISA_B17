# Import everything
from environment import AoIEnvironment
from agent import QLearningAgent
from baseline import GreedyPolicy, PeriodicPolicy

# Create environment
env = AoIEnvironment()

# Create agents
rl_agent = QLearningAgent()
greedy_agent = GreedyPolicy()
periodic_agent = PeriodicPolicy(interval=2)

# -------------------------------
# TRAIN RL AGENT
# -------------------------------
episodes = 50
steps_per_episode = 50

print("Training RL Agent...\n")

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(steps_per_episode):
        action = rl_agent.choose_action(state)
        next_state, reward, done = env.step(action)

        rl_agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    print(f"Episode {ep+1}: Total Reward = {total_reward}")

print("\nTraining Complete!\n")


# -------------------------------
# TEST FUNCTION (for all policies)
# -------------------------------
def test_policy(policy, name):
    print(f"\nTesting {name}...\n")

    state = env.reset()
    total_reward = 0

    for step in range(20):
        action = policy.choose_action(state)
        next_state, reward, done = env.step(action)

        print(f"Step {step+1}: State={next_state}, Action={action}, Reward={reward}")

        total_reward += reward
        state = next_state

    print(f"\n{name} Total Reward: {total_reward}")


# -------------------------------
# TEST ALL POLICIES
# -------------------------------

# RL (use best action, no exploration)
class RLTestWrapper:
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, state):
        q_values = [self.agent.get_q(state, a) for a in [0, 1]]
        return [0, 1][q_values.index(max(q_values))]

rl_test_agent = RLTestWrapper(rl_agent)

# Run tests
test_policy(rl_test_agent, "RL Agent")
test_policy(greedy_agent, "Greedy Policy")
test_policy(periodic_agent, "Periodic Policy")

print("\nDone.")
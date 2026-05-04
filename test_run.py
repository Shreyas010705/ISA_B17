from environment import AoIEnvironment
from agent import QLearningAgent

env = AoIEnvironment()
agent = QLearningAgent()

state = env.reset()

for i in range(30):
    action = agent.choose_action(state)
    next_state, reward, _ = env.step(action)

    agent.update(state, action, reward, next_state)

    print(f"{i} | State={state} | Action={action} | Reward={reward}")

    state = next_state

from environment import AoIEnvironment
from agent import QLearningAgent

env = AoIEnvironment(
    sleep_cycle=10,
    sleep_duration=3,
    delay_steps=2
)

agent = QLearningAgent()

state = env.reset()

for i in range(30):
    action = agent.choose_action(state)
    next_state, reward, _ = env.step(action)

    agent.update(state, action, reward, next_state)

    print(
        f"{i} | State={state} | Action={action} | Reward={round(reward,2)}"
    )

    state = next_state
from environment import AoIEnvironment

env = AoIEnvironment()

state = env.reset()
print("Initial:", state)

for i in range(20):
    action = i % 2

    next_state, reward, _ = env.step(action)

    print(
        f"Step {i} | State={state} | Action={action} | "
        f"Reward={round(reward,2)} | Next={next_state}"
    )

    state = next_state

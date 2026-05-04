from environment import AoIEnvironment

# Use deterministic parameters
env = AoIEnvironment(
    energy_rate=0.5,
    battery_size=5,
    sleep_cycle=10,
    sleep_duration=3,
    delay_steps=2
)

state = env.reset()
print("Initial State:", state)

for i in range(20):
    action = i % 2  # alternate actions (0,1)

    next_state, reward, _ = env.step(action)

    print(
        f"Step {i} | "
        f"State={state} | "
        f"Action={action} | "
        f"Reward={round(reward,2)} | "
        f"Next={next_state}"
    )

    state = next_state

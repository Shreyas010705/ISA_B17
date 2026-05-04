from environment import AoIEnvironment
from agent import QLearningAgent


# -------------------------
# Initialize
# -------------------------
env = AoIEnvironment(
    energy_rate=0.5,
    battery_size=5,
    duty_cycle=3,
    outage_duration=5
)

agent = QLearningAgent()

episodes = 5
steps = 30


# -------------------------
# Training loop
# -------------------------
for ep in range(episodes):
    state = env.reset()
    print(f"\n--- Episode {ep+1} ---")

    for step in range(steps):
        action = agent.choose_action(state)
        next_state, reward, _ = env.step(action)

        agent.update(state, action, reward, next_state)

        # Debug print (VERY IMPORTANT)
        print(
            f"Step {step} | "
            f"State={state} | "
            f"Action={action} | "
            f"Reward={round(reward,2)} | "
            f"Next={next_state}"
        )

        state = next_state


# -------------------------
# Testing trained agent
# -------------------------
print("\n--- TESTING TRAINED AGENT ---")

state = env.reset()

for step in range(20):
    # Greedy action (no exploration)
    q_values = [agent.get_q(state, a) for a in [0, 1]]
    action = [0, 1][q_values.index(max(q_values))]

    next_state, reward, _ = env.step(action)

    print(
        f"Step {step} | "
        f"State={state} | "
        f"Action={action} | "
        f"Reward={round(reward,2)}"
    )

    state = next_state

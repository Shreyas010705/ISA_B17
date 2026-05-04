import csv
import numpy as np

from environment import AoIEnvironment
from agent import QLearningAgent
from baseline import GreedyPolicy, PeriodicPolicy

np.random.seed(42)

# -------------------------
# Experiment settings
# -------------------------
energy_levels = [0.2, 0.5, 0.8]
battery_sizes = [3, 5, 7]
duty_cycles = [2, 3, 5]
outage_levels = [0, 5, 10]

trials = 10
episodes = 30
steps = 50


# -------------------------
# Train RL
# -------------------------
def train_rl(env):
    agent = QLearningAgent()

    for _ in range(episodes):
        state = env.reset()

        for _ in range(steps):
            action = agent.choose_action(state)
            next_state, reward, _ = env.step(action)

            agent.update(state, action, reward, next_state)
            state = next_state

    return agent


# -------------------------
# Evaluate
# -------------------------
def evaluate(env, policy):
    state = env.reset()
    total_aoi = 0
    peak_aoi = 0
    values = []

    for _ in range(steps):
        action = policy.choose_action(state)
        state, _, _ = env.step(action)

        aoi = state[0]
        total_aoi += aoi
        peak_aoi = max(peak_aoi, aoi)
        values.append(aoi)

    return total_aoi / steps, peak_aoi, np.var(values)


# -------------------------
# RL wrapper
# -------------------------
class RLWrapper:
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, state):
        q_vals = [self.agent.get_q(state, a) for a in [0, 1]]
        return [0, 1][q_vals.index(max(q_vals))]


# -------------------------
# Run experiments
# -------------------------
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Energy",
        "Battery",
        "Duty",
        "Outage",
        "Policy",
        "Avg_AoI",
        "Peak_AoI",
        "Variance"
    ])

    print("Running experiments...\n")

    for energy in energy_levels:
        for battery in battery_sizes:
            for duty in duty_cycles:
                for outage in outage_levels:

                    print(f"E={energy}, B={battery}, D={duty}, O={outage}")

                    for _ in range(trials):

                        env = AoIEnvironment(
                            energy_rate=energy,
                            battery_size=battery,
                            duty_cycle=duty,
                            outage_duration=outage
                        )

                        # RL
                        agent = train_rl(env)
                        rl = RLWrapper(agent)
                        avg, peak, var = evaluate(env, rl)
                        writer.writerow([energy, battery, duty, outage, "RL", avg, peak, var])

                        # Greedy
                        greedy = GreedyPolicy()
                        avg, peak, var = evaluate(env, greedy)
                        writer.writerow([energy, battery, duty, outage, "Greedy", avg, peak, var])

                        # Periodic
                        periodic = PeriodicPolicy(interval=2)
                        avg, peak, var = evaluate(env, periodic)
                        writer.writerow([energy, battery, duty, outage, "Periodic", avg, peak, var])

    print("\nDone! Results saved to results.csv")

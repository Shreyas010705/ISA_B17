import csv
import numpy as np

from environment import AoIEnvironment
from agent import QLearningAgent
from baseline import GreedyPolicy, PeriodicPolicy

# Fix randomness (VERY IMPORTANT for research)
np.random.seed(42)

# -------------------------------
# Experiment settings
# -------------------------------

energy_levels = [0.2, 0.5, 0.8]        # low, medium, high
battery_sizes = [3, 5, 7]              # small, medium, large
duty_cycles = [1, 2, 3]                # strict → loose

trials = 20
episodes = 30
steps_per_episode = 50


# -------------------------------
# Helper: Train RL Agent
# -------------------------------
def train_rl(env):
    agent = QLearningAgent()

    for ep in range(episodes):
        state = env.reset()

        for step in range(steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state)
            state = next_state

    return agent


# -------------------------------
# Helper: Test a policy
# -------------------------------
def evaluate_policy(env, policy):
    state = env.reset()
    total_aoi = 0
    peak_aoi = 0
    aoi_values = []

    for step in range(50):
        action = policy.choose_action(state)
        next_state, reward, done = env.step(action)

        aoi = next_state[0]  # AoI is first element of state
        total_aoi += aoi
        peak_aoi = max(peak_aoi, aoi)
        aoi_values.append(aoi)

        state = next_state

    avg_aoi = total_aoi / 50
    variance = np.var(aoi_values)

    return avg_aoi, peak_aoi, variance


# -------------------------------
# RL wrapper (for testing only)
# -------------------------------
class RLTestWrapper:
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, state):
        q_values = [self.agent.get_q(state, a) for a in [0, 1]]
        return [0, 1][q_values.index(max(q_values))]


# -------------------------------
# Run experiments
# -------------------------------

with open("results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    # CSV header
    writer.writerow([
        "energy", "battery", "duty",
        "policy", "avg_aoi", "peak_aoi", "variance"
    ])

    print("Starting experiments...\n")

    for energy in energy_levels:
        for battery in battery_sizes:
            for duty in duty_cycles:

                print(f"Running condition: Energy={energy}, Battery={battery}, Duty={duty}")

                for t in range(trials):

                    # Create environment
                    env = AoIEnvironment(
                        energy_rate=energy,
                        battery_size=battery,
                        duty_cycle=duty
                    )

                    # ---------------- RL ----------------
                    rl_agent = train_rl(env)
                    rl_policy = RLTestWrapper(rl_agent)

                    avg, peak, var = evaluate_policy(env, rl_policy)
                    writer.writerow([energy, battery, duty, "RL", avg, peak, var])

                    # ---------------- Greedy ----------------
                    greedy = GreedyPolicy()
                    avg, peak, var = evaluate_policy(env, greedy)
                    writer.writerow([energy, battery, duty, "Greedy", avg, peak, var])

                    # ---------------- Periodic ----------------
                    periodic = PeriodicPolicy(interval=2)
                    avg, peak, var = evaluate_policy(env, periodic)
                    writer.writerow([energy, battery, duty, "Periodic", avg, peak, var])

    print("\nExperiments complete! Results saved to results.csv")
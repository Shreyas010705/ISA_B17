import csv
import numpy as np

from environment import AoIEnvironment
from agent import QLearningAgent
from baseline import GreedyPolicy, PeriodicPolicy

np.random.seed(42)

# PARAMETERS
energy_levels = [0.2, 0.5, 0.8]
battery_sizes = [3, 5, 7]
outage_levels = [0, 5, 10]

sleep_cycles = [10]
sleep_durations = [2, 4]

delay_levels = [1, 2, 3]

trials = 10
episodes = 30
steps = 50


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


def evaluate(env, policy):
    state = env.reset()
    values = []
    peak = 0

    for _ in range(steps):
        action = policy.choose_action(state)
        state, _, _ = env.step(action)

        aoi = state[0]
        values.append(aoi)
        peak = max(peak, aoi)

    return np.mean(values), peak, np.var(values)


class RLWrapper:
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, state):
        q_vals = [self.agent.get_q(state, a) for a in [0, 1]]
        return [0, 1][q_vals.index(max(q_vals))]


with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Energy", "Battery", "Outage",
        "SleepCycle", "SleepDuration",
        "Delay", "Policy", "Avg_AoI", "Peak_AoI", "Variance"
    ])

    print("Running experiments...\n")

    for energy in energy_levels:
        for battery in battery_sizes:
            for outage in outage_levels:
                for cycle in sleep_cycles:
                    for duration in sleep_durations:
                        for delay in delay_levels:

                            print(f"E={energy}, B={battery}, O={outage}, C={cycle}, D={duration}, Delay={delay}")

                            for _ in range(trials):

                                env = AoIEnvironment(
                                    energy_rate=energy,
                                    battery_size=battery,
                                    outage_duration=outage,
                                    sleep_cycle=cycle,
                                    sleep_duration=duration,
                                    delay_steps=delay
                                )

                                # RL
                                agent = train_rl(env)
                                rl = RLWrapper(agent)
                                writer.writerow([energy, battery, outage, cycle, duration, delay, "RL", *evaluate(env, rl)])

                                # Greedy
                                writer.writerow([energy, battery, outage, cycle, duration, delay, "Greedy", *evaluate(env, GreedyPolicy())])

                                # Periodic
                                writer.writerow([energy, battery, outage, cycle, duration, delay, "Periodic", *evaluate(env, PeriodicPolicy())])

    print("\nDone! results.csv generated.")

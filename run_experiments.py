import csv
import numpy as np

from environment import AoIEnvironment
from agent import QLearningAgent
from baseline import GreedyPolicy, PeriodicPolicy

np.random.seed(42)

energy_levels = [0.2, 0.5, 0.8]
battery_sizes = [3, 5, 7]

sleep_cycles = [10]
sleep_durations = [2, 4]

outage_levels = [0, 5, 10]

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
        "Energy", "Battery", "SleepDur", "Outage",
        "Policy", "Avg_AoI", "Peak_AoI", "Variance"
    ])

    for energy in energy_levels:
        for battery in battery_sizes:
            for sleep_dur in sleep_durations:
                for outage in outage_levels:

                    for _ in range(trials):

                        env = AoIEnvironment(
                            energy_rate=energy,
                            battery_size=battery,
                            sleep_cycle=10,
                            sleep_duration=sleep_dur,
                            outage_duration=outage
                        )

                        agent = train_rl(env)
                        rl = RLWrapper(agent)
                        writer.writerow([energy, battery, sleep_dur, outage, "RL", *evaluate(env, rl)])

                        writer.writerow([energy, battery, sleep_dur, outage, "Greedy", *evaluate(env, GreedyPolicy())])

                        writer.writerow([energy, battery, sleep_dur, outage, "Periodic", *evaluate(env, PeriodicPolicy())])

print("Done! results.csv generated.")

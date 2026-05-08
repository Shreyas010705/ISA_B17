import csv
import numpy as np

from environment import AoIEnvironment
from agent import QLearningAgent
<<<<<<< HEAD
from baseline import GreedyPolicy, PeriodicPolicy, ThresholdPolicy
=======
from baseline import GreedyPolicy, PeriodicPolicy
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d

# -------------------------
# PARAMETERS
# -------------------------
energy_levels = [0.2, 0.5, 0.8]
battery_sizes = [3, 5, 7]
outage_levels = [0, 5, 10]

sleep_cycles = [10]
sleep_durations = [0, 2, 4]

delay_levels = [1, 2, 3]

<<<<<<< HEAD
trials = 30            # 🔥 increased
episodes = 50
steps = 200

=======
trials = 20            # 🔥 increased
episodes = 50
steps = 200

periodic_intervals = [3, 5, 7]
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d


# -------------------------
# TRAIN RL
# -------------------------
def train_rl(env):
    agent = QLearningAgent()

    episode_rewards = []  # <-- added

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0  # <-- added

        for _ in range(steps):
            action = agent.choose_action(state)
            next_state, reward, _ = env.step(action)

            agent.update(state, action, reward, next_state)
            state = next_state

<<<<<<< HEAD
            total_reward += reward  # <-- added

=======
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d
        agent.decay_epsilon()

        episode_rewards.append(total_reward)  # <-- added

    # Optional: quick sanity print (last 5 episodes)
    print("Last 5 episode rewards:", episode_rewards[-5:])

    return agent


# -------------------------
# EVALUATE
# -------------------------
def evaluate(env, policy):
    state = env.reset()

    # 🔥 CRITICAL: reset policy state every trial
    if hasattr(policy, "reset"):
        policy.reset()

    values = []
    peak = 0

    for _ in range(steps):
        action = policy.choose_action(state)
        state, _, _ = env.step(action)

        # Use TRUE AoI (not delayed state)
        aoi = env.aoi

        values.append(aoi)
        peak = max(peak, aoi)

    p95 = np.percentile(values, 95)

<<<<<<< HEAD
    tau = 20  # fixed spike threshold (DO NOT change later)
    spike_freq = np.mean([v > tau for v in values])

    return np.mean(values), peak, np.var(values), p95, spike_freq

=======
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d
# -------------------------
# RL WRAPPER
# -------------------------
class RLWrapper:
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, state):
        q_vals = [self.agent.get_q(state, a) for a in [0, 1]]

        max_q = max(q_vals)
        best_actions = [a for a, q in zip([0, 1], q_vals) if q == max_q]

        return best_actions[0]


# -------------------------
# RUN EXPERIMENTS
# -------------------------
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Trial_ID",
        "Energy", "Battery", "Outage",
        "SleepCycle", "SleepDuration",
        "Delay",
        "Policy", "Avg_AoI", "Peak_AoI", "Variance", "P95_AoI", "Spike_Freq"
    ])

    print("Running experiments...\n")

    trial_counter = 0

    for energy in energy_levels:
        for battery in battery_sizes:
            for outage in outage_levels:
                for cycle in sleep_cycles:
                    for duration in sleep_durations:
                        for delay in delay_levels:

                            print(f"E={energy}, B={battery}, O={outage}, C={cycle}, D={duration}, Delay={delay}")

                            # -------------------------
                            # TRAIN RL ONCE
                            # -------------------------
<<<<<<< HEAD
                            np.random.seed(42)
=======
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d
                            train_env = AoIEnvironment(
                                energy_rate=energy,
                                battery_size=battery,
                                outage_duration=outage,
                                sleep_cycle=cycle,
                                sleep_duration=duration,
                                delay_steps=delay
                            )

                            agent = train_rl(train_env)
                            rl = RLWrapper(agent)

                            # -------------------------
                            # MULTIPLE TRIALS
                            # -------------------------
                            for t in range(trials):

                                trial_counter += 1

                                # independent randomness
<<<<<<< HEAD
                                np.random.seed(1000 + t)
=======
                                np.random.seed(None)
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d

                                # -------------------------
                                # RL
                                # -------------------------
                                eval_env = AoIEnvironment(
                                    energy_rate=energy,
                                    battery_size=battery,
                                    outage_duration=outage,
                                    sleep_cycle=cycle,
                                    sleep_duration=duration,
                                    delay_steps=delay
                                )

<<<<<<< HEAD
                                avg, peak, var, p95, spike = evaluate(eval_env, rl)
=======
                                avg, peak, var = evaluate(eval_env, rl)
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d

                                writer.writerow([
                                    trial_counter,
                                    energy, battery, outage,
                                    cycle, duration, delay,
                                    "RL", avg, peak, var, p95, spike
                                ])

                                # -------------------------
                                # GREEDY
                                # -------------------------
                                eval_env = AoIEnvironment(
                                    energy_rate=energy,
                                    battery_size=battery,
                                    outage_duration=outage,
                                    sleep_cycle=cycle,
                                    sleep_duration=duration,
                                    delay_steps=delay
                                )

<<<<<<< HEAD
                                avg, peak, var, p95, spike = evaluate(eval_env, GreedyPolicy())
=======
                                avg, peak, var = evaluate(eval_env, GreedyPolicy())
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d

                                writer.writerow([
                                    trial_counter,
                                    energy, battery, outage,
                                    cycle, duration, delay,
                                    "Greedy", avg, peak, var, p95, spike
                                ])

                                # -------------------------                             
                                # PERIODIC (FIXED INTERVAL)
                                # -------------------------
                                eval_env = AoIEnvironment(
                                    energy_rate=energy,
                                    battery_size=battery,
                                    outage_duration=outage,
                                    sleep_cycle=cycle,
                                    sleep_duration=duration,
                                    delay_steps=delay
                                )

                                avg, peak, var, p95, spike = evaluate(eval_env, PeriodicPolicy(interval=5))

                                writer.writerow([
                                    trial_counter,
                                    energy, battery, outage,
                                    cycle, duration, delay,
                                    "Periodic", avg, peak, var, p95, spike
                                ])

                                # -------------------------
<<<<<<< HEAD
                                # THRESHOLD POLICY
                                # -------------------------
                                eval_env = AoIEnvironment(
                                    energy_rate=energy,
                                    battery_size=battery,
                                    outage_duration=outage,
                                    sleep_cycle=cycle,
                                    sleep_duration=duration,
                                    delay_steps=delay
                                )

                                avg, peak, var, p95, spike = evaluate(eval_env, ThresholdPolicy())
=======
                                # PERIODIC (BEST INTERVAL)
                                # -------------------------
                                best_avg = float("inf")
                                best_peak = None
                                best_var = None

                                for interval in periodic_intervals:
                                    eval_env = AoIEnvironment(
                                        energy_rate=energy,
                                        battery_size=battery,
                                        outage_duration=outage,
                                        sleep_cycle=cycle,
                                        sleep_duration=duration,
                                        delay_steps=delay
                                    )

                                    avg, peak, var = evaluate(eval_env, PeriodicPolicy(interval=interval))

                                    if avg < best_avg:
                                        best_avg = avg
                                        best_peak = peak
                                        best_var = var
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d

                                writer.writerow([
                                    trial_counter,
                                    energy, battery, outage,
                                    cycle, duration, delay,
<<<<<<< HEAD
                                    "Threshold", avg, peak, var, p95, spike
=======
                                    "Periodic", best_avg, best_peak, best_var
>>>>>>> 848fd0fc16cbda277b73f996508fff017907ea5d
                                ])

    print("\nDone! results.csv generated.")
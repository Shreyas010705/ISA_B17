import csv
import random
import numpy as np

from environment import AoIEnvironment
from agent import QLearningAgent
from baseline import GreedyPolicy, PeriodicPolicy, ThresholdPolicy

# -------------------------
# PARAMETERS
# -------------------------
energy_levels = [0.2, 0.5, 0.8]
battery_sizes = [3, 5, 7]
outage_levels = [0, 5, 10]

sleep_cycles = [10]
sleep_durations = [0, 2, 4]

delay_levels = [1, 2, 3]

trials = 30            # 🔥 increased
episodes = 50
steps = 200

# -------------------------
# MASTER SEEDS
# -------------------------
MASTER_SEEDS = list(range(1, 21))


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

            total_reward += reward  # <-- added

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

    actions = []

    # --------------------------------------------------
    # RECOVERY FAILURE TRACKING
    # --------------------------------------------------
    burst_lengths = []
    current_burst = 0

    for _ in range(steps):
        action = policy.choose_action(state)
        actions.append(action)
        state, _, _ = env.step(action)

        # Use TRUE AoI (not delayed state)
        aoi = env.aoi

        values.append(aoi)
        peak = max(peak, aoi)
        # --------------------------------------------------
        # TRACK HIGH-AOI BURSTS
        # --------------------------------------------------
        if aoi > 20:

            current_burst += 1

        else:

            if current_burst > 0:
                burst_lengths.append(current_burst)

            current_burst = 0

    p95 = np.percentile(values, 95)

    tau = 20  # fixed spike threshold (DO NOT change later)
    spike_freq = np.mean([v > tau for v in values])

    # --------------------------------------------------
    # ACTION SWITCHING RATE
    # --------------------------------------------------
    switches = 0

    for i in range(1, len(actions)):
        if actions[i] != actions[i - 1]:
            switches += 1

    switch_rate = switches / (len(actions) - 1)

    # --------------------------------------------------
    # HANDLE BURST REACHING FINAL TIMESTEP
    # --------------------------------------------------
    if current_burst > 0:
        burst_lengths.append(current_burst)

    # --------------------------------------------------
    # RECOVERY DURATION METRIC
    # --------------------------------------------------
    if len(burst_lengths) > 0:
        recovery_duration = np.mean(burst_lengths)
    else:
        recovery_duration = 0

    return (
        np.mean(values),
        peak,
        np.var(values),
        p95,
        spike_freq,
        switch_rate,
        recovery_duration
    )

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
if __name__ == "__main__":
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "MasterSeed",
            "Trial_ID",
            "Energy", "Battery", "Outage",
            "SleepCycle", "SleepDuration",
            "Delay",
            "Policy", "Avg_AoI", "Peak_AoI", "Variance", "P95_AoI", "Spike_Freq",
"Switch_Rate", "Recovery_Duration"
        ])

        print("Running experiments...\n")

        trial_counter = 0

        for master_seed in MASTER_SEEDS:

            print(f"\n==============================")
            print(f"MASTER SEED = {master_seed}")
            print(f"==============================\n")

            random.seed(master_seed)
            np.random.seed(master_seed)
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
                                    random.seed(master_seed)
                                    np.random.seed(master_seed)
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
                                        
                                        # -------------------------
                                        # RL
                                        # -------------------------
                                        random.seed(master_seed * 1000 + t)
                                        np.random.seed(master_seed * 1000 + t)
                                        eval_env = AoIEnvironment(
                                            energy_rate=energy,
                                            battery_size=battery,
                                            outage_duration=outage,
                                            sleep_cycle=cycle,
                                            sleep_duration=duration,
                                            delay_steps=delay
                                        )

                                        avg, peak, var, p95, spike, switch, recovery = evaluate(eval_env, rl)

                                        writer.writerow([
                                            master_seed,
                                            trial_counter,
                                            energy, battery, outage,
                                            cycle, duration, delay,
                                            "RL", avg, peak, var, p95, spike, switch, recovery
                                        ])

                                        # -------------------------
                                        # GREEDY
                                        # -------------------------
                                        random.seed(master_seed * 1000 + t)
                                        np.random.seed(master_seed * 1000 + t)
                                        eval_env = AoIEnvironment(
                                            energy_rate=energy,
                                            battery_size=battery,
                                            outage_duration=outage,
                                            sleep_cycle=cycle,
                                            sleep_duration=duration,
                                            delay_steps=delay
                                        )

                                        avg, peak, var, p95, spike, switch, recovery = evaluate(eval_env, GreedyPolicy())

                                        writer.writerow([
                                            master_seed,
                                            trial_counter,
                                            energy, battery, outage,
                                            cycle, duration, delay,
                                            "Greedy", avg, peak, var, p95, spike, switch, recovery
                                        ])

                                        # -------------------------                             
                                        # PERIODIC (FIXED INTERVAL)
                                        # -------------------------
                                        random.seed(master_seed * 1000 + t)
                                        np.random.seed(master_seed * 1000 + t)
                                        eval_env = AoIEnvironment(
                                            energy_rate=energy,
                                            battery_size=battery,
                                            outage_duration=outage,
                                            sleep_cycle=cycle,
                                            sleep_duration=duration,
                                            delay_steps=delay
                                        )

                                        avg, peak, var, p95, spike, switch, recovery = evaluate(eval_env, PeriodicPolicy(interval=5))

                                        writer.writerow([
                                            master_seed,
                                            trial_counter,
                                            energy, battery, outage,
                                            cycle, duration, delay,
                                            "Periodic", avg, peak, var, p95, spike, switch, recovery
                                        ])

                                        # -------------------------
                                        # THRESHOLD POLICY
                                        # -------------------------
                                        random.seed(master_seed * 1000 + t)
                                        np.random.seed(master_seed * 1000 + t)
                                        eval_env = AoIEnvironment(
                                            energy_rate=energy,
                                            battery_size=battery,
                                            outage_duration=outage,
                                            sleep_cycle=cycle,
                                            sleep_duration=duration,
                                            delay_steps=delay
                                        )

                                        avg, peak, var, p95, spike, switch, recovery = evaluate(eval_env, ThresholdPolicy())

                                        writer.writerow([
                                            master_seed,
                                            trial_counter,
                                            energy, battery, outage,
                                            cycle, duration, delay,
                                            "Threshold", avg, peak, var, p95, spike, switch, recovery
                                        ])

        print("\nDone! results.csv generated.")
import random
import numpy as np
import pandas as pd

from environment import AoIEnvironment

from agent import QLearningAgent as DefaultAgent
from agent_coarse import QLearningAgent as CoarseAgent
from agent_fine import QLearningAgent as FineAgent


# ==================================================
# PARAMETERS
# ==================================================
MASTER_SEEDS = list(range(1, 21))

episodes = 50
steps = 200
trials = 10

# Representative harsh regime
energy = 0.2
battery = 5
outage = 10

sleep_cycle = 10
sleep_duration = 4

delay = 1


# ==================================================
# TRAIN RL
# ==================================================
def train_rl(env, agent_class):

    agent = agent_class()

    for _ in range(episodes):

        state = env.reset()

        for _ in range(steps):

            action = agent.choose_action(state)

            next_state, reward, _ = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state

        agent.decay_epsilon()

    return agent


# ==================================================
# EVALUATE
# ==================================================
def evaluate(env, agent):

    state = env.reset()

    values = []

    burst_lengths = []
    current_burst = 0

    for _ in range(steps):

        q_vals = [agent.get_q(state, a) for a in [0, 1]]

        action = int(np.argmax(q_vals))

        state, _, _ = env.step(action)

        aoi = env.aoi

        values.append(aoi)

        # Recovery burst tracking
        if aoi > 20:
            current_burst += 1
        else:
            if current_burst > 0:
                burst_lengths.append(current_burst)

            current_burst = 0

    if current_burst > 0:
        burst_lengths.append(current_burst)

    if len(burst_lengths) > 0:
        recovery_duration = np.mean(burst_lengths)
    else:
        recovery_duration = 0

    return {
        "Variance": np.var(values),
        "P95": np.percentile(values, 95),
        "Recovery": recovery_duration
    }


# ==================================================
# RUN SENSITIVITY TEST
# ==================================================
results = []

agent_configs = [
    ("Coarse", CoarseAgent),
    ("Default", DefaultAgent),
    ("Fine", FineAgent)
]

for label, agent_class in agent_configs:

    print(f"\n======================")
    print(f"{label} discretization")
    print(f"======================")

    variances = []
    p95s = []
    recoveries = []

    for seed in MASTER_SEEDS:

        random.seed(seed)
        np.random.seed(seed)

        train_env = AoIEnvironment(
            energy_rate=energy,
            battery_size=battery,
            outage_duration=outage,
            sleep_cycle=sleep_cycle,
            sleep_duration=sleep_duration,
            delay_steps=delay
        )

        agent = train_rl(train_env, agent_class)

        for t in range(trials):

            random.seed(seed * 1000 + t)
            np.random.seed(seed * 1000 + t)

            eval_env = AoIEnvironment(
                energy_rate=energy,
                battery_size=battery,
                outage_duration=outage,
                sleep_cycle=sleep_cycle,
                sleep_duration=sleep_duration,
                delay_steps=delay
            )

            metrics = evaluate(eval_env, agent)

            variances.append(metrics["Variance"])
            p95s.append(metrics["P95"])
            recoveries.append(metrics["Recovery"])

    results.append({
        "Discretization": label,
        "Variance": np.mean(variances),
        "P95": np.mean(p95s),
        "Recovery_Duration": np.mean(recoveries)
    })

# ==================================================
# RESULTS TABLE
# ==================================================
df = pd.DataFrame(results)

print("\n===================================")
print("DISCRETIZATION SENSITIVITY RESULTS")
print("===================================\n")

print(df.round(2).to_string(index=False))
import numpy as np
import matplotlib.pyplot as plt

from environment import AoIEnvironment
from baseline import ThresholdPolicy
from run_experiments import train_rl

# --------------------------------------------------
# REPRESENTATIVE HARSH CONFIGURATION
# --------------------------------------------------
energy = 0.2
battery = 5
outage_cycle = 20
outage = 10

sleep_cycle = 10
sleep_duration = 4

delay = 3
steps = 200

# --------------------------------------------------
# TRAIN RL AGENT
# --------------------------------------------------
np.random.seed(42)

train_env = AoIEnvironment(
    energy_rate=energy,
    battery_size=battery,
    outage_cycle=outage_cycle,
    outage_duration=outage,
    sleep_cycle=sleep_cycle,
    sleep_duration=sleep_duration,
    delay_steps=delay
)

agent = train_rl(train_env)

# --------------------------------------------------
# RL EVALUATION
# --------------------------------------------------
np.random.seed(1000)

rl_env = AoIEnvironment(
    energy_rate=energy,
    battery_size=battery,
    outage_cycle=outage_cycle,
    outage_duration=outage,
    sleep_cycle=sleep_cycle,
    sleep_duration=sleep_duration,
    delay_steps=delay
)

state = rl_env.reset()

rl_aoi = []

for _ in range(steps):

    # Greedy action from learned Q-table
    q_vals = [agent.get_q(state, a) for a in [0, 1]]
    action = int(np.argmax(q_vals))

    state, _, _ = rl_env.step(action)

    rl_aoi.append(rl_env.aoi)

# --------------------------------------------------
# THRESHOLD POLICY EVALUATION
# --------------------------------------------------
np.random.seed(1000)

th_env = AoIEnvironment(
    energy_rate=energy,
    battery_size=battery,
    outage_cycle=outage_cycle,
    outage_duration=outage,
    sleep_cycle=sleep_cycle,
    sleep_duration=sleep_duration,
    delay_steps=delay
)

threshold = ThresholdPolicy()

state = th_env.reset()

threshold_aoi = []

for _ in range(steps):

    action = threshold.choose_action(state)

    state, _, _ = th_env.step(action)

    threshold_aoi.append(th_env.aoi)

# --------------------------------------------------
# PLOT
# --------------------------------------------------
plt.figure(figsize=(12, 5))

plt.plot(
    rl_aoi,
    label='RL',
    linewidth=2
)

plt.plot(
    threshold_aoi,
    label='Threshold',
    linewidth=2
)

# --------------------------------------------------
# HIGHLIGHT OUTAGE REGIONS
# --------------------------------------------------
for start in range(0, steps, outage_cycle):

    plt.axvspan(
        start,
        start + outage,
        alpha=0.12,
        color='gray'
    )

# --------------------------------------------------
# LABELS
# --------------------------------------------------
plt.xlabel('Timestep', fontsize=12)
plt.ylabel('AoI', fontsize=12)

plt.title(
    'AoI Evolution Under Combined Constraints '
    '(RL vs Threshold)',
    fontsize=14
)

plt.legend(fontsize=11)

plt.grid(True, alpha=0.3)

# --------------------------------------------------
# CLEAN LAYOUT
# --------------------------------------------------
plt.tight_layout()

plt.savefig(
    'plot_failure_story.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print('Failure story plot generated successfully!')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("results.csv")
NUM_SEEDS = 20

# --------------------------------------------------
# STAGE 1:
# Average trials within each master seed
# --------------------------------------------------
seed_grouped = df.groupby(
    [
        "MasterSeed",
        "Energy",
        "Battery",
        "Outage",
        "SleepCycle",
        "SleepDuration",
        "Delay",
        "Policy"
    ]
).mean(numeric_only=True).reset_index()


# --------------------------------------------------
# STAGE 2:
# Aggregate across master seeds
# --------------------------------------------------
grouped = seed_grouped.groupby(
    [
        "Energy",
        "Battery",
        "Outage",
        "SleepCycle",
        "SleepDuration",
        "Delay",
        "Policy"
    ]
).agg({
    "Avg_AoI": ["mean", "std"],
    "Peak_AoI": ["mean", "std"],
    "Variance": ["mean", "std"],
    "P95_AoI": ["mean", "std"],
    "Spike_Freq": ["mean", "std"],
    "Switch_Rate": ["mean", "std"],
    "Recovery_Duration": ["mean", "std"]
}).reset_index()

# --------------------------------------------------
# CLEAN COLUMN NAMES
# --------------------------------------------------
grouped.columns = [
    '_'.join(col).strip('_')
    if isinstance(col, tuple)
    else col
    for col in grouped.columns
]

# --------------------------------------------------
# COMPUTE 95% CONFIDENCE INTERVALS
# --------------------------------------------------
grouped["Avg_AoI_ci95"] = (
    1.96 * grouped["Avg_AoI_std"] / np.sqrt(NUM_SEEDS)
)

grouped["Peak_AoI_ci95"] = (
    1.96 * grouped["Peak_AoI_std"] / np.sqrt(NUM_SEEDS)
)

grouped["Variance_ci95"] = (
    1.96 * grouped["Variance_std"] / np.sqrt(NUM_SEEDS)
)

grouped["P95_AoI_ci95"] = (
    1.96 * grouped["P95_AoI_std"] / np.sqrt(NUM_SEEDS)
)

grouped["Spike_Freq_ci95"] = (
    1.96 * grouped["Spike_Freq_std"] / np.sqrt(NUM_SEEDS)
)

grouped["Switch_Rate_ci95"] = (
    1.96 * grouped["Switch_Rate_std"] / np.sqrt(NUM_SEEDS)
)

grouped["Recovery_Duration_ci95"] = (
    1.96 * grouped["Recovery_Duration_std"] / np.sqrt(NUM_SEEDS)
)

# ==================================================
# 1. AVG AOI VS ENERGY (ERROR BARS)
# ==================================================
for policy in grouped["Policy"].unique():

    subset = grouped[grouped["Policy"] == policy]

    plt.figure(figsize=(7, 5))

    for delay in sorted(subset["Delay"].unique()):

        data = subset[
            subset["Delay"] == delay
        ].sort_values("Energy")

        plt.errorbar(
            data["Energy"],
            data["Avg_AoI_mean"],
            yerr=data["Avg_AoI_ci95"],
            marker='o',
            linewidth=2,
            capsize=4,
            label=f"Delay {delay}"
        )

    plt.title(
        f"Average AoI vs Energy ({policy})",
        fontsize=13
    )

    plt.xlabel("Energy Harvesting Rate", fontsize=11)
    plt.ylabel("Average AoI", fontsize=11)

    plt.legend(fontsize=9)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"plot_energy_{policy}.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

# ==================================================
# 2. CLEANER SLEEP PLOT
# ==================================================
#
# Simplified for readability:
# - Fix Battery = 5
# - Fix Outage = 5
# - Compare only RL vs Threshold
#
sleep_subset = grouped[
    (grouped["Battery"] == 5) &
    (grouped["Outage"] == 5) &
    (grouped["Policy"].isin(["RL", "Threshold"]))
]

plt.figure(figsize=(7, 5))

for policy in ["RL", "Threshold"]:

    data = sleep_subset[
        (sleep_subset["Policy"] == policy) &
        (sleep_subset["Delay"] == 2)
    ].sort_values("SleepDuration")

    plt.errorbar(
        data["SleepDuration"],
        data["Avg_AoI_mean"],
        yerr=data["Avg_AoI_ci95"],
        marker='o',
        linewidth=2,
        capsize=4,
        label=policy
    )

plt.title(
    "Average AoI vs Sleep Duration",
    fontsize=13
)

plt.xlabel("Sleep Duration", fontsize=11)
plt.ylabel("Average AoI", fontsize=11)

plt.legend(fontsize=10)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plot_sleep_clean.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==================================================
# 3. AVG AOI VS OUTAGE (ERROR BARS)
# ==================================================
for policy in grouped["Policy"].unique():

    subset = grouped[grouped["Policy"] == policy]

    plt.figure(figsize=(7, 5))

    for delay in sorted(subset["Delay"].unique()):

        data = subset[
            subset["Delay"] == delay
        ].sort_values("Outage")

        plt.errorbar(
            data["Outage"],
            data["Avg_AoI_mean"],
            yerr=data["Avg_AoI_ci95"],
            marker='o',
            linewidth=2,
            capsize=4,
            label=f"Delay {delay}"
        )

    plt.title(
        f"Average AoI vs Outage ({policy})",
        fontsize=13
    )

    plt.xlabel("Outage Duration", fontsize=11)
    plt.ylabel("Average AoI", fontsize=11)

    plt.legend(fontsize=9)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        f"plot_outage_{policy}.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

# ==================================================
# 4. PEAK AOI COMPARISON
# ==================================================
pivot_peak = grouped.pivot_table(
    index="Delay",
    columns="Policy",
    values="Peak_AoI_mean"
)

ax = pivot_peak.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title(
    "Peak AoI vs Delay",
    fontsize=13
)

plt.xlabel("Observation Delay", fontsize=11)
plt.ylabel("Peak AoI", fontsize=11)

plt.xticks(rotation=0)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plot_peak.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==================================================
# 5. VARIANCE COMPARISON
# ==================================================
pivot_var = grouped.pivot_table(
    index="Delay",
    columns="Policy",
    values="Variance_mean"
)

ax = pivot_var.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title(
    "Variance vs Delay",
    fontsize=13
)

plt.xlabel("Observation Delay", fontsize=11)
plt.ylabel("Variance", fontsize=11)

plt.xticks(rotation=0)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plot_variance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==================================================
# 6. P95 AOI COMPARISON
# ==================================================
pivot_p95 = grouped.pivot_table(
    index="Delay",
    columns="Policy",
    values="P95_AoI_mean"
)

ax = pivot_p95.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title(
    "P95 AoI vs Delay",
    fontsize=13
)

plt.xlabel("Observation Delay", fontsize=11)
plt.ylabel("P95 AoI", fontsize=11)

plt.xticks(rotation=0)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plot_p95.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==================================================
# 7. SPIKE FREQUENCY COMPARISON
# ==================================================
pivot_spike = grouped.pivot_table(
    index="Delay",
    columns="Policy",
    values="Spike_Freq_mean"
)

ax = pivot_spike.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title(
    "Spike Frequency vs Delay",
    fontsize=13
)

plt.xlabel("Observation Delay", fontsize=11)
plt.ylabel("Spike Frequency", fontsize=11)

plt.xticks(rotation=0)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plot_spike.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==================================================
# 8. ACTION SWITCH RATE COMPARISON
# ==================================================
pivot_switch = grouped.pivot_table(
    index="Delay",
    columns="Policy",
    values="Switch_Rate_mean"
)

ax = pivot_switch.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title(
    "Action Switch Rate vs Delay",
    fontsize=13
)

plt.xlabel("Observation Delay", fontsize=11)
plt.ylabel("Switch Rate", fontsize=11)

plt.xticks(rotation=0)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plot_switch_rate.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==================================================
# 9. RECOVERY DURATION COMPARISON
# ==================================================
pivot_recovery = grouped.pivot_table(
    index="Delay",
    columns="Policy",
    values="Recovery_Duration_mean"
)

ax = pivot_recovery.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title(
    "Recovery Duration vs Delay",
    fontsize=13
)

plt.xlabel("Observation Delay", fontsize=11)
plt.ylabel("Average Recovery Duration", fontsize=11)

plt.xticks(rotation=0)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "plot_recovery_duration.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("All plots generated successfully!")
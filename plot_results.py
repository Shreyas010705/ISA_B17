import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("results.csv")

# -------------------------
# GROUP DATA (mean + std over trials)
# -------------------------
grouped = df.groupby(
    ["Energy", "Battery", "Outage", "SleepDuration", "Delay", "Policy"]
).agg({
    "Avg_AoI": ["mean", "std"],
    "Peak_AoI": "mean",
    "Variance": "mean",
    "P95_AoI": "mean",
    "Spike_Freq": "mean"
}).reset_index()

# -------------------------
# FLATTEN COLUMN NAMES
# -------------------------
grouped.columns = [
    "Energy",
    "Battery",
    "Outage",
    "SleepDuration",
    "Delay",
    "Policy",
    "Mean_AoI",
    "Std_AoI",
    "Peak_AoI",
    "Variance",
    "P95_AoI",
    "Spike_Freq"
]

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
            data["Mean_AoI"],
            yerr=data["Std_AoI"],
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
        data["Mean_AoI"],
        yerr=data["Std_AoI"],
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
            data["Mean_AoI"],
            yerr=data["Std_AoI"],
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
    values="Peak_AoI"
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
    values="Variance"
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
    values="P95_AoI"
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
    values="Spike_Freq"
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

print("All plots generated successfully!")
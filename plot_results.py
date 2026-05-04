import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("results.csv")

# -------------------------
# GROUP DATA (average over trials)
# -------------------------
grouped = df.groupby(
    ["Energy", "Battery", "Outage", "SleepDuration", "Delay", "Policy"]
).mean().reset_index()

# -------------------------
# 1. Avg AoI vs Energy
# -------------------------
for policy in grouped["Policy"].unique():
    subset = grouped[grouped["Policy"] == policy]

    plt.figure()
    for delay in subset["Delay"].unique():
        data = subset[subset["Delay"] == delay]
        plt.plot(data["Energy"], data["Avg_AoI"], marker='o', label=f"Delay {delay}")

    plt.title(f"Avg AoI vs Energy ({policy})")
    plt.xlabel("Energy Rate")
    plt.ylabel("Average AoI")
    plt.legend()
    plt.grid()
    plt.savefig(f"plot_energy_{policy}.png")
    plt.close()


# -------------------------
# 2. Avg AoI vs Sleep Duration
# -------------------------
for policy in grouped["Policy"].unique():
    subset = grouped[grouped["Policy"] == policy]

    plt.figure()
    for delay in subset["Delay"].unique():
        data = subset[subset["Delay"] == delay]
        plt.plot(data["SleepDuration"], data["Avg_AoI"], marker='o', label=f"Delay {delay}")

    plt.title(f"Avg AoI vs Sleep Duration ({policy})")
    plt.xlabel("Sleep Duration")
    plt.ylabel("Average AoI")
    plt.legend()
    plt.grid()
    plt.savefig(f"plot_sleep_{policy}.png")
    plt.close()


# -------------------------
# 3. Avg AoI vs Outage
# -------------------------
for policy in grouped["Policy"].unique():
    subset = grouped[grouped["Policy"] == policy]

    plt.figure()
    for delay in subset["Delay"].unique():
        data = subset[subset["Delay"] == delay]
        plt.plot(data["Outage"], data["Avg_AoI"], marker='o', label=f"Delay {delay}")

    plt.title(f"Avg AoI vs Outage ({policy})")
    plt.xlabel("Outage Duration")
    plt.ylabel("Average AoI")
    plt.legend()
    plt.grid()
    plt.savefig(f"plot_outage_{policy}.png")
    plt.close()


# -------------------------
# 4. Peak AoI comparison
# -------------------------
pivot_peak = grouped.pivot_table(
    index="Delay",
    columns="Policy",
    values="Peak_AoI"
)

pivot_peak.plot(kind="bar")
plt.title("Peak AoI vs Delay")
plt.ylabel("Peak AoI")
plt.grid()
plt.savefig("plot_peak.png")
plt.close()


# -------------------------
# 5. Variance comparison
# -------------------------
pivot_var = grouped.pivot_table(
    index="Delay",
    columns="Policy",
    values="Variance"
)

pivot_var.plot(kind="bar")
plt.title("Variance vs Delay")
plt.ylabel("Variance")
plt.grid()
plt.savefig("plot_variance.png")
plt.close()


print("All plots generated successfully!")
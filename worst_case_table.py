import pandas as pd

# -----------------------------------
# LOAD RESULTS
# -----------------------------------
df = pd.read_csv("results.csv")

# -----------------------------------
# FILTER STATISTICALLY WORST CONFIGURATION
# -----------------------------------
worst = df[
    (df["Energy"] == 0.2) &
    (df["Outage"] == 10) &
    (df["SleepDuration"] == 0) &
    (df["Delay"] == 1)
]

# -----------------------------------
# STAGE 1:
# Average trials within each seed
# -----------------------------------
seed_grouped = worst.groupby(
    [
        "MasterSeed",
        "Policy"
    ]
).mean(numeric_only=True).reset_index()


# -----------------------------------
# STAGE 2:
# Aggregate across seeds
# -----------------------------------
summary = seed_grouped.groupby("Policy").agg({
    "Avg_AoI": "mean",
    "Variance": "mean",
    "Peak_AoI": "mean",
    "P95_AoI": "mean",
    "Spike_Freq": "mean"
}).reset_index()

# -----------------------------------
# ROUND VALUES
# -----------------------------------
summary = summary.round(2)

# -----------------------------------
# SORT FOR READABILITY
# -----------------------------------
policy_order = ["Greedy", "Periodic", "Threshold", "RL"]

summary["Policy"] = pd.Categorical(
    summary["Policy"],
    categories=policy_order,
    ordered=True
)

summary = summary.sort_values("Policy")

# -----------------------------------
# PRINT TABLE
# -----------------------------------
print("\n=== WORST-CASE CONFIGURATION SUMMARY ===\n")

print(summary.to_string(index=False))

# -----------------------------------
# SAVE TABLE
# -----------------------------------
summary.to_csv("worst_case_summary.csv", index=False)

print("\nSaved as worst_case_summary.csv")
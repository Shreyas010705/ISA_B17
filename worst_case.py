import pandas as pd

# Load results
df = pd.read_csv("results.csv")

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
        "SleepDuration",
        "Delay",
        "Policy"
    ]
).agg({
    "Avg_AoI": "mean",
    "Variance": "mean",
    "Peak_AoI": "mean",
    "P95_AoI": "mean",
    "Spike_Freq": "mean"
}).reset_index()


# --------------------------------------------------
# RL ONLY
# --------------------------------------------------
rl = grouped[grouped["Policy"] == "RL"].copy()


# --------------------------------------------------
# DEFINE INSTABILITY SCORE
# --------------------------------------------------
rl["InstabilityScore"] = (
    rl["Peak_AoI"] +
    rl["P95_AoI"] +
    rl["Variance"]
)


# --------------------------------------------------
# GET STATISTICALLY WORST CONFIGURATION
# --------------------------------------------------
worst = rl.loc[rl["InstabilityScore"].idxmax()]

print("\n=== WORST RL CONFIGURATION ===\n")

print(f"Energy Rate     : {worst['Energy']}")
print(f"Battery Size    : {worst['Battery']}")
print(f"Outage Duration : {worst['Outage']}")
print(f"Sleep Duration  : {worst['SleepDuration']}")
print(f"Delay           : {worst['Delay']}")

print("\n--- Metrics ---")

print(f"Average AoI     : {worst['Avg_AoI']:.2f}")
print(f"Variance        : {worst['Variance']:.2f}")
print(f"Peak AoI        : {worst['Peak_AoI']:.2f}")
print(f"P95 AoI         : {worst['P95_AoI']:.2f}")
print(f"Spike Frequency : {worst['Spike_Freq']:.4f}")
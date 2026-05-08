import pandas as pd

# -------------------------
# LOAD DATA
# -------------------------
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
    "Peak_AoI": "mean",
    "Variance": "mean",
    "P95_AoI": "mean",
    "Spike_Freq": "mean"
}).reset_index()

# -------------------------
# 1. OVERALL PERFORMANCE
# -------------------------
print("\n===== OVERALL AVERAGE AoI =====")

overall = grouped.groupby("Policy")["Avg_AoI"].mean()

print(overall)

# -------------------------
# 2. BEST POLICY PER CONDITION
# -------------------------
print("\n===== BEST POLICY PER CONDITION =====")

best = grouped.loc[grouped.groupby(
    ["Energy", "Battery", "Outage", "SleepDuration", "Delay"]
)["Avg_AoI"].idxmin()]

print(best[[
    "Energy", "Battery", "Outage", "SleepDuration", "Delay", "Policy", "Avg_AoI"
]])

# -------------------------
# 3. EFFECT OF DELAY
# -------------------------
print("\n===== EFFECT OF DELAY =====")

delay_effect = grouped.groupby(["Delay", "Policy"])["Avg_AoI"].mean().unstack()

print(delay_effect)

# -------------------------
# 4. EFFECT OF ENERGY
# -------------------------
print("\n===== EFFECT OF ENERGY =====")

energy_effect = grouped.groupby(["Energy", "Policy"])["Avg_AoI"].mean().unstack()

print(energy_effect)

# -------------------------
# 5. PEAK AoI COMPARISON
# -------------------------
print("\n===== PEAK AoI =====")

peak = grouped.groupby("Policy")["Peak_AoI"].mean()

print(peak)

# -------------------------
# 6. VARIANCE (STABILITY)
# -------------------------
print("\n===== VARIANCE (Stability) =====")

variance = grouped.groupby("Policy")["Variance"].mean()

print(variance)

print("\nSummary complete ✅")
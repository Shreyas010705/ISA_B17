import pandas as pd

# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------
df = pd.read_csv("results.csv")

# --------------------------------------------------
# GROUP RESULTS
# --------------------------------------------------
grouped = df.groupby(
    ["Energy", "Outage", "SleepDuration", "Delay", "Policy"]
).agg({
    "Avg_AoI": "mean",
    "Variance": "mean",
    "Peak_AoI": "mean",
    "P95_AoI": "mean",
    "Spike_Freq": "mean"
}).reset_index()

# ==================================================
# 1. RL VARIANCE VS DELAY
# ==================================================
print("\n========================================")
print("RL VARIANCE VS DELAY")
print("========================================")

rl = grouped[grouped["Policy"] == "RL"]

delay_var = rl.groupby("Delay")["Variance"].mean()

for delay, value in delay_var.items():
    print(f"Delay {delay}: Variance = {value:.2f}")

# ==================================================
# 2. RL P95 VS DELAY
# ==================================================
print("\n========================================")
print("RL P95 AoI VS DELAY")
print("========================================")

delay_p95 = rl.groupby("Delay")["P95_AoI"].mean()

for delay, value in delay_p95.items():
    print(f"Delay {delay}: P95 AoI = {value:.2f}")

# ==================================================
# 3. RL SPIKE FREQUENCY VS DELAY
# ==================================================
print("\n========================================")
print("RL SPIKE FREQUENCY VS DELAY")
print("========================================")

delay_spike = rl.groupby("Delay")["Spike_Freq"].mean()

for delay, value in delay_spike.items():
    print(f"Delay {delay}: Spike Frequency = {value:.4f}")

# ==================================================
# 4. POLICY COMPARISON UNDER HARSH CONDITIONS
# ==================================================
print("\n========================================")
print("HARSH-CONDITION POLICY COMPARISON")
print("========================================")

harsh = grouped[
    (grouped["Energy"] == 0.2) &
    (grouped["Outage"] == 10) &
    (grouped["SleepDuration"] == 4) &
    (grouped["Delay"] == 3)
]

print(
    harsh[
        [
            "Policy",
            "Avg_AoI",
            "Variance",
            "Peak_AoI",
            "P95_AoI",
            "Spike_Freq"
        ]
    ].round(2).to_string(index=False)
)

# ==================================================
# 5. WORST RL CONFIGURATION
# ==================================================
print("\n========================================")
print("WORST RL CONFIGURATION")
print("========================================")

rl["InstabilityScore"] = (
    rl["Peak_AoI"] +
    rl["P95_AoI"] +
    rl["Variance"]
)

worst = rl.loc[rl["InstabilityScore"].idxmax()]

print(f"Energy Rate     : {worst['Energy']}")
print(f"Outage Duration : {worst['Outage']}")
print(f"Sleep Duration  : {worst['SleepDuration']}")
print(f"Delay           : {worst['Delay']}")

print("\n--- Metrics ---")

print(f"Average AoI     : {worst['Avg_AoI']:.2f}")
print(f"Variance        : {worst['Variance']:.2f}")
print(f"Peak AoI        : {worst['Peak_AoI']:.2f}")
print(f"P95 AoI         : {worst['P95_AoI']:.2f}")
print(f"Spike Frequency : {worst['Spike_Freq']:.4f}")

print("\n========================================")
print("SUMMARY COMPLETE")
print("========================================")
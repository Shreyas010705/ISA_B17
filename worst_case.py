import pandas as pd

# Load results
df = pd.read_csv("results.csv")

# Focus on RL only
rl = df[df["Policy"] == "RL"]

# Define instability score
# (simple combined severity metric)
rl["InstabilityScore"] = (
    rl["Peak_AoI"] +
    rl["P95_AoI"] +
    rl["Variance"]
)

# Get worst configuration
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
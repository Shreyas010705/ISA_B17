# ISA_B17
A Q-learning approach to optimize data freshness (AoI) in energy-harvesting systems under varying and extreme conditions.

This project simulates a wireless sensor system where updates must be sent to keep information fresh. However, the system has **limited and randomly arriving energy**, so it cannot send updates all the time.

The goal is to **decide when to send updates** such that:
- Data remains fresh (low Age of Information)
- Energy is used efficiently

We use **Q-learning (Reinforcement Learning)** to learn the best decision strategy and compare it with simple baseline methods.

---

## 🎯 Key Objective

> Minimize **Age of Information (AoI)** while managing limited energy resources.

---

## 🧩 What is Age of Information (AoI)?

AoI represents how old the last received update is.

Example:
- If an update was just sent → AoI = 0 ✅
- If no update is sent → AoI increases ❌


---

## 🧪 Step-by-Step Explanation

---

### 🔹 Step 1: Environment (environment.py)

This file simulates a real system:
- Energy arrives randomly
- Battery stores energy
- AoI increases if no update is sent
- Sending update consumes energy

👉 This is the “world” in which the agent operates.

---

### 🔹 Step 2: Agent (agent.py)

We use **Q-learning**, where the agent:
- Observes current state: (AoI, battery, time since last transmission)
- Chooses action:
  - 0 → Wait
  - 1 → Transmit
- Learns from reward:
  - +10 → successful update
  - Negative → stale data

---

### 🔹 Step 3: Baselines (baseline.py)

We compare RL with simple strategies:

1. **Greedy Policy**
   - Always transmit if possible
   - Wastes energy

2. **Periodic Policy**
   - Transmit at fixed intervals
   - Ignores system state

3. **Random Policy**
   - Random decisions

👉 These help evaluate if RL is actually better.

---

### 🔹 Step 4: Testing Environment (test_env.py)

This checks:
- Energy behavior
- AoI increase/reset
- Battery usage

👉 Ensures simulation is working correctly.

---

### 🔹 Step 5: Training RL Agent (test_run.py)

- Agent interacts with environment
- Learns over multiple episodes
- Improves decision-making

After training:
- We test RL, Greedy, and Periodic policies
- Compare their performance

---

### 🔹 Step 6: Large Experiments (run_experiments.py)

This is the **core research part**

We test under different conditions:

- Energy levels → low to high
- Battery sizes → small to large
- Duty cycles → strict to relaxed

Each combination is tested multiple times.

👉 Total simulations: **hundreds/thousands**

---

### 📊 Results Stored (results.csv)

Each row represents one experiment:

| Column | Meaning |
|--------|--------|
| Energy_Arrival_Rate | Probability of energy arrival |
| Battery_Capacity | Maximum battery size |
| Duty_Cycle | Minimum gap between transmissions |
| Policy_Type | RL / Greedy / Periodic |
| Average_AoI | Overall freshness (lower is better) |
| Peak_AoI | Worst-case delay |
| AoI_Variance | Stability of system |

---

## 🔍 Key Observations

From the results:

- RL performs better under **low energy conditions**
- Greedy policy causes **battery depletion and instability**
- Periodic policy is **not adaptive**
- Higher energy → lower variance (more stable system)

---

## 💥 Advanced Behavior (Extreme Conditions)

We introduced:
- Energy droughts (no energy for long time)
- Burst energy arrivals
- Transmission failures

👉 This helps simulate real-world uncertainty and system stress

---

## 🧠 Algorithm Used

We used **Q-learning**, a reinforcement learning algorithm:

- Learns from trial and error
- Updates decisions based on rewards
- Does not require prior knowledge of system

---

## 🏁 Conclusion

This project demonstrates that:

- Intelligent decision-making (RL) improves data freshness
- Energy-aware strategies outperform fixed rules
- System behavior changes significantly under extreme conditions




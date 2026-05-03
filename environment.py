import numpy as np

class AoIEnvironment:
    def __init__(self, energy_rate=0.5, battery_size=5, duty_cycle=1):
        self.energy_rate = energy_rate
        self.battery_size = battery_size
        self.duty_cycle = duty_cycle

        # NEW: extreme behavior controls
        self.drought_mode = False
        self.drought_timer = 0

        self.reset()

    def reset(self):
        self.aoi = 0
        self.battery = self.battery_size // 2
        self.time_since_last_tx = 0
        self.time = 0

        # reset drought
        self.drought_mode = False
        self.drought_timer = 0

        return self.get_state()

    def get_state(self):
        return (min(self.aoi, 15), self.battery, self.time_since_last_tx)

    def step(self, action):
        reward = 0

        # -----------------------------
        # 1. ENERGY ARRIVAL (REALISTIC)
        # -----------------------------

        # Occasionally enter drought mode
        if not self.drought_mode and np.random.rand() < 0.05:
            self.drought_mode = True
            self.drought_timer = np.random.randint(5, 15)

        if self.drought_mode:
            self.drought_timer -= 1
            if self.drought_timer <= 0:
                self.drought_mode = False
        else:
            # normal energy arrival
            if np.random.rand() < self.energy_rate:
                self.battery = min(self.battery + 1, self.battery_size)

            # occasional burst energy
            if np.random.rand() < 0.1:
                self.battery = min(self.battery + 2, self.battery_size)

        # -----------------------------
        # 2. TRANSMISSION CHECK
        # -----------------------------
        can_transmit = (
            self.battery > 0 and
            self.time_since_last_tx >= self.duty_cycle
        )

        if action == 1 and can_transmit:

            # NEW: transmission may fail
            if np.random.rand() < 0.85:  # 85% success rate
                self.aoi = 0
                reward = 10
            else:
                # failure → AoI still increases
                self.aoi += 1
                reward = -5

            self.battery -= 1
            self.time_since_last_tx = 0

        else:
            self.aoi += 1
            self.time_since_last_tx += 1
            reward = -self.aoi

        # -----------------------------
        # 3. PENALTY FOR LOW BATTERY
        # -----------------------------
        if self.battery == 0:
            reward -= 2

        self.time += 1

        return self.get_state(), reward, False
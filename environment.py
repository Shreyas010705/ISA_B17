import numpy as np


class AoIEnvironment:
    def __init__(
        self,
        energy_rate=0.5,
        battery_size=5,
        sleep_cycle=10,
        sleep_duration=3,
        outage_cycle=20,
        outage_duration=5
    ):
        # Core parameters
        self.energy_rate = energy_rate
        self.battery_size = battery_size

        # Duty cycle (CORRECTED)
        self.sleep_cycle = sleep_cycle
        self.sleep_duration = sleep_duration

        # Energy outage (controlled)
        self.outage_cycle = outage_cycle
        self.outage_duration = outage_duration

        self.reset()

    # -------------------------
    # Reset
    # -------------------------
    def reset(self):
        self.aoi = 0
        self.battery = self.battery_size // 2
        self.time = 0

        # Partial observability memory
        self.prev_aoi = 0
        self.prev_battery = self.battery
        self.prev_sleep = 0

        return self.get_state()

    # -------------------------
    # Observed state (delayed)
    # -------------------------
    def get_state(self):
        return (
            self.prev_aoi,
            self.prev_battery,
            self.prev_sleep
        )

    # -------------------------
    # Step
    # -------------------------
    def step(self, action):

        # -------------------------
        # 1. ENERGY ARRIVAL (CONTROLLED)
        # -------------------------
        if self.time % self.outage_cycle < self.outage_duration:
            energy_arrival = False
        else:
            energy_arrival = np.random.rand() < self.energy_rate

        if energy_arrival:
            self.battery = min(self.battery + 1, self.battery_size)

        # -------------------------
        # 2. DUTY CYCLE (FIXED)
        # -------------------------
        if self.time % self.sleep_cycle < self.sleep_duration:
            is_sleep = 1
        else:
            is_sleep = 0

        # If sleeping → cannot act
        if is_sleep == 1:
            action = 0

        # -------------------------
        # 3. ACTION EXECUTION (NO RATE LIMITING)
        # -------------------------
        can_transmit = self.battery > 0

        if action == 1 and can_transmit:
            self.aoi = 0
            self.battery -= 1
        else:
            self.aoi += 1

        # -------------------------
        # 4. REWARD
        # -------------------------
        reward = -self.aoi

        if action == 1 and can_transmit:
            reward -= 1  # transmission cost

        # -------------------------
        # 5. PARTIAL OBSERVABILITY UPDATE
        # -------------------------
        observed_aoi = self.prev_aoi
        observed_battery = self.prev_battery
        observed_sleep = self.prev_sleep

        self.prev_aoi = self.aoi
        self.prev_battery = self.battery
        self.prev_sleep = is_sleep

        # -------------------------
        # 6. TIME UPDATE
        # -------------------------
        self.time += 1

        return (
            (observed_aoi, observed_battery, observed_sleep),
            reward,
            False
        )

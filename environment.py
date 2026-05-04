import numpy as np


class AoIEnvironment:
    def __init__(
        self,
        energy_rate=0.5,
        battery_size=5,
        duty_cycle=3,
        outage_cycle=20,
        outage_duration=5
    ):
        # Core parameters
        self.energy_rate = energy_rate
        self.battery_size = battery_size
        self.duty_cycle = duty_cycle

        # Controlled energy disruption
        self.outage_cycle = outage_cycle
        self.outage_duration = outage_duration

        self.reset()

    # -------------------------
    # Reset environment
    # -------------------------
    def reset(self):
        self.aoi = 0
        self.battery = self.battery_size // 2
        self.time_since_last_tx = 0
        self.time = 0

        # Partial observability memory
        self.prev_aoi = 0
        self.prev_battery = self.battery

        return self.get_state()

    # -------------------------
    # Observed state (delayed)
    # -------------------------
    def get_state(self):
        return (
            self.prev_aoi,
            self.prev_battery,
            self.time_since_last_tx
        )

    # -------------------------
    # Step function
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
        # 2. DUTY CYCLE (SLEEP MODE)
        # -------------------------
        # Device is inactive during sleep slots
        if self.time % self.duty_cycle == 0:
            is_sleep = True
        else:
            is_sleep = False

        if is_sleep:
            action = 0  # forced wait

        # -------------------------
        # 3. ACTION EXECUTION
        # -------------------------
        can_transmit = (
            self.battery > 0 and
            self.time_since_last_tx >= self.duty_cycle
        )

        if action == 1 and can_transmit:
            self.aoi = 0
            self.battery -= 1
            self.time_since_last_tx = 0
        else:
            self.aoi += 1
            self.time_since_last_tx += 1

        # -------------------------
        # 4. REWARD (CLEAN)
        # -------------------------
        reward = -self.aoi

        if action == 1 and can_transmit:
            reward -= 1  # transmission cost

        # -------------------------
        # 5. PARTIAL OBSERVABILITY UPDATE
        # -------------------------
        observed_aoi = self.prev_aoi
        observed_battery = self.prev_battery

        self.prev_aoi = self.aoi
        self.prev_battery = self.battery

        # -------------------------
        # 6. TIME UPDATE
        # -------------------------
        self.time += 1

        return (
            (observed_aoi, observed_battery, self.time_since_last_tx),
            reward,
            False
        )

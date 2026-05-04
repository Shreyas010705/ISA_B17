import numpy as np


class AoIEnvironment:
    def __init__(
        self,
        energy_rate=0.5,
        battery_size=5,
        outage_cycle=20,
        outage_duration=5,
        delay_steps=1,
        sleep_cycle=10,
        sleep_duration=3
    ):
        self.energy_rate = energy_rate
        self.battery_size = battery_size

        # Energy outage (controlled)
        self.outage_cycle = outage_cycle
        self.outage_duration = outage_duration

        # Delay
        self.delay_steps = delay_steps

        # Deterministic sleep
        self.sleep_cycle = sleep_cycle
        self.sleep_duration = sleep_duration

        self.reset()

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        self.aoi = 1
        self.battery = self.battery_size // 2
        self.time = 0

        # buffer for delay
        self.state_buffer = []

        return self.get_state()

    # -------------------------
    # SAFE DELAY HANDLING
    # -------------------------
    def get_state(self):
        if len(self.state_buffer) == 0:
            return (1, self.battery, 0)

        if len(self.state_buffer) < self.delay_steps:
            return self.state_buffer[0]

        return self.state_buffer[-self.delay_steps]

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):

        # -------------------------
        # 1. ENERGY MODEL
        # -------------------------
        if self.time % self.outage_cycle < self.outage_duration:
            energy_arrival = False
        else:
            energy_arrival = np.random.rand() < self.energy_rate

        if energy_arrival:
            self.battery = min(self.battery + 1, self.battery_size)

        # -------------------------
        # 2. DUTY CYCLE (DETERMINISTIC)
        # -------------------------
        if self.time % self.sleep_cycle < self.sleep_duration:
            is_sleep = 1
        else:
            is_sleep = 0

        # Force no action during sleep
        if is_sleep == 1:
            action = 0

        # -------------------------
        # 3. ACTION EXECUTION
        # -------------------------
        can_transmit = self.battery > 0

        if action == 1 and can_transmit:
            self.aoi = 1
            self.battery -= 1
        else:
            self.aoi += 1

        # -------------------------
        # 4. REWARD
        # -------------------------
        reward = -self.aoi

        if action == 1 and can_transmit:
            reward -= 1

        # -------------------------
        # 5. STORE STATE FOR DELAY
        # -------------------------
        current_state = (self.aoi, self.battery, is_sleep)
        self.state_buffer.append(current_state)

        # -------------------------
        # 6. TIME UPDATE
        # -------------------------
        self.time += 1

        return self.get_state(), reward, False

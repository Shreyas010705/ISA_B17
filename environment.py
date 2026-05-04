import numpy as np


class AoIEnvironment:
    def __init__(
        self,
        energy_rate=0.5,
        battery_size=5,
        outage_cycle=20,
        outage_duration=5,
        delay_steps=1,
        sleep_prob=0.2,
        sleep_duration=3   # max sleep duration
    ):
        self.energy_rate = energy_rate
        self.battery_size = battery_size

        # Energy outage
        self.outage_cycle = outage_cycle
        self.outage_duration = outage_duration

        # Delay (partial observability)
        self.delay_steps = delay_steps

        # Sleep parameters
        self.sleep_prob = sleep_prob
        self.max_sleep_duration = sleep_duration

        self.reset()

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        self.aoi = 1
        self.battery = self.battery_size // 2
        self.time = 0

        # Sleep state
        self.sleep_timer = 0

        # Buffer for delayed state
        self.state_buffer = []

        return self.get_state()

    # -------------------------
    # GET DELAYED STATE
    # -------------------------
    def get_state(self):
        if len(self.state_buffer) < self.delay_steps:
            return (1, self.battery, 0)

        return self.state_buffer[-self.delay_steps]

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action):

        # -------------------------
        # ENERGY MODEL
        # -------------------------
        if self.time % self.outage_cycle < self.outage_duration:
            energy_arrival = False
        else:
            energy_arrival = np.random.rand() < self.energy_rate

        if energy_arrival:
            self.battery = min(self.battery + 1, self.battery_size)

        # -------------------------
        # STOCHASTIC SLEEP MODEL
        # -------------------------
        # Start sleep randomly
        if self.sleep_timer == 0 and np.random.rand() < self.sleep_prob:
            self.sleep_timer = np.random.randint(1, self.max_sleep_duration + 1)

        # Continue sleep
        if self.sleep_timer > 0:
            is_sleep = 1
            self.sleep_timer -= 1
        else:
            is_sleep = 0

        # Force wait during sleep
        if is_sleep == 1:
            action = 0

        # -------------------------
        # ACTION EXECUTION
        # -------------------------
        can_transmit = self.battery > 0

        if action == 1 and can_transmit:
            self.aoi = 1   # reset AoI
            self.battery -= 1
        else:
            self.aoi += 1

        # -------------------------
        # REWARD
        # -------------------------
        reward = -self.aoi

        if action == 1 and can_transmit:
            reward -= 1

        # -------------------------
        # STORE STATE FOR DELAY
        # -------------------------
        current_state = (self.aoi, self.battery, is_sleep)
        self.state_buffer.append(current_state)

        # -------------------------
        # TIME UPDATE
        # -------------------------
        self.time += 1

        return self.get_state(), reward, False

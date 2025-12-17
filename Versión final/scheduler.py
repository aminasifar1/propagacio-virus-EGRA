# calendar_sim.py
DAY_KEYS = ["mon", "tue", "wed", "thu", "fri"]
SLOTS_PER_DAY = 28  # (22-8)*60 / 30

class SimCalendar:
    def __init__(self, slot_sim_minutes=5, start_weekday=0):
        self.slot_sim_seconds = slot_sim_minutes * 60.0
        self.weekday = start_weekday   # 0=mon ... 4=fri
        self.t_day = 0.0               # segundos simulados desde las 08:00

    def step(self, dt_sim: float):
        self.t_day += dt_sim

        # rollover de día lectivo (28 slots)
        day_len = SLOTS_PER_DAY * self.slot_sim_seconds
        while self.t_day >= day_len:
            self.t_day -= day_len
            self.weekday += 1
            if self.weekday >= 5:
                self.weekday = 0  # saltas finde directamente al lunes

    def current_slot(self) -> int:
        return int(self.t_day // self.slot_sim_seconds)  # 0..27

    def time_in_slot(self) -> float:
        return self.t_day % self.slot_sim_seconds  # segundos dentro del slot actual

    def weekday_key(self) -> str:
        return DAY_KEYS[self.weekday]

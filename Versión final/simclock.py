from dataclasses import dataclass

@dataclass
class SimClock:
    start_hour: int = 8
    end_hour: int = 22
    day_sim_seconds: float = 20 * 60  # cuánto dura un día lectivo "en pantalla"
    speed_mult: float = 1.0           # turbo (x1, x2, x10...)

    def __post_init__(self):
        self.real_day_seconds = (self.end_hour - self.start_hour) * 3600.0  # 14h -> 50400s
        self.time_scale = self.real_day_seconds / float(self.day_sim_seconds)
        self.world_time = 0.0  # segundos real-equivalentes desde el inicio del “día lectivo”

    def step(self, dt_wall: float) -> float:
        """
        Avanza el reloj del mundo en segundos real-equivalentes.
        dt_wall: segundos reales por frame.
        return: dt_world (segundos real-equivalentes avanzados este frame)
        """
        dt_world = dt_wall * self.speed_mult * self.time_scale
        self.world_time += dt_world
        return dt_world

    def minute_of_day(self) -> int:
        """Minuto dentro del día lectivo [0..839] (08:00 -> 0)."""
        sec = self.world_time % self.real_day_seconds
        return int(sec // 60)

    def slot_30min(self) -> int:
        """Índice de slot de 30 min dentro del día lectivo."""
        return self.minute_of_day() // 30
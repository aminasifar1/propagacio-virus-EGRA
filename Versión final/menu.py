import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 400, 300
clock = pygame.time.Clock()

_CONTENT_BOTTOM = 0

def get_content_bottom():
    return _CONTENT_BOTTOM

FONT = None
FONT_SMALL = None

def ensure_fonts():
    global FONT, FONT_SMALL
    if FONT is None or FONT_SMALL is None:
        # Asegura que pygame.font está listo
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        FONT = pygame.font.Font(None, 36)
        FONT_SMALL = pygame.font.Font(None, 24)

# ============================================================
#     CLASE SLIDER
# ============================================================
class Slider:
    def __init__(self, x, y, width, label, vmin, vmax, initial):
        self.x = x
        self.y = y
        self.width = width
        self.height = 6
        self.radius = 10

        self.label = label
        self.min = vmin
        self.max = vmax
        self.value = initial

        self.dragging = False

    def draw(self, surf):
        ensure_fonts()
        
        # Texto Label
        text = FONT.render(self.label, True, (255, 255, 255))
        surf.blit(text, (self.x, self.y - 25))

        # Texto Valor
        val_text = FONT_SMALL.render(f"{self.value:.2f}", True, (255, 255, 255))
        surf.blit(val_text, (self.x + self.width + 20, self.y - 10))

        # Barra
        pygame.draw.rect(surf, (180, 180, 180),
            (self.x, self.y, self.width, self.height))

        # Posición knob
        t = (self.value - self.min) / (self.max - self.min)
        knob_x = self.x + int(t * self.width)
        knob_y = self.y + self.height // 2

        pygame.draw.circle(surf, (0, 150, 255), (knob_x, knob_y), self.radius)

    def handle_event(self, event):
        mx, my = pygame.mouse.get_pos()
        t = (self.value - self.min) / (self.max - self.min)
        knob_x = self.x + int(t * self.width)
        knob_y = self.y + self.height // 2

        if event.type == pygame.MOUSEBUTTONDOWN:
            if (mx - knob_x)**2 + (my - knob_y)**2 <= self.radius**2:
                self.dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        if event.type == pygame.MOUSEMOTION and self.dragging:
            # Nuevo valor basado en posición del mouse
            rel = (mx - self.x) / self.width
            rel = max(0, min(1, rel))
            self.value = self.min + rel * (self.max - self.min)

def render_menu(surface):
    global _CONTENT_BOTTOM

    surface.fill((30,30,30))
    slider1.draw(surface)
    slider2.draw(surface)
    slider3.draw(surface)

    _CONTENT_BOTTOM = slider3.y + 40

def handle_menu_event(event):
    for s in sliders:
        s.handle_event(event)

def render_stats_panel(surface, y, stats, width=400, height=260):
    """
    stats: dict con métricas ya calculadas desde el motor.
    y: posición vertical donde empieza el panel.
    """
    # Panel background
    panel_rect = pygame.Rect(10, y, width - 20, height)
    pygame.draw.rect(surface, (20, 20, 20), panel_rect, border_radius=10)
    pygame.draw.rect(surface, (70, 70, 70), panel_rect, width=2, border_radius=10)

    x = panel_rect.x + 10
    yy = panel_rect.y + 10

    title = FONT.render("Estado simulación", True, (255, 255, 255))
    surface.blit(title, (x, yy))
    yy += 30

    def line(label, value):
        nonlocal yy
        txt = FONT_SMALL.render(f"{label}: {value}", True, (235, 235, 235))
        surface.blit(txt, (x, yy))
        yy += 22

    line("Personas", stats.get("total", 0))
    line("Presentes", stats.get("present", 0))
    line("Infectados", stats.get("infected", 0))
    line("Sanos", stats.get("healthy", 0))
    line("Moviéndose", stats.get("moving", 0))
    line("Sentados", stats.get("seated", 0))
    line("Speed", f"x{stats.get('speed', 1.0)}")
    line("Día", stats.get("sim_day", "-"))
    line("Hora", stats.get("sim_time", "-"))
    if "sim_time" in stats:
        line("Hora sim", stats["sim_time"])

    yy += 10
    # Top aulas por infección (si lo pasas)
    top_rooms = stats.get("top_rooms", [])
    if top_rooms:
        subt = FONT_SMALL.render("Top salas infección:", True, (255, 255, 255))
        surface.blit(subt, (x, yy))
        yy += 20
        for name, val in top_rooms[:6]:
            txt = FONT_SMALL.render(f"{name}: {val:.3f}", True, (220, 220, 220))
            surface.blit(txt, (x, yy))
            yy += 18


# ============================================================
#     CREACIÓN DE LOS 3 SLIDERS
# ============================================================
slider1 = Slider(40, 80, 250, "Probabilidad de contagio", 0.0, 1.0, 0.5)
slider2 = Slider(40, 150, 250, "Distancia de contagio", 0.0, 20.0, 10.0)
slider3 = Slider(40, 220, 250, "Velocidad de simulacion", 0.5, 5.0, 1.0)


sliders = [slider1, slider2, slider3]


# ============================================================
#                       MAIN LOOP
# ============================================================
if __name__ == "__main__":
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Menu con Sliders")

    FONT = pygame.font.SysFont(None, 30)
    FONT_SMALL = pygame.font.SysFont(None, 24)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            for s in sliders:
                s.handle_event(event)

        screen.fill((25, 25, 25))

        for s in sliders:
            s.draw(screen)

        pygame.display.flip()
        clock.tick(60)

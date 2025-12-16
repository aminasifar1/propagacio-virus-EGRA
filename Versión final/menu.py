import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 400, 300
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Menu con Sliders")

FONT = pygame.font.SysFont(None, 30)
FONT_SMALL = pygame.font.SysFont(None, 24)

clock = pygame.time.Clock()


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
    surface.fill((30,30,30))
    slider1.draw(surface)
    slider2.draw(surface)
    slider3.draw(surface)

def handle_menu_event(event):
    for s in sliders:
        s.handle_event(event)

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

import pygame as pg

class InfectionBar:
    """Barra de progreso para mostrar el porcentaje de infectados."""

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Dimensiones de la barra
        self.bar_width = screen_width - 40  # Margen de 20px a cada lado
        self.bar_height = 40
        self.bar_x = 20
        self.bar_y = screen_height - 60  # 60px desde abajo

        # Colores
        self.bg_color = (30, 30, 35, 200)  # Fondo semi-transparente
        self.healthy_color = (100, 200, 100)  # Verde para sanos
        self.infected_color = (220, 50, 50)  # Rojo para infectados
        self.border_color = (200, 200, 200)  # Borde blanco
        self.text_color = (255, 255, 255)  # Texto blanco

        # Font para el texto
        pg.font.init()
        self.font = pg.font.Font(None, 28)
        self.small_font = pg.font.Font(None, 22)

    def render(self, screen, num_infected, total_people):
        """Renderiza la barra de infección."""
        if total_people == 0:
            return

        # Calcular porcentajes
        infected_ratio = num_infected / total_people
        healthy_ratio = 1.0 - infected_ratio

        # Crear superficie con transparencia para el fondo
        bg_surface = pg.Surface((self.bar_width + 10, self.bar_height + 40), pg.SRCALPHA)
        bg_surface.fill(self.bg_color)
        screen.blit(bg_surface, (self.bar_x - 5, self.bar_y - 5))

        # Dibujar borde de la barra
        pg.draw.rect(screen, self.border_color,
                     (self.bar_x, self.bar_y, self.bar_width, self.bar_height), 2)

        # Dibujar parte de sanos (verde)
        healthy_width = int(self.bar_width * healthy_ratio)
        if healthy_width > 0:
            pg.draw.rect(screen, self.healthy_color,
                         (self.bar_x, self.bar_y, healthy_width, self.bar_height))

        # Dibujar parte de infectados (rojo)
        infected_width = int(self.bar_width * infected_ratio)
        if infected_width > 0:
            pg.draw.rect(screen, self.infected_color,
                         (self.bar_x + healthy_width, self.bar_y, infected_width, self.bar_height))

        # Texto: Porcentajes
        infected_pct = infected_ratio * 100
        healthy_pct = healthy_ratio * 100

        # Texto principal en el centro
        main_text = f"Infectados: {infected_pct:.1f}% ({num_infected}/{total_people})"
        text_surface = self.font.render(main_text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(self.screen_width // 2, self.bar_y + self.bar_height // 2))

        # Sombra para el texto (mejor legibilidad)
        shadow_surface = self.font.render(main_text, True, (0, 0, 0))
        shadow_rect = shadow_surface.get_rect(center=(text_rect.centerx + 2, text_rect.centery + 2))
        screen.blit(shadow_surface, shadow_rect)
        screen.blit(text_surface, text_rect)

        # Etiquetas a los lados
        healthy_label = self.small_font.render(f"Sanos: {healthy_pct:.1f}%", True, self.healthy_color)
        infected_label = self.small_font.render(f"Infectados: {infected_pct:.1f}%", True, self.infected_color)

        # Posicionar etiquetas
        screen.blit(healthy_label, (self.bar_x, self.bar_y + self.bar_height + 5))
        infected_label_rect = infected_label.get_rect()
        infected_label_rect.topright = (self.bar_x + self.bar_width, self.bar_y + self.bar_height + 5)
        screen.blit(infected_label, infected_label_rect)

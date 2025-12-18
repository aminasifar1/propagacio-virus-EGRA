import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random


class Camera:
    def __init__(self, app):
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]

        # Posición y rotación de cámara tipo FPS
        self.position = glm.vec3(2, 4, 40)
        self.yaw = -90.0  # Ángulo horizontal
        self.pitch = 0.0  # Ángulo vertical
        self.front = glm.vec3(0, 0, -1)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.world_up = glm.vec3(0, 1, 0)

        # Velocidades ajustables
        self.move_speed = 5.0  # Unidades por segundo
        self.mouse_sensitivity = 0.1
        self.sprint_multiplier = 2.0

        self.perspective = True
        self.m_proj = self.get_projection_matrix()
        self.m_view = self.get_view_matrix()

        # Control del ratón
        self.mouse_captured = True
        pg.mouse.set_visible(False)
        pg.event.set_grab(True)

        self.camera_presets = [
            {'mode': 'free'},
            {
                'mode': 'fixed',
                'position': glm.vec3(-9.676, 8.998, -5.207),
                'target':   glm.vec3(-9.016, 8.687, -5.891),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(9.738, 4.098, -62.695),
                'target':   glm.vec3(10.201, 3.863, -61.841),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(17.237, 3.9, -33.186),
                'target':   glm.vec3(18.196, 3.735, -33.416),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(17.326, 4.317, -61.892),
                'target':   glm.vec3(18.279, 4.149, -62.145),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(17.185, 4.858, -90.798),
                'target':   glm.vec3(18.122, 4.667, -91.089),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(17.427, 5.735, -120.0),
                'target':   glm.vec3(18.38, 5.565, -120.251),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(17.27, 7.063, -46.063),
                'target':   glm.vec3(18.212, 6.819, -45.83),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(89.303, 7.08, -43.379),
                'target':   glm.vec3(89.529, 6.811, -44.315),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(89.245, 7.111, -43.363),
                'target':   glm.vec3(90.2, 6.991, -43.635),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(89.418, 8.155, -83.238),
                'target':   glm.vec3(89.532, 7.891, -84.196),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(89.418, 8.155, -83.238),
                'target':   glm.vec3(90.376, 7.99, -83.471),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(16.793, 7.25, -33.163),
                'target':   glm.vec3(17.751, 7.083, -33.398),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(16.971, 7.681, -61.937),
                'target':   glm.vec3(17.912, 7.492, -62.219),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(16.976, 8.139, -91.01),
                'target':   glm.vec3(17.936, 7.984, -91.247),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(16.863, 8.924, -120.083),
                'target':   glm.vec3(17.819, 8.739, -120.311),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(9.816, 8.783, -73.782),
                'target':   glm.vec3(10.2, 8.566, -72.885),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(9.985, 9.013, -134.387),
                'target':   glm.vec3(10.334, 8.755, -133.486),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(77.453, 6.826, -46.193),
                'target':   glm.vec3(77.259, 6.429, -45.296),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(77.0, 7.753, -74.767),
                'target':   glm.vec3(76.8, 7.32, -73.889),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(76.915, 8.151, -104.003),
                'target':   glm.vec3(76.743, 7.714, -103.12),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(77.738, 8.422, -132.775),
                'target':   glm.vec3(77.49, 8.028, -131.891),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(124.535, 10.369, -56.925),
                'target':   glm.vec3(124.273, 10.162, -55.983),
            }
        ]
        self.current_preset = 0
        self.mode = 'free' 

        # --- NUEVO: estado recordado de la cámara libre ---
        self.free_position = glm.vec3(self.position)
        self.free_yaw = self.yaw
        self.free_pitch = self.pitch

        self.update_vectors()
        self.apply_preset()

    def update_vectors(self):
        """Actualiza los vectores de dirección de la cámara."""
        front = glm.vec3()
        front.x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        front.y = math.sin(glm.radians(self.pitch))
        front.z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def apply_preset(self):
        """Aplica el preset actual (posición + orientación o modo libre)."""
        preset = self.camera_presets[self.current_preset]
        new_mode = preset.get('mode', 'free')

        # Si vamos a pasar de libre a fija, guardamos el estado libre
        if self.mode == 'free' and new_mode == 'fixed':
            self.free_position = glm.vec3(self.position)
            self.free_yaw = self.yaw
            self.free_pitch = self.pitch

        self.mode = new_mode

        if self.mode == 'free':
            self.position = glm.vec3(self.free_position)
            self.yaw = self.free_yaw
            self.pitch = self.free_pitch
            self.update_vectors()
            print(f"[CAM] Modo libre (preset {self.current_preset}) -> "
                  f"pos={self.position}, yaw={self.yaw:.1f}, pitch={self.pitch:.1f}")
            return

        pos = preset.get('position')
        target = preset.get('target')
        if pos is None or target is None:
            print("[CAM] Preset fijo mal definido")
            return
        
        self.position = glm.vec3(pos)
        dir_vec = glm.normalize(target - pos)

        # pitch: respecto al eje X (vertical)
        self.pitch = glm.degrees(glm.asin(dir_vec.y))
        # yaw: alrededor del eje Y; atan2(z, x) consistente con update_vectors
        self.yaw = glm.degrees(glm.atan(dir_vec.z, dir_vec.x))

        self.update_vectors()
        print(f"[CAM] Preset {self.current_preset} FIXED -> pos={pos}, target={target}")    
    
    def next_preset(self):
        self.current_preset = (self.current_preset + 1) % len(self.camera_presets)
        self.apply_preset()

    def prev_preset(self):
        self.current_preset = (self.current_preset - 1) % len(self.camera_presets)
        self.apply_preset()
    
    def handle_mouse(self, event):
        if event.type == pg.MOUSEMOTION and self.mouse_captured and self.mode == 'free':
            xoffset = event.rel[0] * self.mouse_sensitivity
            yoffset = -event.rel[1] * self.mouse_sensitivity

            self.yaw += xoffset
            self.pitch += yoffset

            # Limitar el pitch para evitar volteo
            if self.pitch > 89.0:
                self.pitch = 89.0
            if self.pitch < -89.0:
                self.pitch = -89.0

            self.update_vectors()

            self.free_yaw = self.yaw
            self.free_pitch = self.pitch

        elif event.type == pg.KEYDOWN:
            if event.key == pg.K_TAB:
                # Toggle captura del ratón
                self.mouse_captured = not self.mouse_captured
                pg.mouse.set_visible(not self.mouse_captured)
                pg.event.set_grab(self.mouse_captured)

    def move(self, delta_time):
        """Mueve la cámara según las teclas presionadas."""
        keys = pg.key.get_pressed()
        velocity = self.move_speed * delta_time

        if self.mode != 'free':
            return

        # Sprint con Shift
        if keys[pg.K_LSHIFT]:
            velocity *= self.sprint_multiplier

        # WASD movement
        moved = False
        if keys[pg.K_w]:
            self.position += self.front * velocity
            moved = True
        if keys[pg.K_s]:
            self.position -= self.front * velocity
            moved = True
        if keys[pg.K_a]:
            self.position -= self.right * velocity
            moved = True
        if keys[pg.K_d]:
            self.position += self.right * velocity
            moved = True

        # Arriba/Abajo con espacio y control
        if keys[pg.K_SPACE]:
            self.position += self.world_up * velocity
            moved = True
        if keys[pg.K_LCTRL]:
            self.position -= self.world_up * velocity
            moved = True

        if moved:
            self.free_position = glm.vec3(self.position)
            self.free_yaw = self.yaw
            self.free_pitch = self.pitch
    
    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)

    def get_projection_matrix(self):
        if self.perspective:
            return glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100)
        else:
            return glm.ortho(-8, 8, -8, 8, 0.1, 100)

    def update_matrices(self):
        self.m_view = self.get_view_matrix()
        self.m_proj = self.get_projection_matrix()

    def _vec3_literal(self, v: glm.vec3, decimals: int = 3) -> str:
        return f"glm.vec3({round(float(v.x), decimals)}, {round(float(v.y), decimals)}, {round(float(v.z), decimals)})"

    def capture_current_preset(self, distance: float = 1.0, append: bool = True) -> dict:
        """
        Captura un preset FIXED desde el estado actual:
        - position: posición actual
        - target: punto hacia donde mira (position + front * distance)
        """
        pos = glm.vec3(self.position)
        target = glm.vec3(self.position + self.front * distance)

        preset = {
            "mode": "fixed",
            "position": pos,
            "target": target,
        }

        idx = None
        if append:
            self.camera_presets.append(preset)
            idx = len(self.camera_presets) - 1

        # Snippet copiable para dejarlo permanente en self.camera_presets
        snippet = (
            "{\n"
            "    'mode': 'fixed',\n"
            f"    'position': {self._vec3_literal(pos)},\n"
            f"    'target':   {self._vec3_literal(target)},\n"
            "},"
        )

        if idx is None:
            print("[CAM] Captured preset (no añadido). Copia y pega esto:")
        else:
            print(f"[CAM] Captured preset añadido a camera_presets[{idx}]. Para hacerlo permanente, copia y pega esto:")
        print(snippet)

        return preset

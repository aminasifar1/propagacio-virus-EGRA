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
            # EJEMPLOS: ajusta posiciones/targets según tu escena
            {
                'mode': 'fixed',
                'position': glm.vec3(0.0, 15.0, 40.0),
                'target':   glm.vec3(0.0, 0.0, 0.0),
            },
            {
                'mode': 'fixed',
                'position': glm.vec3(40.0, 20.0, 0.0),
                'target':   glm.vec3(0.0, 0.0, 0.0),
            },
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

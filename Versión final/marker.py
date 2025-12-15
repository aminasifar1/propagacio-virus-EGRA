import numpy as np
import glm
import pygame as pg

class Marker:
    def __init__(self, ctx, camera):
        self.ctx = ctx
        self.camera = camera
        self.position = glm.vec3(0, 0, 0)
        self.speed = 0.05  # velocidad de movimiento
        self.size = 0.1   # tamaño del cubo marcador

        # Vertices de un cubo pequeño centrado en (0,0,0)
        s = self.size
        vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s,  s], [s, -s,  s], [s, s,  s], [-s, s,  s]
        ], dtype='f4')

        indices = np.array([
            0, 1, 2, 2, 3, 0,   # trasera
            4, 5, 6, 6, 7, 4,   # frontal
            0, 1, 5, 5, 4, 0,   # inferior
            2, 3, 7, 7, 6, 2,   # superior
            1, 2, 6, 6, 5, 1,   # derecha
            3, 0, 4, 4, 7, 3    # izquierda
        ], dtype='i4')

        # Programa simple
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                in vec3 in_vert;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(1.0, 0.0, 0.0, 1.0); // rojo
                }
            '''
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, '3f', 'in_vert')], self.ibo
        )

    # --- Movimiento del marcador ---
    def handle_input(self, keys):
        """Mover el marcador según las teclas presionadas."""
        # Detectar si Alt está presionado para aumentar velocidad
        speed_multiplier = 5.0 if (keys[pg.K_LALT] or keys[pg.K_RALT]) else 1.0
        # current_speed = self.speed * speed_multiplier
        current_speed = self.speed

        if keys[pg.K_UP]:
            self.position.z -= current_speed * speed_multiplier
        if keys[pg.K_DOWN]:
            self.position.z += current_speed * speed_multiplier
        if keys[pg.K_LEFT]:
            self.position.x -= current_speed * speed_multiplier
        if keys[pg.K_RIGHT]:
            self.position.x += current_speed * speed_multiplier
        if keys[pg.K_x]:
            self.position.y += current_speed * speed_multiplier
        if keys[pg.K_z]:
            self.position.y -= current_speed * speed_multiplier
        if keys[pg.K_RETURN]:
            self.print_position()

    # --- Mostrar posición actual ---
    def print_position(self):
        print(f"📍 Posición marcador: x={self.position.x:.2f}, y={self.position.y:.2f}, z={self.position.z:.2f}")

    # --- Dibujado del marcador ---
    def render(self):
        model = glm.translate(glm.mat4(1.0), self.position)
        self.prog['m_proj'].write(self.camera.m_proj)
        self.prog['m_view'].write(self.camera.m_view)
        self.prog['m_model'].write(model)
        self.vao.render()
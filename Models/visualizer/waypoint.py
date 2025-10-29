import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random

class Waypoint:
    """Punto de ruta para el sistema de navegación."""
    def __init__(self, position, connections=None):
        self.position = position
        self.connections = connections if connections else []



class WaypointVisualizer:
    """Renderitza la xarxa de Waypoints com a línies."""

    def __init__(self, ctx, camera):
        self.ctx = ctx
        self.camera = camera
        self.shader = self.get_shader()
        self.vbo = None
        self.vao = None
        self.vertex_count = 0

    def get_shader(self):
        """Shader simple per dibuixar línies d'un color pla."""
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 grid_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(grid_color, 1.0);
                }
            '''
        )

    def build_from_system(self, pathfinding_system):
        """Construeix el VBO i VAO a partir del sistema de pathfinding."""
        vertices = []
        drawn_connections = set()  # Per evitar dibuixar línies duplicades

        for wp in pathfinding_system.waypoints:
            for neighbor in wp.connections:
                # Creem una ID única per a la connexió per evitar duplicats
                # (ordre alfabètic basat en l'ID de memòria)
                edge = tuple(sorted((id(wp), id(neighbor))))

                if edge not in drawn_connections:
                    vertices.extend(wp.position)
                    vertices.extend(neighbor.position)
                    drawn_connections.add(edge)

        if not vertices:
            return  # No hi ha res a dibuixar

        self.vbo = self.ctx.buffer(np.array(vertices, dtype='f4'))
        self.vao = self.ctx.vertex_array(self.shader, [(self.vbo, '3f', 'in_position')])
        self.vertex_count = len(vertices) // 3  # Nombre total de vèrtexs individuals

    def render(self):
        """Dibuixa la graella."""
        if not self.vao:
            return  # No s'ha construït res

        m_model = glm.mat4(1.0)  # Model identity

        # Actualitzem uniformes (important per la càmera lliure)
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(m_model)
        self.shader['grid_color'].value = (0.0, 1.0, 0.0)  # Color verd

        self.ctx.line_width = 2.0  # Línies una mica més gruixudes
        self.vao.render(mode=mgl.LINES)
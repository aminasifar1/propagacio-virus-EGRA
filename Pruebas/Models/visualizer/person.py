import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random

from object3d import load_obj
from ring import Ring

class Person:
    def __init__(self, ctx, camera, obj_path, pathfinding_system, ground_y=0.0, is_infected=False):
        """
        Crea una persona que camina siguiendo waypoints.
        ground_y: Altura del suelo donde deben caminar las personas
        is_infected: Si la persona comença amb el 'ring'
        """
        self.ctx = ctx
        self.camera = camera
        self.pathfinding = pathfinding_system
        tri_data, normals, line_data, _ = load_obj(obj_path)
        self.tri_vbo = self.ctx.buffer(tri_data)
        self.nrm_vbo = self.ctx.buffer(normals)
        self.line_vbo = self.ctx.buffer(line_data)

        # Posición inicial en un waypoint aleatorio
        if self.pathfinding.waypoints:
            start_wp = random.choice(self.pathfinding.waypoints)
            self.position = glm.vec3(start_wp.position.x, ground_y, start_wp.position.z)
            self.current_waypoint = start_wp
            self.target_waypoint = random.choice(start_wp.connections) if start_wp.connections else start_wp
        else:
            self.position = glm.vec3(0, ground_y, 0)
            self.current_waypoint = None
            self.target_waypoint = None

        self.ground_y = ground_y
        self.speed = random.uniform(0.8, 1.5)
        self.rotation_angle = 0.0

        # --- LÒGICA D'INFECCIÓ ---
        self.ring = None
        if is_infected:
            self.infect()

    def update(self, delta_time):
        """Actualiza la posición siguiendo waypoints."""
        if not self.target_waypoint:
            return

        target_pos = glm.vec3(self.target_waypoint.position.x, self.ground_y, self.target_waypoint.position.z)
        direction = target_pos - self.position
        # Calculem la distància només en 2D (X, Z)
        distance = glm.length(glm.vec2(direction.x, direction.z))

        # Si llegó al waypoint, elige el siguiente
        if distance < 0.3:
            self.current_waypoint = self.target_waypoint
            if self.current_waypoint.connections:
                # Evitar tornar al waypoint immediatament anterior si n'hi ha més
                possible_targets = self.current_waypoint.connections
                if len(possible_targets) > 1 and self.current_waypoint in possible_targets:
                    # Aquesta lògica no és perfecta, hauria de guardar el 'previous_waypoint'
                    pass
                self.target_waypoint = random.choice(possible_targets)

            direction = glm.vec3(self.target_waypoint.position.x, self.ground_y,
                                 self.target_waypoint.position.z) - self.position
            distance = glm.length(glm.vec2(direction.x, direction.z))

        # Mueve hacia el objetivo
        if distance > 0:
            direction_2d = glm.normalize(glm.vec2(direction.x, direction.z))
            movement = glm.vec3(direction_2d.x, 0, direction_2d.y) * self.speed * delta_time
            self.position += movement
            self.position.y = self.ground_y  # Mantener en el suelo

            # Calcula el ángulo de rotación (orientar cap a on camina)
            self.rotation_angle = math.atan2(direction.x, direction.z)

        # --- ACTUALITZAR ANELL ---
        if self.ring:
            self.ring.position = self.position + glm.vec3(0, 0.05, 0)

    def get_model_matrix(self):
        """Retorna la matriz de modelo para esta persona."""
        m_model = glm.mat4(1.0)
        m_model = glm.translate(m_model, self.position)
        m_model = glm.rotate(m_model, self.rotation_angle, glm.vec3(0, 1, 0))
        # m_model = glm.scale(m_model, glm.vec3(0.8, 0.8, 0.8)) # Escala 0.8
        return m_model

    def render(self, shader, vao_tri, vao_line, light_pos):
        """Renderiza la persona i el seu anell si existeix."""

        # Renderitzar la persona (malla)
        m_model = self.get_model_matrix()
        shader['m_model'].write(m_model)
        shader['light_pos'].value = light_pos
        shader['view_pos'].value = tuple(self.camera.position)
        # Actualitzar vista/projecció per si la càmera s'ha mogut
        shader['m_proj'].write(self.camera.m_proj)
        shader['m_view'].write(self.camera.m_view)

        vao_tri.render(mode=mgl.TRIANGLES)
        self.ctx.line_width = 1.0
        vao_line.render(mode=mgl.LINES)

        # --- RENDERITZAR ANELL ---
        if self.ring:
            self.ring.render(light_pos)
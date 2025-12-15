import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random

class PuffSystem:
    """Sistema para gestionar múltiples partículas puff."""
    def __init__(self, ctx, camera):
        self.ctx = ctx
        self.camera = camera
        self.particles = []
    
    def create_puff(self, position, num_particles=8):
        """Crea un efecto puff en una posición."""
        colors = [
            (1.0, 0.3, 0.1),  # Naranja rojizo
            (1.0, 0.5, 0.0),  # Naranja
            (1.0, 0.7, 0.2),  # Amarillo anaranjado
        ]
        
        for _ in range(num_particles):
            color = random.choice(colors)
            particle = PuffParticle(self.ctx, self.camera, position, color)
            self.particles.append(particle)
    
    def update(self, delta_time):
        """Actualiza todas las partículas."""
        for particle in self.particles[:]:
            particle.update(delta_time)
            if particle.is_dead:
                self.particles.remove(particle)
    
    def render(self):
        """Renderiza todas las partículas."""
        # Habilitar blending para transparencia
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        
        for particle in self.particles:
            particle.render()
        
        self.ctx.disable(mgl.BLEND)

class PuffParticle:
    _shared = None  # Compartido entre todas las partículas
    """Partícula de humo para la animación de contagio."""
    def __init__(self, ctx, camera, position, color=(1.0, 0.5, 0.0)):
        self.ctx = ctx
        self.camera = camera
        self.position = glm.vec3(position)
        self.color = color
        self.lifetime = 1.0  # Duración en segundos
        self.age = 0.0
        self.scale = 0.1
        self.max_scale = 1.2
        self.velocity = glm.vec3(
            random.uniform(-0.5, 0.5),
            random.uniform(1.0, 2.0),  # Hacia arriba
            random.uniform(-0.5, 0.5)
        )
        self.is_dead = False
        
        # Crear geometría de esfera simple para la partícula
        self.create_sphere()
        
    def create_sphere(self):
        """STEP 4: Crea una esfera compartida para todas las partículas (mismo look)."""
        if PuffParticle._shared is not None:
            self.vbo = PuffParticle._shared["vbo"]
            self.shader = PuffParticle._shared["shader"]
            self.vao = PuffParticle._shared["vao"]
            return

        vertices = []
        segments = 8

        for i in range(segments):
            theta1 = (i / segments) * math.pi
            theta2 = ((i + 1) / segments) * math.pi

            for j in range(segments * 2):
                phi1 = (j / (segments * 2)) * 2 * math.pi
                phi2 = ((j + 1) / (segments * 2)) * 2 * math.pi

                v1 = (math.sin(theta1) * math.cos(phi1), math.cos(theta1), math.sin(theta1) * math.sin(phi1))
                v2 = (math.sin(theta1) * math.cos(phi2), math.cos(theta1), math.sin(theta1) * math.sin(phi2))
                v3 = (math.sin(theta2) * math.cos(phi2), math.cos(theta2), math.sin(theta2) * math.sin(phi2))
                v4 = (math.sin(theta2) * math.cos(phi1), math.cos(theta2), math.sin(theta2) * math.sin(phi1))

                vertices.extend([v1, v2, v3, v1, v3, v4])

        vbo = self.ctx.buffer(np.array(vertices, dtype="f4").flatten())
        shader = self.get_shader()
        vao = self.ctx.vertex_array(shader, [(vbo, "3f", "in_position")])

        PuffParticle._shared = {"vbo": vbo, "shader": shader, "vao": vao}

        self.vbo = vbo
        self.shader = shader
        self.vao = vao
    
    def get_shader(self):
        """Shader con transparencia para la partícula."""
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
                uniform vec3 particle_color;
                uniform float alpha;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(particle_color, alpha);
                }
            '''
        )
    
    def update(self, delta_time):
        """Actualiza la partícula."""
        self.age += delta_time
        
        if self.age >= self.lifetime:
            self.is_dead = True
            return
        
        # Movimiento
        self.position += self.velocity * delta_time
        
        # Desaceleración vertical (gravedad suave)
        self.velocity.y -= 2.0 * delta_time
        
        # Expansión rápida al inicio, luego se mantiene
        progress = self.age / self.lifetime
        if progress < 0.3:
            self.scale = self.max_scale * (progress / 0.3)
        else:
            self.scale = self.max_scale
    
    def render(self):
        """Renderiza la partícula."""
        if self.is_dead:
            return
        
        # Calcular alpha (transparencia) - se desvanece al final
        progress = self.age / self.lifetime
        alpha = 1.0 - progress  # De 1 a 0
        
        # Matriz de modelo
        m_model = glm.mat4(1.0)
        m_model = glm.translate(m_model, self.position)
        m_model = glm.scale(m_model, glm.vec3(self.scale))
        
        # Actualizar uniforms
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(m_model)
        self.shader['particle_color'].value = self.color
        self.shader['alpha'].value = alpha
        
        self.vao.render(mode=mgl.TRIANGLES)
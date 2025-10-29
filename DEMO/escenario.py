import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random

class Escenario:
    def __init__(self, ctx, camera, tri_data, normals, line_data, bounding_box):
        self.ctx = ctx
        self.bounding_box = bounding_box
        self.tri_vbo = self.ctx.buffer(tri_data)
        self.nrm_vbo = self.ctx.buffer(normals)
        self.line_vbo = self.ctx.buffer(line_data)
        self.shader = self.get_shader()
        self.vao_tri = self.ctx.vertex_array(
            self.shader,
            [(self.tri_vbo, '3f', 'in_position'),
             (self.nrm_vbo, '3f', 'in_normal')]
        )
        self.vao_line = self.ctx.vertex_array(self.shader, [(self.line_vbo, '3f', 'in_position')])
        self.m_model = glm.mat4()
        self.camera = camera
        self.light_angle = 0.0
        self.light_radius = 15.0
        self.light_speed = 0.0005
        self.update_uniforms()

    def get_shader(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                out vec3 v_normal;
                out vec3 v_frag_pos;
                void main() {
                    vec4 world_pos = m_model * vec4(in_position, 1.0);
                    v_frag_pos = world_pos.xyz;
                    // Normalizar correctamente la normal
                    mat3 normal_matrix = mat3(transpose(inverse(m_model)));
                    v_normal = normalize(normal_matrix * in_normal);
                    gl_Position = m_proj * m_view * world_pos;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_normal;
                in vec3 v_frag_pos;
                uniform vec3 light_pos;
                uniform vec3 view_pos;
                out vec4 fragColor;
                void main() {
                    vec3 color = vec3(0.6, 0.7, 1.0);

                    // Normalizar la normal interpolada
                    vec3 norm = normalize(v_normal);

                    // Diffuse
                    vec3 light_dir = normalize(light_pos - v_frag_pos);
                    float diff = max(dot(norm, light_dir), 0.0);

                    // Specular
                    vec3 view_dir = normalize(view_pos - v_frag_pos);
                    vec3 reflect_dir = reflect(-light_dir, norm);
                    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);

                    // Componentes
                    vec3 ambient = 0.3 * color;
                    vec3 diffuse = diff * color * 0.7;
                    vec3 specular = spec * vec3(1.0) * 0.5;

                    fragColor = vec4(ambient + diffuse + specular, 1.0);
                }
            '''
        )

    def update_light_position(self):
        """Actualitza la posició de la llum perquè roti al voltant de l'objecte."""
        # NOTA: He fet que la llum es mogui molt més lent (light_speed)
        self.light_angle += self.light_speed * (self.app.delta_time * 1000)  # Ajustat per delta_time
        light_x = self.light_radius * math.cos(self.light_angle)
        light_z = self.light_radius * math.sin(self.light_angle)
        return (light_x, 12.0, light_z)

    def update_uniforms(self):
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        # self.shader['light_pos'].value = self.update_light_position() # Es mourà al render
        self.shader['view_pos'].value = tuple(self.camera.position)

    def render(self):
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = self.update_light_position()
        self.shader['view_pos'].value = tuple(self.camera.position)
        # Actualitzar vista/projecció per si la càmera s'ha mogut
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)

        self.vao_tri.render(mode=mgl.TRIANGLES)
        self.ctx.line_width = 1.0
        self.vao_line.render(mode=mgl.LINES)

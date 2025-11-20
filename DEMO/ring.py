import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random

class Ring:
    """Anell amb volum, paràmetres ajustables i afectat per la il·luminació."""

    def __init__(self, ctx, camera, radius=0.9, thickness=0.15, height=0.1, segments=8, color=(1.0, 0.2, 0.2),
                 position=glm.vec3(0, 0, 0), altura= 1):
        self.ctx = ctx
        self.camera = camera
        self.color = color
        self.position = position
        self.m_model = glm.translate(glm.mat4(), self.position)
        self.contagion_radius = radius + thickness
        self.altura = altura

        # --- Generació de la geometria 3D i normals ---
        vertices = []
        normals = []

        r_outer = radius + thickness / 2
        r_inner = radius - thickness / 2

        for i in range(segments):
            theta1 = (i / segments) * 2 * math.pi
            theta2 = ((i + 1) / segments) * 2 * math.pi

            c1, s1 = math.cos(theta1), math.sin(theta1)
            c2, s2 = math.cos(theta2), math.sin(theta2)

            # Punts per a aquest segment
            p_ob1 = (c1 * r_outer, 0, s1 * r_outer)  # Outer-Bottom 1
            p_ot1 = (c1 * r_outer, height, s1 * r_outer)  # Outer-Top 1
            p_ib1 = (c1 * r_inner, 0, s1 * r_inner)  # Inner-Bottom 1
            p_it1 = (c1 * r_inner, height, s1 * r_inner)  # Inner-Top 1

            p_ob2 = (c2 * r_outer, 0, s2 * r_outer)  # Outer-Bottom 2
            p_ot2 = (c2 * r_outer, height, s2 * r_outer)  # Outer-Top 2
            p_ib2 = (c2 * r_inner, 0, s2 * r_inner)  # Inner-Bottom 2
            p_it2 = (c2 * r_inner, height, s2 * r_inner)  # Inner-Top 2

            # Normales
            n_up = (0, 1, 0)
            n_down = (0, -1, 0)
            n_out1 = (c1, 0, s1)
            n_out2 = (c2, 0, s2)
            n_in1 = (-c1, 0, -s1)
            n_in2 = (-c2, 0, -s2)

            # Cara superior (2 triangles)
            vertices.extend([p_it1, p_ot2, p_ot1, p_it1, p_it2, p_ot2])
            normals.extend([n_up] * 6)

            # Cara inferior (2 triangles)
            vertices.extend([p_ib1, p_ob1, p_ob2, p_ib1, p_ob2, p_ib2])
            normals.extend([n_down] * 6)

            # Cara exterior (2 triangles)
            vertices.extend([p_ob1, p_ot1, p_ot2, p_ob1, p_ot2, p_ob2])
            normals.extend([n_out1, n_out1, n_out2, n_out1, n_out2, n_out2])

            # Cara interior (2 triangles)
            vertices.extend([p_ib1, p_it2, p_it1, p_ib1, p_ib2, p_it2])
            normals.extend([n_in1, n_in2, n_in1, n_in1, n_in2, n_in2])

        self.vbo = self.ctx.buffer(np.array(vertices, dtype='f4').flatten())
        self.nbo = self.ctx.buffer(np.array(normals, dtype='f4').flatten())

        self.shader = self.get_shader()
        self.vao = self.ctx.vertex_array(
            self.shader,
            [(self.vbo, '3f', 'in_position'),
             (self.nbo, '3f', 'in_normal')]
        )
        self.update_uniforms()

    def get_shader(self):
        # Shader de Phong, similar al de l'objecte principal
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
                uniform vec3 ring_color;
                out vec4 fragColor;
                void main() {
                    vec3 norm = normalize(v_normal);

                    // Diffuse
                    vec3 light_dir = normalize(light_pos - v_frag_pos);
                    float diff = max(dot(norm, light_dir), 0.0);

                    // Specular
                    vec3 view_dir = normalize(view_pos - v_frag_pos);
                    vec3 reflect_dir = reflect(-light_dir, norm);
                    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);

                    // Components
                    vec3 ambient = 0.3 * ring_color;
                    vec3 diffuse = diff * ring_color * 0.7;
                    vec3 specular = spec * vec3(1.0) * 0.5;

                    fragColor = vec4(ambient + diffuse + specular, 1.0);
                }
            '''
        )

    def update_uniforms(self):
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['ring_color'].value = self.color
        self.shader['view_pos'].value = tuple(self.camera.position)

    def update(self, position):
        position.y = self.altura
        self.position = position

    def destroy(self):
        self.vbo.release()
        self.shader.release()
        self.vao.release()
    
    def render(self, light_pos):  # Afegim light_pos com a argument
        self.m_model = glm.translate(glm.mat4(), self.position)

        # Actualitzem les matrius de càmera i uniformes de llum CADA FRAME
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = light_pos
        self.shader['view_pos'].value = tuple(self.camera.position)

        self.vao.render(mode=mgl.TRIANGLES)
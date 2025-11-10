import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random


def load_obj(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[glm.vec3, glm.vec3]]:
    """
    Carrega vèrtexs, cares (triangles) i arestes d'un fitxer OBJ, calculant normals per vèrtex
    per a un ombrejat suau (Phong).

    Args:
        path (str): La ruta al fitxer .obj.

    Returns:
        tuple: Una tupla contenint:
            - np.ndarray: Vèrtexs dels triangles per al renderitzat.
            - np.ndarray: Normals per vèrtex per a la il·luminació.
            - np.ndarray: Vèrtexs de les arestes per al renderitzat de línies.
            - tuple: El bounding box (min_coords, max_coords) de l'objecte.
    """
    vertices = []
    faces = []
    edges = set()

    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    _, x, y, z = line.strip().split()
                    vertices.append((float(x), float(y), float(z)))
                elif line.startswith('f '):
                    # Gestiona cares amb 3 o més vèrtexs (triangulació simple)
                    parts = line.strip().split()[1:]
                    face_indices = [int(p.split('/')[0]) - 1 for p in parts]

                    # Triangula la cara si té més de 3 vèrtexs
                    for i in range(1, len(face_indices) - 1):
                        faces.append((face_indices[0], face_indices[i], face_indices[i + 1]))

                    # Afegeix les arestes de la cara
                    for i in range(len(face_indices)):
                        p1 = face_indices[i]
                        p2 = face_indices[(i + 1) % len(face_indices)]
                        edge = tuple(sorted((p1, p2)))
                        edges.add(edge)
    except FileNotFoundError:
        print(f"Error: El fitxer '{path}' no s'ha trobat.")
        raise
    except Exception as e:
        print(f"Error en processar el fitxer OBJ '{path}': {e}")
        raise

    if not vertices:
        return np.array([]), np.array([]), np.array([]), (glm.vec3(0), glm.vec3(0))

    # --- Càlcul del Bounding Box ---
    np_vertices = np.array(vertices, dtype='f4')
    min_coords = np.min(np_vertices, axis=0)
    max_coords = np.max(np_vertices, axis=0)
    bounding_box = (glm.vec3(min_coords), glm.vec3(max_coords))

    # --- Càlcul de Normals per Vèrtex (Phong Shading) ---
    vertex_normals = [np.zeros(3) for _ in range(len(vertices))]
    for face in faces:
        v0, v1, v2 = (np.array(vertices[i]) for i in face)
        face_normal = np.cross(v1 - v0, v2 - v0)

        # Acumula la normal de la cara a cada vèrtex que la compon
        for vertex_index in face:
            vertex_normals[vertex_index] += face_normal

    # Normalitza totes les normals acumulades
    vertex_normals = [
        v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0, 1, 0])
        for v in vertex_normals
    ]

    # --- Preparació de Dades per a OpenGL ---
    tri_vertices_data = []
    normals_data = []
    for face in faces:
        for vertex_index in face:
            tri_vertices_data.extend(vertices[vertex_index])
            normals_data.extend(vertex_normals[vertex_index])

    line_vertices_data = []
    for edge in edges:
        line_vertices_data.extend(vertices[edge[0]])
        line_vertices_data.extend(vertices[edge[1]])

    return (
        np.array(tri_vertices_data, dtype='f4'),
        np.array(normals_data, dtype='f4'),
        np.array(line_vertices_data, dtype='f4'),
        bounding_box
    )

class Object3D:
    def __init__(self, ctx, obj_path, camera):
        self.ctx = ctx
        tri_data, normals, line_data, bounding_box = load_obj(obj_path)
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

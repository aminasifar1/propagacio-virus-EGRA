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

class Camera:
    def __init__(self, app):
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]
        
        # Posición y rotación de cámara tipo FPS
        self.position = glm.vec3(0, 2, 5)
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
        
        self.update_vectors()

    def update_vectors(self):
        """Actualiza los vectores de dirección de la cámara."""
        front = glm.vec3()
        front.x = math.cos(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        front.y = math.sin(glm.radians(self.pitch))
        front.z = math.sin(glm.radians(self.yaw)) * math.cos(glm.radians(self.pitch))
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, self.world_up))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def handle_mouse(self, event):
        if event.type == pg.MOUSEMOTION and self.mouse_captured:
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
        
        # Sprint con Shift
        if keys[pg.K_LSHIFT]:
            velocity *= self.sprint_multiplier
        
        # WASD movement
        if keys[pg.K_w]:
            self.position += self.front * velocity
        if keys[pg.K_s]:
            self.position -= self.front * velocity
        if keys[pg.K_a]:
            self.position -= self.right * velocity
        if keys[pg.K_d]:
            self.position += self.right * velocity
        
        # Arriba/Abajo con espacio y control
        if keys[pg.K_SPACE]:
            self.position += self.world_up * velocity
        if keys[pg.K_LCTRL]:
            self.position -= self.world_up * velocity

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
        self.light_angle += self.light_speed * (self.app.delta_time * 1000) # Ajustat per delta_time
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

# --- INICI CLASSE RING (DE INFECTION_BLENDER.PY) ---
class Ring:
    """Anell amb volum, paràmetres ajustables i afectat per la il·luminació."""
    def __init__(self, ctx, camera, radius=0.9, thickness=0.15, height=0.1, segments=64, color=(1.0, 0.2, 0.2), position=glm.vec3(0,0,0)):
        self.ctx = ctx
        self.camera = camera
        self.color = color
        self.position = position
        self.m_model = glm.translate(glm.mat4(), self.position)

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
            p_ob1 = (c1 * r_outer, 0, s1 * r_outer) # Outer-Bottom 1
            p_ot1 = (c1 * r_outer, height, s1 * r_outer) # Outer-Top 1
            p_ib1 = (c1 * r_inner, 0, s1 * r_inner) # Inner-Bottom 1
            p_it1 = (c1 * r_inner, height, s1 * r_inner) # Inner-Top 1

            p_ob2 = (c2 * r_outer, 0, s2 * r_outer) # Outer-Bottom 2
            p_ot2 = (c2 * r_outer, height, s2 * r_outer) # Outer-Top 2
            p_ib2 = (c2 * r_inner, 0, s2 * r_inner) # Inner-Bottom 2
            p_it2 = (c2 * r_inner, height, s2 * r_inner) # Inner-Top 2

            # Normales
            n_up = (0, 1, 0)
            n_down = (0, -1, 0)
            n_out1 = (c1, 0, s1)
            n_out2 = (c2, 0, s2)
            n_in1 = (-c1, 0, -s1)
            n_in2 = (-c2, 0, -s2)

            # Cara superior (2 triangles)
            vertices.extend([p_it1, p_ot2, p_ot1,  p_it1, p_it2, p_ot2])
            normals.extend([n_up]*6)
            
            # Cara inferior (2 triangles)
            vertices.extend([p_ib1, p_ob1, p_ob2,  p_ib1, p_ob2, p_ib2])
            normals.extend([n_down]*6)

            # Cara exterior (2 triangles)
            vertices.extend([p_ob1, p_ot1, p_ot2,  p_ob1, p_ot2, p_ob2])
            normals.extend([n_out1, n_out1, n_out2,  n_out1, n_out2, n_out2])

            # Cara interior (2 triangles)
            vertices.extend([p_ib1, p_it2, p_it1,  p_ib1, p_ib2, p_it2])
            normals.extend([n_in1, n_in2, n_in1,  n_in1, n_in2, n_in2])

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

    def render(self, light_pos): # Afegim light_pos com a argument
        self.m_model = glm.translate(glm.mat4(), self.position)
        
        # Actualitzem les matrius de càmera i uniformes de llum CADA FRAME
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = light_pos
        self.shader['view_pos'].value = tuple(self.camera.position)
        
        self.vao.render(mode=mgl.TRIANGLES)


class Waypoint:
    """Punto de ruta para el sistema de navegación."""
    def __init__(self, position, connections=None):
        self.position = position
        self.connections = connections if connections else []

class PathfindingSystem:
    """Sistema simple de navegación con waypoints."""
    def __init__(self):
        self.waypoints = []
    
    def add_waypoint(self, position):
        """Añade un waypoint."""
        wp = Waypoint(glm.vec3(position))
        self.waypoints.append(wp)
        return wp
    
    def connect(self, wp1, wp2):
        """Conecta dos waypoints bidireccionalemente."""
        if wp2 not in wp1.connections:
            wp1.connections.append(wp2)
        if wp1 not in wp2.connections:
            wp2.connections.append(wp1)
    
    def get_nearest_waypoint(self, position):
        """Encuentra el waypoint más cercano a una posición."""
        if not self.waypoints:
            return None
        min_dist = float('inf')
        nearest = None
        for wp in self.waypoints:
            dist = glm.length(wp.position - position)
            if dist < min_dist:
                min_dist = dist
                nearest = wp
        return nearest
    

class PuffParticle:
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
        """Crea una esfera simple para representar la partícula."""
        vertices = []
        segments = 8
        
        for i in range(segments):
            theta1 = (i / segments) * math.pi
            theta2 = ((i + 1) / segments) * math.pi
            
            for j in range(segments * 2):
                phi1 = (j / (segments * 2)) * 2 * math.pi
                phi2 = ((j + 1) / (segments * 2)) * 2 * math.pi
                
                # Vértices de un quad en la esfera
                v1 = (math.sin(theta1) * math.cos(phi1), math.cos(theta1), math.sin(theta1) * math.sin(phi1))
                v2 = (math.sin(theta1) * math.cos(phi2), math.cos(theta1), math.sin(theta1) * math.sin(phi2))
                v3 = (math.sin(theta2) * math.cos(phi2), math.cos(theta2), math.sin(theta2) * math.sin(phi2))
                v4 = (math.sin(theta2) * math.cos(phi1), math.cos(theta2), math.sin(theta2) * math.sin(phi1))
                
                # Dos triángulos
                vertices.extend([v1, v2, v3, v1, v3, v4])
        
        self.vbo = self.ctx.buffer(np.array(vertices, dtype='f4').flatten())
        self.shader = self.get_shader()
        self.vao = self.ctx.vertex_array(self.shader, [(self.vbo, '3f', 'in_position')])
    
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
        
    def infect(self):
        """Infecta a la persona creant un anell."""
        if not self.ring:
            # Crea l'anell a la posició actual de la persona
            self.ring = Ring(
                self.ctx, self.camera,
                radius=0.9, thickness=0.15, height=0.1,
                position=self.position + glm.vec3(0, 0.05, 0) # Offset Y per l'anell
            )
            print("Una persona s'ha infectat!")

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
            
            direction = glm.vec3(self.target_waypoint.position.x, self.ground_y, self.target_waypoint.position.z) - self.position
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
        drawn_connections = set() # Per evitar dibuixar línies duplicades

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
            return # No hi ha res a dibuixar
            
        self.vbo = self.ctx.buffer(np.array(vertices, dtype='f4'))
        self.vao = self.ctx.vertex_array(self.shader, [(self.vbo, '3f', 'in_position')])
        self.vertex_count = len(vertices) // 3 # Nombre total de vèrtexs individuals

    def render(self):
        """Dibuixa la graella."""
        if not self.vao:
            return # No s'ha construït res

        m_model = glm.mat4(1.0) # Model identity
        
        # Actualitzem uniformes (important per la càmera lliure)
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(m_model)
        self.shader['grid_color'].value = (0.0, 1.0, 0.0) # Color verd

        self.ctx.line_width = 2.0 # Línies una mica més gruixudes
        self.vao.render(mode=mgl.LINES)


class ViewerApp:
    def __init__(self, obj_path, win_size=(1536, 864)):
        pg.init()
        pg.display.set_caption("3D Viewer - WASD para moverte, TAB para soltar ratón")
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.front_face = 'ccw'  # Counter-clockwise para consistencia
        self.aspect_ratio = win_size[0] / win_size[1]
        self.camera = Camera(self)
        # Sistema de partículas para efectos puff
        self.puff_system = PuffSystem(self.ctx, self.camera)
        
        self.clock = pg.time.Clock()
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.delta_time = 0.016 # Temps inicial per evitar pics

        self.tick_duration = 0.2  # Segons per tick (p.ex., 0.5 = 2 ticks per segon)
        self.tick_timer = 0.0
        self.infection_probability = 1 # probabilitat d'infecció per tick

        self.object = Object3D(self.ctx, obj_path, self.camera)
        self.object.app = self # Donem accés a 'app' a l'objecte pel delta_time

        # Control de la visualització de la graella
        self.show_grid = False

        # Obtenim el Bounding Box de l'escenari
        min_coords, max_coords = self.object.bounding_box
        print(f"Escenari carregat. Bounding Box:")
        print(f"  MIN: {min_coords}")
        print(f"  MAX: {max_coords}")
        
        # Sistema de navegación con waypoints
        self.pathfinding = PathfindingSystem()
        self.setup_waypoints(self.object.bounding_box)

        # Visualitzador de Waypoints
        self.waypoint_visualizer = WaypointVisualizer(self.ctx, self.camera)
        self.waypoint_visualizer.build_from_system(self.pathfinding)
        
        # Crear personas que caminan
        self.people = []
        num_people = 50
        ground_y = min_coords.y + 0.1
        print(f"Terra (ground_y) establert a: {ground_y}")

        try:
            for i in range(num_people):
                is_infected = (i == 0) 
                person = Person(
                    self.ctx, self.camera, 
                    "person_1.obj", 
                    self.pathfinding, 
                    ground_y, # Passem el ground_y dinàmic
                    is_infected=is_infected
                )
                self.people.append(person)
            
            # ... (la resta de __init__ és igual)
            if self.people:
                first_person = self.people[0]
                self.person_vao_tri = self.ctx.vertex_array(
                    self.object.shader,
                    [(first_person.tri_vbo, '3f', 'in_position'),
                     (first_person.nrm_vbo, '3f', 'in_normal')]
                )
                self.person_vao_line = self.ctx.vertex_array(
                    self.object.shader,
                    [(first_person.line_vbo, '3f', 'in_position')]
                )
            else:
                self.person_vao_tri = None
                self.person_vao_line = None

        except FileNotFoundError:
            print("Advertencia: No se encontró person_1.obj. No se crearan personas.")
            self.people = []
            self.person_vao_tri = None
            self.person_vao_line = None


    def setup_waypoints(self, bounding_box):
        """Configura una red de waypoints AUTOMÀTICAMENT basada en el Bounding Box."""
        
        min_coords, max_coords = bounding_box
        ground_y = min_coords.y # El terra
        
        wp_grid = {}
        spacing = 2.0 # Distància entre waypoints (pots ajustar-la)
        
        # Calculem el rang del grid basat en el BBox
        # Afegim +1 als 'end' per assegurar que cobrim la cantonada
        x_start = int(min_coords.x / spacing)
        x_end = int(max_coords.x / spacing) + 1
        
        z_start = int(min_coords.z / spacing)
        z_end = int(max_coords.z / spacing) + 1

        print(f"Generant grid de waypoints de ({x_start},{z_start}) a ({x_end},{z_end})")

        # Generem els punts
        for x in range(x_start, x_end):
            for z in range(z_start, z_end):
                pos = (x * spacing, ground_y, z * spacing)
                wp = self.pathfinding.add_waypoint(pos)
                wp_grid[(x, z)] = wp
        
        # Conectar waypoints adyacentes (grid)
        for x in range(x_start, x_end):
            for z in range(z_start, z_end):
                current = wp_grid.get((x, z))
                if current:
                    # Conectar con vecinos (8 direccions)
                    for dx, dz in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]:
                        neighbor = wp_grid.get((x+dx, z+dz))
                        if neighbor:
                            self.pathfinding.connect(current, neighbor)

    def check_infections(self):
        """Comprova col·lisions per transferir infecció."""
        if not self.people:
            return

        infected_people = [p for p in self.people if p.ring]
        uninfected_people = [p for p in self.people if not p.ring]
        
        if not uninfected_people:
            return

        newly_infected = []
        infection_radius = 1.0

        for infected in infected_people:
            for uninfected in uninfected_people:
                if uninfected in newly_infected:
                    continue
                
                dist = glm.length(infected.position - uninfected.position)
                
                if dist < infection_radius:
                    if random.random() < self.infection_probability:
                        newly_infected.append(uninfected)
        
        # Infectar als nous i crear efecte puff
        for person_to_infect in newly_infected:
            person_to_infect.infect()
            # CREAR EFECTO PUFF EN LA POSICIÓN DE LA PERSONA
            puff_position = person_to_infect.position + glm.vec3(0, 1.0, 0)
            self.puff_system.create_puff(puff_position, num_particles=12)

    def run(self):
        last_frame_time = time.time()
        
        while True:
            current_frame_time = time.time()
            self.delta_time = current_frame_time - last_frame_time
            if self.delta_time == 0: # Evitar divisió per zero
                self.delta_time = 1e-6 
            last_frame_time = current_frame_time
            
            for e in pg.event.get():
                self.camera.handle_mouse(e)
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    pg.event.set_grab(False)
                    pg.quit()
                    sys.exit()
                if e.type == pg.KEYDOWN:
                    if e.key == pg.K_g:
                        self.show_grid = not self.show_grid
                        print(f"Graella de waypoints: {'Visible' if self.show_grid else 'Oculta'}")

            # --- LÒGICA D'ACTUALITZACIÓ ---
            
            # Mover cámara
            self.camera.move(self.delta_time)
            
            # Actualizar personas (moviment)
            for person in self.people:
                person.update(self.delta_time)
                
            # Actualizar sistema de partículas
            self.puff_system.update(self.delta_time)

            # Acumulem el temps del frame al comptador de ticks
            self.tick_timer += self.delta_time

            # Si ha passat prou temps per un tick, executem la comprovació
            if self.tick_timer >= self.tick_duration:
                self.tick_timer -= self.tick_duration # Restem la duració per no perdre temps acumulat
                self.check_infections()
            
            # Actualitzar matrius de càmera
            self.camera.update_matrices()
            
            # --- RENDERITZAT ---
            self.ctx.clear(0.07, 0.07, 0.09)
            
            # Renderizar el objeto principal (escenari)
            self.object.render()

            # Renderitzar la graella de waypoints si està activada
            if self.show_grid:
                self.waypoint_visualizer.render()
            
            # Renderizar personas
            if self.people and self.person_vao_tri:
                light_pos = self.object.update_light_position() # Obtenim la posició de la llum
                for person in self.people:
                    person.render(self.object.shader, self.person_vao_tri, 
                                self.person_vao_line, light_pos)
                    
            # Renderizar sistema de partículas (después de todo lo demás)
            self.puff_system.render()

            # --- Càlcul FPS ---
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time
                pg.display.set_caption(f"3D Viewer - FPS: {self.fps:.1f} - WASD moverte, TAB soltar ratón")

            pg.display.flip()
            # self.clock.tick() # No fem servir tick() per tenir delta_time real

if __name__ == "__main__":
    obj_path = "OBJ.obj" # El teu escenari
    try:
        app = ViewerApp(obj_path)
        app.run()
    except FileNotFoundError:
        print(f"Error: No s'ha trobat el fitxer {obj_path}.")
        print("Assegura't que el fitxer .obj de l'escenari està a la mateixa carpeta.")
        sys.exit()
    except Exception as e:
        print(f"S'ha produït un error: {e}")
        pg.quit()
        sys.exit()
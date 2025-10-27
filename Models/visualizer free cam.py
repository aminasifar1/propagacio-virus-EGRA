import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math

def load_obj(path):
    """Carrega vèrtexs, cares (triangles) i arestes del fitxer OBJ."""
    vertices = []
    faces = []
    edges = set()

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = [int(p.split('/')[0]) - 1 for p in parts]
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        faces.append((face[0], face[i], face[i + 1]))
                    for i in range(len(face)):
                        a, b = face[i], face[(i + 1) % len(face)]
                        edge = tuple(sorted((a, b)))
                        edges.add(edge)

    tri_vertices = []
    normals = []
    for a, b, c in faces:
        v0 = np.array(vertices[a])
        v1 = np.array(vertices[b])
        v2 = np.array(vertices[c])
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / np.linalg.norm(normal)
        for v in (v0, v1, v2):
            tri_vertices.extend(v)
            normals.extend(normal)

    line_vertices = []
    for a, b in edges:
        line_vertices.extend(vertices[a])
        line_vertices.extend(vertices[b])

    return (
        np.array(tri_vertices, dtype='f4'),
        np.array(normals, dtype='f4'),
        np.array(line_vertices, dtype='f4')
    )

class Camera:
    def __init__(self, app):
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]
        self.radius = 8.0
        self.theta = math.radians(45)
        self.phi = math.radians(45)
        self.target = glm.vec3(0, 0, 0)
        self.up = glm.vec3(0, 1, 0)
        self.position = self.get_position()
        self.perspective = True
        self.m_proj = self.get_projection_matrix()
        self.m_view = self.get_view_matrix()
        self.last_mouse_pos = None
        self.sensitivity = 0.005
        self.zoom_speed = 0.5
        self.pan_speed = 0.01

    def handle_mouse(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:  # Botó esquerre -> rotació de la càmera
                self.last_mouse_pos = pg.mouse.get_pos()
                pg.mouse.set_visible(False)
            elif event.button == 3:  # Botó dret -> desplaçament
                self.last_mouse_pos = pg.mouse.get_pos()
                pg.mouse.set_visible(False)
            elif event.button == 4:  # Roda amunt -> zoom in
                self.radius = max(1.5, self.radius - self.zoom_speed)
                self.update_matrices()
            elif event.button == 5:  # Roda avall -> zoom out
                self.radius += self.zoom_speed
                self.update_matrices()

        elif event.type == pg.MOUSEBUTTONUP:
            if event.button == 1 or event.button == 3:  # Botó esquerre o dret
                self.last_mouse_pos = None
                pg.mouse.set_visible(True)

        elif event.type == pg.MOUSEMOTION and self.last_mouse_pos:
            dx, dy = event.rel
            self.last_mouse_pos = pg.mouse.get_pos()

            if event.buttons[2]:  # Botó dret premut -> desplaçament
                right = glm.normalize(glm.cross(self.position - self.target, self.up))
                up = glm.normalize(glm.cross(right, self.position - self.target))
                delta = right * (dx * self.pan_speed) + up * (-dy * self.pan_speed)
                self.target += delta
                self.update_matrices()
            elif event.buttons[0]:  # Botó esquerre premut -> rotació de la càmera
                self.theta -= -dx * self.sensitivity  # Invertim signe
                self.phi += dy * self.sensitivity   # Invertim signe
                self.phi = max(0.1, min(math.pi - 0.1, self.phi))
                self.update_matrices()

    def reset(self):
        self.radius = 8.0
        self.theta = math.radians(45)
        self.phi = math.radians(45)
        self.target = glm.vec3(0, 0, 0)
        self.update_matrices()

    def get_position(self):
        x = self.radius * math.sin(self.phi) * math.cos(self.theta)
        y = self.radius * math.cos(self.phi)
        z = self.radius * math.sin(self.phi) * math.sin(self.theta)
        return glm.vec3(x, y, z) + self.target

    def get_view_matrix(self):
        self.position = self.get_position()
        return glm.lookAt(self.position, self.target, self.up)

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
        tri_data, normals, line_data = load_obj(obj_path)
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
        self.light_radius = 10.0
        self.light_speed = 0.0001
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
                    v_normal = mat3(transpose(inverse(m_model))) * in_normal;
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
                    vec3 norm = normalize(v_normal);
                    vec3 light_dir = normalize(light_pos - v_frag_pos);
                    float diff = max(dot(norm, light_dir), 0.0);
                    vec3 view_dir = normalize(view_pos - v_frag_pos);
                    vec3 reflect_dir = reflect(-light_dir, norm);
                    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
                    vec3 ambient = 0.2 * color;
                    vec3 diffuse = diff * color;
                    vec3 specular = spec * vec3(1.0);
                    fragColor = vec4(ambient + diffuse + specular, 1.0);
                }
            '''
        )

    def update_light_position(self):
        """Actualitza la posició de la llum perquè roti al voltant de l'objecte."""
        self.light_angle += self.light_speed
        light_x = self.light_radius * math.cos(self.light_angle)
        light_z = self.light_radius * math.sin(self.light_angle)
        return (light_x, 10.0, light_z)

    def update_uniforms(self):
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = self.update_light_position()
        self.shader['view_pos'].value = tuple(self.camera.position)

    def render(self):
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = self.update_light_position()
        self.shader['view_pos'].value = tuple(self.camera.position)
        self.vao_tri.render(mode=mgl.TRIANGLES)
        self.ctx.line_width = 1.0
        self.vao_line.render(mode=mgl.LINES)

class ViewerApp:
    def __init__(self, obj_path, win_size=(1080, 1920)):
        pg.init()
        pg.display.set_caption("3D Object Viewer with Lighting")
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.aspect_ratio = win_size[0] / win_size[1]
        self.camera = Camera(self)
        self.object = Object3D(self.ctx, obj_path, self.camera)
        self.clock = pg.time.Clock()
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def run(self):
        pg.event.set_grab(True)
        while True:
            for e in pg.event.get():
                self.camera.handle_mouse(e)
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    pg.event.set_grab(False)
                    pg.quit()
                    sys.exit()

            self.camera.update_matrices()
            self.ctx.clear(0.07, 0.07, 0.09)
            self.object.update_uniforms()
            self.object.render()

            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time
                pg.display.set_caption(f"3D Object Viewer with Lighting - FPS: {self.fps:.1f}")

            pg.display.flip()
            self.clock.tick()

if __name__ == "__main__":
    obj_path = "OBJ.obj"  # Assegura't que aquest fitxer existeixi!
    app = ViewerApp(obj_path)
    app.run()
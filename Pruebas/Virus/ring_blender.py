import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys


def load_obj(path):
    """Carrega vèrtexs, cares i normals d’un fitxer OBJ."""
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
                        edges.add(tuple(sorted((a, b))))

    tri_vertices = []
    normals = []
    for a, b, c in faces:
        v0, v1, v2 = np.array(vertices[a]), np.array(vertices[b]), np.array(vertices[c])
        normal = np.cross(v1 - v0, v2 - v0)
        nrm_len = np.linalg.norm(normal)
        normal = normal / nrm_len if nrm_len > 0 else np.array([0, 1, 0])
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
    def __init__(self, aspect):
        self.position = glm.vec3(10, 10, -15)
        self.target = glm.vec3(0, 0, 0)
        self.up = glm.vec3(0, 1, 0)
        self.m_view = glm.lookAt(self.position, self.target, self.up)
        self.m_proj = glm.perspective(glm.radians(35), aspect, 0.1, 200)


class Object3D:
    def __init__(self, ctx, obj_path, camera):
        self.ctx = ctx
        tri_data, normals, line_data = load_obj(obj_path)
        self.tri_vbo = self.ctx.buffer(tri_data)
        self.nrm_vbo = self.ctx.buffer(normals)
        self.line_vbo = self.ctx.buffer(line_data)
        self.camera = camera
        self.shader = self.get_shader()
        self.vao_tri = self.ctx.vertex_array(
            self.shader,
            [(self.tri_vbo, '3f', 'in_position'),
             (self.nrm_vbo, '3f', 'in_normal')]
        )
        self.vao_line = self.ctx.vertex_array(self.shader, [(self.line_vbo, '3f', 'in_position')])
        self.m_model = glm.mat4()
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

    def update_uniforms(self):
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = (10.0, 10.0, -10.0)
        self.shader['view_pos'].value = tuple(self.camera.position)

    def render(self):
        self.shader['m_model'].write(self.m_model)
        self.vao_tri.render(mode=mgl.TRIANGLES)
        self.ctx.line_width = 1.0
        self.vao_line.render(mode=mgl.LINES)


class Ring:
    """Anell amb paràmetres ajustables: radi i gruix."""
    def __init__(self, ctx, camera, radius=0.9, thickness=0.15, segments=128, color=(1.0, 0.2, 0.2)):
        self.ctx = ctx
        self.camera = camera
        self.radius = radius
        self.thickness = thickness
        self.color = color
        self.m_model = glm.translate(glm.mat4(), glm.vec3(0, 0.05, 0))

        # Construcció del cercle
        theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        outer = np.array([(np.cos(t) * (radius + thickness / 2), 0.0, np.sin(t) * (radius + thickness / 2)) for t in theta])
        inner = np.array([(np.cos(t) * (radius - thickness / 2), 0.0, np.sin(t) * (radius - thickness / 2)) for t in theta])

        # Bucle tancat
        ring_vertices = []
        for i in range(segments):
            i_next = (i + 1) % segments
            ring_vertices.extend(outer[i])
            ring_vertices.extend(inner[i])
            ring_vertices.extend(outer[i_next])
            ring_vertices.extend(inner[i])
            ring_vertices.extend(inner[i_next])
            ring_vertices.extend(outer[i_next])

        ring_vertices = np.array(ring_vertices, dtype='f4')
        self.vbo = ctx.buffer(ring_vertices)
        self.shader = self.get_shader()
        self.vao = ctx.vertex_array(self.shader, [(self.vbo, '3f', 'in_position')])
        self.update_uniforms()

    def get_shader(self):
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
                uniform vec3 ring_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(ring_color, 1.0);
                }
            '''
        )

    def update_uniforms(self):
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['ring_color'].value = self.color

    def render(self):
        self.shader['m_model'].write(self.m_model)
        self.vao.render(mode=mgl.TRIANGLES)


class ViewerApp:
    def __init__(self, person_path, win_size=(600, 600)):
        pg.init()
        pg.display.set_caption("Person with Adjustable Ring (radius + width)")
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.aspect_ratio = win_size[0] / win_size[1]
        self.camera = Camera(self.aspect_ratio)

        # Models
        self.person = Object3D(self.ctx, person_path, self.camera)
        self.ring = Ring(self.ctx, self.camera, radius=0.9, thickness=0.15)

        # Models
        self.person_no_ring = Object3D(self.ctx, "person2_noring.obj", self.camera)  # nou
        self.ring = Ring(self.ctx, self.camera, radius=0.9, thickness=0.15)

        self.clock = pg.time.Clock()

    def run(self):
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    pg.quit()
                    sys.exit()
                elif e.type == pg.KEYDOWN:
                    # Radi
                    if e.key == pg.K_UP:
                        self.ring.radius += 0.1
                    elif e.key == pg.K_DOWN:
                        self.ring.radius = max(0.3, self.ring.radius - 0.1)
                    # Amplada (thickness)
                    elif e.key == pg.K_RIGHT:
                        self.ring.thickness = min(1.0, self.ring.thickness + 0.05)
                    elif e.key == pg.K_LEFT:
                        self.ring.thickness = max(0.02, self.ring.thickness - 0.05)

                    # Re-crea l'anell amb nous valors
                    self.ring = Ring(
                        self.ctx, self.camera,
                        radius=self.ring.radius,
                        thickness=self.ring.thickness
                    )

            self.ctx.clear(0.07, 0.07, 0.09)
            self.person.render()
            self.person_no_ring.render()  # nou
            self.ring.render()
            pg.display.flip()
            self.clock.tick()


if __name__ == "__main__":
    person_path = "person2.obj"
    app = ViewerApp(person_path)
    app.run()

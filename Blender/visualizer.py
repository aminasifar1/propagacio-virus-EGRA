import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys


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

    # --- Crea buffers ---
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
    def __init__(self, aspect):
        # 📸 Vista més llunyana i centrada
        self.position = glm.vec3(20, 25, 25)
        self.target = glm.vec3(0, 0, 0)
        self.up = glm.vec3(0, 1, 0)
        self.m_view = glm.lookAt(self.position, self.target, self.up)
        self.m_proj = glm.perspective(glm.radians(35), aspect, 0.1, 200)


class Object3D:
    def __init__(self, ctx, obj_path, camera):
        self.ctx = ctx
        tri_data, normals, line_data = load_obj(obj_path)

        # --- VAOs ---
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

                    // Il·luminació bàsica Phong
                    vec3 view_dir = normalize(view_pos - v_frag_pos);
                    vec3 reflect_dir = reflect(-light_dir, norm);
                    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);

                    vec3 ambient = 0.2 * color;
                    vec3 diffuse = diff * color;
                    vec3 specular = spec * vec3(1.0);

                    fragColor = vec4(ambient + diffuse + specular, 1.0);
                }
            ''',
        )

    def update_uniforms(self):
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.shader['m_model'].write(self.m_model)
        self.shader['light_pos'].value = (10.0, 10.0, 10.0)
        self.shader['view_pos'].value = tuple(self.camera.position)

    def render(self):
        self.update_uniforms()

        # ✨ Cares amb llum
        self.vao_tri.render(mode=mgl.TRIANGLES)

        # 🔲 Línies negres (wireframe)
        self.ctx.line_width = 1.0
        self.shader['m_model'].write(self.m_model)
        self.shader['m_proj'].write(self.camera.m_proj)
        self.shader['m_view'].write(self.camera.m_view)
        self.vao_line.render(mode=mgl.LINES)


class ViewerApp:
    def __init__(self, obj_path, win_size=(900, 900)):
        pg.init()
        pg.display.set_caption("3D Object Viewer with Lighting")
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.aspect_ratio = win_size[0] / win_size[1]
        self.camera = Camera(self.aspect_ratio)
        self.object = Object3D(self.ctx, obj_path, self.camera)

    def run(self):
        clock = pg.time.Clock()
        angle = 0
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    pg.quit()
                    sys.exit()

            # 🔄 Rotació
            angle += 0.3
            self.object.m_model = glm.rotate(glm.mat4(), glm.radians(angle), glm.vec3(0, 1, 0))

            self.ctx.clear(0.07, 0.07, 0.09)
            self.object.render()
            pg.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    obj_path = "surface.obj"  # Aquí el nom de l'objecte que vulguem visualitzar
    app = ViewerApp(obj_path)
    app.run()

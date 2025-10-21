import pygame as pg
import moderngl as mgl
import glm
import sys
import numpy as np

# ---------------------- CAMERA ----------------------
class Camera:
    def __init__(self, app):
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]
        self.position = glm.vec3(4, 3, 3)
        self.up = glm.vec3(0, 1, 0)
        self.m_view = self.get_view_matrix()
        self.m_proj = self.get_projection_matrix()

    def get_view_matrix(self):
        return glm.lookAt(self.position, glm.vec3(0), self.up)

    def get_projection_matrix(self):
        return glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100)


# ---------------------- STUDENT (ESFERA CON WIREFRAME) ----------------------
class Student:
    def __init__(self, app, radius=1.0, subdivisions=2):
        self.app = app
        self.ctx = app.ctx
        self.radius = radius
        self.subdivisions = subdivisions
        self.shader_program = self.get_shader_program()
        self.shader_program_wire = self.get_shader_wire()
        # generar geometría
        self.vertices, self.faces = self.generate_faces()
        self.vao = self.get_vao()
        self.vao_wire = self.get_wireframe_vao()
        self.m_model = glm.mat4()

    # ------------------ shaders ------------------
    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location=0) in vec3 in_position;
                layout(location=1) in vec3 in_normal;
                out vec3 v_normal;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                    v_normal = mat3(transpose(inverse(m_model))) * in_normal;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_normal;
                out vec4 fragColor;
                void main() {
                    vec3 light_dir = normalize(vec3(1.0,1.0,1.0));
                    float diff = max(dot(normalize(v_normal), light_dir), 0.0);
                    vec3 color = vec3(1.0,0.8,0.1)*diff + vec3(0.1);
                    fragColor = vec4(color, 1.0);
                }
            '''
        )

    def get_shader_wire(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location=0) in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main(){
                    gl_Position = m_proj*m_view*m_model*vec4(in_position,1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                void main(){ fragColor = vec4(0.2,0.8,1.0,1.0); }
            '''
        )

    # ------------------ generador de octaedro y subdivisión ------------------
    def generate_faces(self):
        # Vértices iniciales de octaedro
        v = [
            np.array([1,0,0], dtype='f4'),
            np.array([-1,0,0], dtype='f4'),
            np.array([0,1,0], dtype='f4'),
            np.array([0,-1,0], dtype='f4'),
            np.array([0,0,1], dtype='f4'),
            np.array([0,0,-1], dtype='f4')
        ]
        faces = [
            (v[0],v[4],v[2]), (v[2],v[4],v[1]),
            (v[1],v[4],v[3]), (v[3],v[4],v[0]),
            (v[0],v[2],v[5]), (v[2],v[1],v[5]),
            (v[1],v[3],v[5]), (v[3],v[0],v[5])
        ]
        # Subdivisión
        for _ in range(self.subdivisions):
            faces_sub = []
            for tri in faces:
                a,b,c = tri
                ab = self.normalize((a+b)/2)
                bc = self.normalize((b+c)/2)
                ca = self.normalize((c+a)/2)
                faces_sub.extend([
                    (a, ab, ca),
                    (ab, b, bc),
                    (ca, bc, c),
                    (ab, bc, ca)
                ])
            faces = faces_sub

        vertices = []
        for tri in faces:
            for vert in tri:
                pos = vert*self.radius
                norm = self.normalize(vert)
                vertices.append((pos[0], pos[1], pos[2], norm[0], norm[1], norm[2]))
        return np.array(vertices, dtype='f4'), faces

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm==0: return v
        return v/norm

    # ------------------ VAO para esfera ------------------
    def get_vao(self):
        return self.ctx.vertex_array(self.shader_program, [(self.ctx.buffer(self.vertices), '3f 3f', 'in_position','in_normal')])

    # ------------------ VAO para wireframe ------------------
    def get_wireframe_vao(self):
        lines = []
        for tri in self.faces:
            a,b,c = tri
            lines.extend([a,b,b,c,c,a])
        line_data = np.array([v for vert in lines for v in vert], dtype='f4')
        vbo_lines = self.ctx.buffer(line_data)
        return self.ctx.vertex_array(self.shader_program_wire, [(vbo_lines,'3f','in_position')])

    # ------------------ render ------------------
    def render(self):
        self.ctx.enable(mgl.DEPTH_TEST)
        # Wireframe
        self.shader_program_wire['m_proj'].write(self.app.camera.m_proj)
        self.shader_program_wire['m_view'].write(self.app.camera.m_view)
        self.shader_program_wire['m_model'].write(self.m_model)
        self.vao_wire.render(mgl.LINES)
        # Esfera
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(self.m_model)
        self.vao.render(mgl.TRIANGLES)

    def destroy(self):
        self.vao.release()
        self.vao_wire.release()
        self.shader_program.release()
        self.shader_program_wire.release()


# ---------------------- ENGINE ----------------------
class GraphicsEngine:
    def __init__(self, win_size=(800,800)):
        pg.init()
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION,3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION,3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL|pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.camera = Camera(self)
        self.scene = Student(self, radius=1.0, subdivisions=2)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type==pg.KEYDOWN and event.key==pg.K_ESCAPE):
                self.scene.destroy()
                pg.quit()
                sys.exit()

    def render(self):
        self.ctx.clear(0.0,0.0,0.0)
        self.scene.render()
        pg.display.flip()

    def run(self):
        while True:
            self.check_events()
            self.render()


if __name__=='__main__':
    app = GraphicsEngine()
    app.run()

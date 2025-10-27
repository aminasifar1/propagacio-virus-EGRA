import pygame as pg
import moderngl as mgl
import glm
import sys
import numpy as np

# ---------------------- CÀMERA ----------------------
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


# ---------------------- PRIMITIVES ----------------------
class Primitives:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.shader_program = self.get_shader_program()
        self.vertex_data = self.get_vertex_data()
        self.vbo = self.ctx.buffer(self.vertex_data)
        self.vao = self.get_vao()

    def get_vertex_data(self):
        # 3 línies per eixos XYZ
        vertices = [
            0, 0, 0, 1, 0, 0,   # X
            2, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,   # Y
            0, 2, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1,   # Z
            0, 0, 2, 0, 0, 1
        ]
        return np.array(vertices, dtype='f4')

    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location=0) in vec3 in_position;
                layout(location=1) in vec3 in_color;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                out vec3 v_color;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(v_color, 1.0);
                }
            '''
        )

    def get_vao(self):
        return self.ctx.vertex_array(
            self.shader_program,
            [
                (self.vbo, '3f 3f', 'in_position', 'in_color')
            ]
        )

    def render(self):
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(glm.mat4())
        self.vao.render(mgl.LINES)

    def destroy(self):
        self.vbo.release()
        self.shader_program.release()
        self.vao.release()


# ---------------------- PÍLDORA ----------------------
class Pildora:
    def __init__(self, app, radius=0.5, length=1.0, sectors=64, stacks=32, capas=4):
        self.app = app
        self.ctx = app.ctx
        self.radius = radius
        self.length = length
        self.sectors = sectors
        self.stacks = stacks
        self.capas = capas
        self.shader_program = self.get_shader_program()

        self.vertex_data, self.layer_ids = self.get_pill_vertices_with_layers()
        self.vbo = self.ctx.buffer(self.vertex_data)
        self.layer_vbo = self.ctx.buffer(self.layer_ids)
        self.vao = self.get_vao()

        self.m_model = glm.mat4()

    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location=0) in vec3 in_position;
                layout(location=1) in float in_layer;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                out float v_layer;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                    v_layer = in_layer;
                }
            ''',
            fragment_shader='''
                #version 330
                in float v_layer;
                out vec4 fragColor;
                uniform vec3 u_color;
                vec3 layerColor(float layer) {
                    if (layer < 1.0) return vec3(0.8, 0.3, 0.4);
                    if (layer < 2.0) return vec3(0.3, 0.8, 0.6);
                    if (layer < 3.0) return vec3(0.5, 0.4, 0.9);
                    return vec3(0.9, 0.9, 0.2);
                }
                void main() {
                    vec3 base = layerColor(v_layer);
                    fragColor = vec4(base * u_color, 1.0);
                }
            '''
        )

    def get_pill_vertices_with_layers(self):
        vertices = []
        layer_ids = []
        pi = np.pi
        sector_step = 2 * pi / self.sectors
        stack_step = pi / self.stacks

        for i in range(self.stacks + 1):
            theta = i * stack_step
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(self.sectors + 1):
                phi = j * sector_step
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                x = self.radius * sin_theta * cos_phi
                y = self.radius * cos_theta
                z = self.radius * sin_theta * sin_phi

                if abs(y) < self.radius:
                    y += np.sign(y) * self.length / 2.0
                else:
                    y += (self.length / 2.0) * np.sign(y)

                vertices.append((x, y, z))
                layer = (y / ((self.radius + self.length / 2) * 2)) * self.capas
                layer_ids.append((layer,))

        # triangles
        indices = []
        for i in range(self.stacks):
            k1 = i * (self.sectors + 1)
            k2 = k1 + self.sectors + 1
            for j in range(self.sectors):
                indices.append((k1 + j, k2 + j, k1 + j + 1))
                indices.append((k1 + j + 1, k2 + j, k2 + j + 1))

        vertex_array = np.array([vertices[idx] for tri in indices for idx in tri], dtype='f4')
        layer_array = np.array([layer_ids[idx][0] for tri in indices for idx in tri], dtype='f4').reshape(-1, 1)
        return vertex_array, layer_array

    def get_vao(self):
        return self.ctx.vertex_array(
            self.shader_program,
            [
                (self.vbo, '3f', 'in_position'),
                (self.layer_vbo, '1f', 'in_layer'),
            ],
        )

    def render(self):
        self.ctx.enable(mgl.DEPTH_TEST)
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(self.m_model)

        # superfície
        self.shader_program['u_color'].value = (1.0, 1.0, 1.0)
        self.vao.render(mgl.TRIANGLES)
        # wireframe
        self.shader_program['u_color'].value = (0.1, 0.9, 1.0)
        self.vao.render(mgl.LINES)
        # punts
        self.shader_program['u_color'].value = (1.0, 1.0, 0.0)
        self.ctx.point_size = 5.0
        self.vao.render(mgl.POINTS)

    def destroy(self):
        self.vbo.release()
        self.layer_vbo.release()
        self.shader_program.release()
        self.vao.release()


# ---------------------- MOTOR ----------------------
class GraphicsEngine:
    def __init__(self, win_size=(900, 900)):
        pg.init()
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.camera = Camera(self)
        self.primitives = Primitives(self)
        self.scene = Pildora(self, radius=0.4, length=1.0, sectors=48, stacks=24, capas=4)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.scene.destroy()
                self.primitives.destroy()
                pg.quit()
                sys.exit()

    def render(self):
        self.ctx.clear(0.05, 0.05, 0.08)
        self.primitives.render()
        self.scene.render()
        pg.display.flip()

    def run(self):
        while True:
            self.check_events()
            self.render()


if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()

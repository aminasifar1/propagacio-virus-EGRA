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


# ---------------------- PÍLDORA ----------------------
class Pildora:
    def __init__(self, app, radius=0.5, length=1.0, sectors=48, stacks=24):
        self.app = app
        self.ctx = app.ctx
        self.radius = radius
        self.length = length
        self.sectors = sectors
        self.stacks = stacks
        self.shader_program = self.get_shader_program()

        self.vertex_data = self.get_vertices()
        self.vbo = self.ctx.buffer(self.vertex_data)
        self.vao = self.get_vao()
        self.m_model = glm.mat4()

    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location=0) in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                uniform vec3 u_color;
                void main() {
                    fragColor = vec4(u_color, 1.0);
                }
            '''
        )

    def get_vertices(self):
        points = []
        pi = np.pi
        sector_step = 2 * pi / self.sectors
        stack_step = pi / self.stacks

        # generar puntos de la píldora
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

                # alargar la forma
                if abs(y) < self.radius:
                    y += np.sign(y) * self.length / 2.0
                else:
                    y += (self.length / 2.0) * np.sign(y)

                points.append((x, y, z))

        # crear triángulos
        triangles = []
        for i in range(self.stacks):
            k1 = i * (self.sectors + 1)
            k2 = k1 + self.sectors + 1
            for j in range(self.sectors):
                triangles.append(points[k1 + j])
                triangles.append(points[k2 + j])
                triangles.append(points[k1 + j + 1])

                triangles.append(points[k1 + j + 1])
                triangles.append(points[k2 + j])
                triangles.append(points[k2 + j + 1])

        return np.array(triangles, dtype='f4')

    def get_vao(self):
        return self.ctx.vertex_array(
            self.shader_program,
            [(self.vbo, '3f', 'in_position')]
        )

    def render(self):
        self.ctx.enable(mgl.DEPTH_TEST)
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(self.m_model)

        # color amarillo
        self.shader_program['u_color'].value = (1.0, 1.0, 0.0)
        self.vao.render(mgl.TRIANGLES)

    def destroy(self):
        self.vbo.release()
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
        self.scene = Pildora(self, radius=0.4, length=1.0, sectors=48, stacks=24)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.scene.destroy()
                pg.quit()
                sys.exit()

    def render(self):
        self.ctx.clear(0.05, 0.05, 0.08)
        self.scene.render()
        pg.display.flip()

    def run(self):
        while True:
            self.check_events()
            self.render()


if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()

import pygame as pg
import moderngl as mgl
import glm
import sys
import numpy as np
import slimgui
#import glfw

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


# ---------------------- AXIS ----------------------
class Axis:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.vbo = self.get_vbo()
        self.shader_program = self.get_shader_program()
        self.vao = self.get_vao()
        self.m_model = self.get_model_matrix()
        self.with_axis = True
        self.info = "Axis : ON"

    def get_model_matrix(self):
        return glm.mat4()

    def render(self):
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(self.m_model)
        self.vao.render(mgl.LINES)

    def destroy(self):
        self.vbo.release()
        self.shader_program.release()
        self.vao.release()

    def get_vao(self):
        return self.ctx.vertex_array(self.shader_program, [(self.vbo, '3f 3f', 'in_position', 'in_color')])

    def get_vertex_data(self):
        vertices = [
            (0, 0, 0, 0, 0, 0),
            (0, 0, 2, 0, 0, 1),
            (0, 2, 0, 0, 1, 0),
            (2, 0, 0, 1, 0, 0)
        ]
        indices = [(0, 1), (0, 2), (0, 3)]
        data = [vertices[ind] for line in indices for ind in line]
        return np.array(data, dtype='f4')

    def get_vbo(self):
        return self.ctx.buffer(self.get_vertex_data())

    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout (location = 0) in vec3 in_position;
                in vec3 in_color;
                out vec3 v_color;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                layout (location = 0) out vec4 fragColor;
                void main() {
                    fragColor = vec4(v_color, 1.0);
                }
            '''
        )


# ---------------------- PRIMITIVES ----------------------
class Primitives:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.vbo = self.get_vbo()
        self.shader_program = self.get_shader_program()
        self.vao = self.get_vao()
        self.m_model = self.get_model_matrix()
        self.with_edges = True
        self.info = "Primitives : ON"

    def get_model_matrix(self):
        return glm.rotate(glm.mat4(), glm.radians(45), glm.vec3(0, 1, 0))

    def render(self):
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(self.m_model)

        self.ctx.point_size = 5
        self.shader_program['color'].value = (0.4, 0.6, 1)
        #self.vao.render(mgl.POINTS)

        self.ctx.enable(mgl.DEPTH_TEST)
        self.shader_program['color'].value = (1, 0, 1)
        self.vao.render(mgl.LINES)

    def destroy(self):
        self.vbo.release()
        self.shader_program.release()
        self.vao.release()

    def get_vao(self):
        return self.ctx.vertex_array(self.shader_program, [(self.vbo, '3f', 'in_position')])

    def get_vertex_data(self):
        vertices = [(0, 1, 0), (1, 0, 0), (0, 0, 1), (-1, 0, 0)]
        indices = [(1, 2, 3), (0, 1, 3), (0, 1, 2), (0, 2, 3)]
        data = [vertices[i] for tri in indices for i in tri]
        return np.array(data, dtype='f4')

    def get_vbo(self):
        return self.ctx.buffer(self.get_vertex_data())

    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout (location = 0) in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                layout (location = 0) out vec4 fragColor;
                uniform vec3 color;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            '''
        )


# ---------------------- STUDENT (ESFERA) ----------------------
class Student:
    def __init__(self, app, radius=0.1, sectors=2, stacks=3):
        self.app = app
        self.ctx = app.ctx
        self.radius = radius
        self.sectors = sectors
        self.stacks = stacks
        self.vbo = self.get_vbo()
        self.shader_program = self.get_shader_program()
        self.vao = self.get_vao()
        self.m_model = self.get_model_matrix()
        self.on_init()

    def get_model_matrix(self):
        return glm.rotate(glm.mat4(), glm.radians(45), glm.vec3(0, 1, 0))

    def on_init(self):
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(self.m_model)

    def render(self):
        self.ctx.enable(mgl.DEPTH_TEST)
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(self.m_model)
        #self.vao.render(mgl.TRIANGLES)
        self.vao.render(mgl.LINES)

    def destroy(self):
        self.vbo.release()
        self.shader_program.release()
        self.vao.release()

    def get_vao(self):
        return self.ctx.vertex_array(self.shader_program, [(self.vbo, '3f', 'in_position')])

    def get_vbo(self):
        vertex_data = self.get_sphere_vertices()
        return self.ctx.buffer(vertex_data)

    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout (location = 0) in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                layout (location = 0) out vec4 fragColor;
                void main() {
                    vec3 color = vec3(1.0, 0.8, 0.1); // groc
                    fragColor = vec4(color, 1.0);
                }
            '''
        )

    def get_sphere_vertices(self):
        vertices = []
        pi = np.pi
        sector_step = 2 * pi / self.sectors
        stack_step = pi / self.stacks

        # Generar vèrtexs
        for i in range(self.stacks + 1):
            stack_angle = pi / 2 - i * stack_step
            xy = self.radius * np.cos(stack_angle)
            z = self.radius * np.sin(stack_angle)

            for j in range(self.sectors + 1):
                sector_angle = j * sector_step
                x = xy * np.cos(sector_angle)
                y = xy * np.sin(sector_angle)
                vertices.append((x, y, z))

        # Crear índexs de triangles
        indices = []
        for i in range(self.stacks):
            k1 = i * (self.sectors + 1)
            k2 = k1 + self.sectors + 1
            for j in range(self.sectors):
                if i != 0:
                    indices.append((k1 + j, k2 + j, k1 + j + 1))
                if i != (self.stacks - 1):
                    indices.append((k1 + j + 1, k2 + j, k2 + j + 1))

        data = np.array([vertices[idx] for tri in indices for idx in tri], dtype='f4')
        return data


# ---------------------- ENGINE ----------------------
class GraphicsEngine:
    def __init__(self, win_size=(800, 800)):
        pg.init()
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.camera = Camera(self)
        self.axis = Axis(self)
        self.primitives = Primitives(self)
        self.scene = Student(self, radius=1.0, sectors=72, stacks=36)

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.scene.destroy()
                pg.quit()
                sys.exit()

    def render(self):
        self.ctx.clear(color=(0, 0, 0))
        self.axis.render()
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

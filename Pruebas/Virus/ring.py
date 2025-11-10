# aro_persona_axis_scaled_prima.py

import pygame as pg
import moderngl as mgl
import glm
import sys
import numpy as np

class Camera:
    def __init__(self, app):
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]
        self.position = glm.vec3(3, 3, 3)
        self.up = glm.vec3(0, 1, 0)
        self.m_view = self.get_view_matrix()
        self.m_proj = self.get_projection_matrix()

    def get_view_matrix(self):
        return glm.lookAt(self.position, glm.vec3(0), self.up)

    def get_projection_matrix(self):
        return glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100)

class Object:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.vbo = self.get_vbo()
        self.shader_program = self.get_shader_program()
        self.vao = self.get_vao()
        self.m_model = self.get_model_matrix()
        self.on_init()

    def get_model_matrix(self):
        return glm.mat4()

    def on_init(self):
        self.shader_program['m_proj'].write(self.app.camera.m_proj)
        self.shader_program['m_view'].write(self.app.camera.m_view)
        self.shader_program['m_model'].write(self.m_model)

    def render(self):
        self.vao.render()

    def destroy(self):
        self.vbo.release()
        self.shader_program.release()
        self.vao.release()

    def get_vao(self):
        vao = self.ctx.vertex_array(self.shader_program, [(self.vbo, '3f', 'in_position')])
        return vao

    def get_vbo(self):
        vertex_data = self.get_vertex_data()
        vbo = self.ctx.buffer(vertex_data)
        return vbo

# -------------------------
# Persona més primeta
# -------------------------
class Person(Object):
    def get_vertex_data(self):
        vertices = [
            (0,0,0),(1,0,0),(1,1,0),(0,1,0),
            (0,0,1),(1,0,1),(1,1,1),(0,1,1)
        ]
        indices = [
            (0,2,1),(0,3,2),(4,5,6),(4,6,7),
            (0,1,4),(1,4,5),(2,3,7),(2,7,6),
            (1,2,6),(1,6,5),(0,4,7),(0,7,3)
        ]
        data = [vertices[i] for tri in indices for i in tri]
        return np.array(data, dtype='f4')

    def get_model_matrix(self):
        # Escala X i Z més petit, Y mantenim 1.0
        m = glm.mat4(1.0)
        return glm.scale(m, glm.vec3(0.5, 1.0, 0.5))

    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location = 0) in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main(){
                    gl_Position = m_proj * m_view * m_model * vec4(in_position,1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                void main(){ fragColor = vec4(0.6,0.6,0.9,1.0); }
            '''
        )

# -------------------------
# Aro (mida per defecte)
# -------------------------
class VirusRing(Object):
    def __init__(self, app, segments=128, radius=1.0, height=0.0, color=(1.0,0.3,0.0,1.0), thickness=3.0):
        self.segments = segments
        self.radius = radius
        self.height = height
        self.color = color
        self.thickness = thickness
        super().__init__(app)

    def get_vertex_data(self):
        vertices = []
        for i in range(self.segments):
            theta = 2*np.pi*i/self.segments
            x = self.radius*np.cos(theta)
            y = self.height
            z = self.radius*np.sin(theta)
            vertices.append((x,y,z))
        return np.array(vertices,dtype='f4')

    def get_shader_program(self):
        return self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location = 0) in vec3 in_position;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main(){ gl_Position = m_proj * m_view * m_model * vec4(in_position,1.0); }
            ''',
            fragment_shader='''
                #version 330
                out vec4 fragColor;
                uniform vec4 u_color;
                void main(){ fragColor = u_color; }
            '''
        )

    def on_init(self):
        super().on_init()
        self.shader_program['u_color'].value = self.color

    def render(self):
        self.ctx.line_width = self.thickness
        self.vao.render(mode=mgl.LINE_LOOP)
        self.ctx.line_width = 1.0  # Restableix el gruix per altres objectes


# -------------------------
# Eixos (mida per defecte)
# -------------------------
class Axis:
    def __init__(self, app, length=2.0):
        self.app = app
        self.ctx = app.ctx
        self.length = length

        # posició i color en un sol array (x,y,z,r,g,b)
        self.vbo = self.ctx.buffer(np.array([
            # Eix X (vermell)
            0, 0, 0, 1, 0, 0,
            self.length, 0, 0, 1, 0, 0,
            # Eix Y (verd)
            0, 0, 0, 0, 1, 0,
            0, self.length, 0, 0, 1, 0,
            # Eix Z (blau)
            0, 0, 0, 0, 0, 1,
            0, 0, self.length, 0, 0, 1,
        ], dtype='f4'))

        self.shader = self.ctx.program(
            vertex_shader='''
                #version 330
                layout(location = 0) in vec3 in_position;
                layout(location = 1) in vec3 in_color;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                out vec3 v_color;
                void main(){
                    gl_Position = m_proj * m_view * m_model * vec4(in_position,1.0);
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 fragColor;
                void main(){
                    fragColor = vec4(v_color,1.0);
                }
            '''
        )

        # VAO amb posició i color
        self.vao = self.ctx.vertex_array(
            self.shader,
            [(self.vbo, '3f 3f', 'in_position', 'in_color')]
        )

    def render(self):
        self.shader['m_proj'].write(self.app.camera.m_proj)
        self.shader['m_view'].write(self.app.camera.m_view)
        self.shader['m_model'].write(glm.mat4(1.0))
        self.vao.render(mode=mgl.LINES)

# -------------------------
# Motor gràfic
# -------------------------
class GraphicsEngine:
    def __init__(self, win_size=(600,600)):
        pg.init()
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION,3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION,3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.WIN_SIZE,flags=pg.OPENGL|pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.camera = Camera(self)

        # Escena per defecte
        self.axis = Axis(self, length=2.0)
        self.person = Person(self)
        self.ring = VirusRing(self, radius=0.71, height=0.25, thickness=5.0)

    def check_events(self):
        for event in pg.event.get():
            if event.type==pg.QUIT or (event.type==pg.KEYDOWN and event.key==pg.K_ESCAPE):
                self.person.destroy()
                self.ring.destroy()
                pg.quit()
                sys.exit()

    def render(self):
        self.ctx.clear(color=(0,0,0))
        self.axis.render()
        self.person.render()
        self.ring.render()
        pg.display.flip()

    def run(self):
        while True:
            self.check_events()
            self.render()

if __name__=='__main__':
    app = GraphicsEngine()
    app.run()

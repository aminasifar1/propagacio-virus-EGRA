import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import random

from classgrid import ClassGrid
from camera import Camera
from object3d import Object3D
from infectionbar import InfectionBar
from pathfinding import PathfindingSystem
from person import Person
from puff import PuffSystem
from waypoint import WaypointVisualizer
from marker import Marker
from virus import Virus


class ViewerApp:
    def __init__(self, obj_path, win_size=(1536, 864)):
        pg.init()
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        self.screen = pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.front_face = 'ccw'
        self.aspect_ratio = win_size[0] / win_size[1]
        self.camera = Camera(self)
        self.puff_system = PuffSystem(self.ctx, self.camera)

        self.infection_bar = InfectionBar(win_size[0], win_size[1])
        self.ui_surface = pg.Surface(self.WIN_SIZE, pg.SRCALPHA)

        self.clock = pg.time.Clock()
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.delta_time = 0.016
        self.tick_duration = 0.2
        self.tick_timer = 0.0
        self.infection_probability = 1

        self.virus = Virus(self, self.tick_duration, self.tick_timer, self.infection_probability, 1)

        self.object = Object3D(self.ctx, obj_path, self.camera)
        self.object.app = self
        self.show_grid = False

        min_coords, max_coords = self.object.bounding_box
        ground_y = min_coords.y + 0.2

        # --- PATHFINDING NOMÉS DINS DE LES AULES ---
        self.pathfinding = PathfindingSystem()

        self.marker = Marker(self.ctx, self.camera)

        # ---- AULES EXACTES AMB SUBCUADRADOS DE 1.6 ----
        self.class_grids = []

        aules_data = [
            # Aula gran
            {
                "min": glm.vec3(-8.7, ground_y, 15.40),
                "max": glm.vec3(-0.5, ground_y, 29.5),
                "spacing": 1.6
            },
            # Aula petita
            {
                "min": glm.vec3(-8.7, ground_y, 30.40),
                "max": glm.vec3(-0.5, ground_y, 38.90),
                "spacing": 1.6
            }
        ]

        for aula in aules_data:
            grid = ClassGrid(
                self.ctx, self.camera,
                aula["min"], aula["max"],
                spacing=aula["spacing"]
            )
            self.class_grids.append(grid)

        # 🆕 Crea waypoints NOMÉS dins de les aules
        self.build_waypoints_from_class_grids()

        # Visualitzador de waypoints (el "grid verd")
        self.waypoint_visualizer = WaypointVisualizer(self.ctx, self.camera)
        self.waypoint_visualizer.build_from_system(self.pathfinding)

        self.show_class_grids = True

        # ---- PERSONES ----
        self.people = []
        try:
            num_people = 50
            for i in range(num_people):
                is_infected = (i == 0)
                person = Person(
                    self.ctx, self.camera,
                    "Models/visualizer/person_1.obj",
                    self.pathfinding,
                    ground_y,
                    is_infected
                )
                self.people.append(person)

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
            print("No s'ha trobat person_1.obj.")
            self.people = []
            self.person_vao_tri = None
            self.person_vao_line = None


    # 🆕 --- CREA WAYPOINTS NOMÉS DINS DE LES AULES ---
    def build_waypoints_from_class_grids(self):
        """Crea waypoints dins de cada aula segons el seu grid."""
        ground_y = self.object.bounding_box[0].y + 0.2
        spacing = 1.6

        for grid in self.class_grids:
            min_pos = grid.min_corner
            max_pos = grid.max_corner
            x_vals = np.arange(min_pos.x, max_pos.x + 1e-3, spacing)
            z_vals = np.arange(min_pos.z, max_pos.z + 1e-3, spacing)

            wp_grid = {}
            for x in x_vals:
                for z in z_vals:
                    pos = glm.vec3(x, ground_y, z)
                    wp = self.pathfinding.add_waypoint(pos)
                    wp_grid[(x, z)] = wp

            # Connectem veïns (com abans)
            for x, z in wp_grid.keys():
                current = wp_grid[(x, z)]
                for dx, dz in [(spacing, 0), (-spacing, 0), (0, spacing), (0, -spacing)]:
                    neighbor = wp_grid.get((x + dx, z + dz))
                    if neighbor:
                        self.pathfinding.connect(current, neighbor)


    # --- render_ui_to_texture ---
    def render_ui_to_texture(self):
        num_infected = sum(1 for p in self.people if p.ring)
        total_people = len(self.people)
        self.ui_surface.fill((0, 0, 0, 0))
        self.infection_bar.render(self.ui_surface, num_infected, total_people)
        texture_data = pg.image.tostring(self.ui_surface, 'RGBA', True)
        if not hasattr(self, 'ui_texture'):
            self.ui_texture = self.ctx.texture(self.WIN_SIZE, 4, texture_data)
            self.ui_texture.filter = (mgl.LINEAR, mgl.LINEAR)
        else:
            self.ui_texture.write(texture_data)

        # Crear quad para renderizar la textura
        if not hasattr(self, 'ui_vao'):
            vertices = np.array([
                -1, -1, 0, 0,
                 1, -1, 1, 0,
                -1,  1, 0, 1,
                 1,  1, 1, 1,], dtype='f4') 
            self.ui_vbo = self.ctx.buffer(vertices)
            self.ui_shader = self.ctx.program(
                vertex_shader='''
                    #version 330
                    in vec2 in_position;
                    in vec2 in_texcoord;
                    out vec2 v_texcoord;
                    void main() {
                        v_texcoord = in_texcoord;
                        gl_Position = vec4(in_position, 0.0, 1.0);
                    }
                ''',
                fragment_shader='''
                    #version 330
                    in vec2 v_texcoord;
                    uniform sampler2D ui_texture;
                    out vec4 fragColor;
                    void main() {
                        fragColor = texture(ui_texture, v_texcoord);
                    }
                ''')
            self.ui_vao = self.ctx.vertex_array(
                self.ui_shader,
                [(self.ui_vbo, '2f 2f', 'in_position', 'in_texcoord')])

        # Renderizar UI
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        
        self.ui_texture.use(0)
        self.ui_shader['ui_texture'] = 0
        self.ui_vao.render(mode=mgl.TRIANGLE_STRIP)
        
        self.ctx.disable(mgl.BLEND)
        self.ctx.enable(mgl.DEPTH_TEST)

    # --- run ---
    def run(self):
        last_time = time.time()
        while True:
            curr_time = time.time()
            self.delta_time = curr_time - last_time
            if self.delta_time == 0:
                self.delta_time = 1e-6
            last_time = curr_time

            for e in pg.event.get():
                self.camera.handle_mouse(e)
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    pg.quit()
                    sys.exit()
                if e.type == pg.KEYDOWN:
                    if e.key == pg.K_g:
                        self.show_grid = not self.show_grid
                    if e.key == pg.K_h:
                        self.show_class_grids = not self.show_class_grids

            self.camera.move(self.delta_time)

            for p in self.people:
                p.update(self.delta_time)

            self.puff_system.update(self.delta_time)

            self.tick_timer += self.delta_time
            if self.tick_timer >= self.tick_duration:
                self.tick_timer -= self.tick_duration
                self.virus.check_infections(self.people)

            self.camera.update_matrices()
            self.ctx.clear(0.07, 0.07, 0.09)
            self.object.render()

            if self.show_grid:
                self.waypoint_visualizer.render()

            if self.show_class_grids:
                for grid in self.class_grids:
                    grid.render()

            if self.people:
                light_pos = self.object.update_light_position()
                for p in self.people:
                    p.render(self.object.shader, self.person_vao_tri, self.person_vao_line, light_pos)

            # --- Càlcul FPS ---
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time
                pg.display.set_caption(f"3D Viewer - FPS: {self.fps:.1f} - WASD moverte, TAB soltar ratón")

            self.marker.handle_input(pg.key.get_pressed())
            self.marker.print_position()
            self.marker.render()

            self.puff_system.render()
            self.render_ui_to_texture()
            pg.display.flip()


if __name__ == "__main__":
    obj_path = "Models/visualizer/OBJ.obj"
    try:
        app = ViewerApp(obj_path)
        app.run()
    except FileNotFoundError:
        print(f"No s'ha trobat el fitxer {obj_path}.")
        sys.exit()
    except Exception as e:
        print(f"S'ha produït un error: {e}")
        pg.quit()
        sys.exit()

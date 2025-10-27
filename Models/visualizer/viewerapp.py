import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random

class ViewerApp:
    def __init__(self, obj_path, win_size=(1536, 864)):
        pg.init()
        pg.display.set_caption("3D Viewer - WASD para moverte, TAB para soltar ratón")
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        # IMPORTANTE: Cambiamos aquí para tener acceso a pygame surface
        self.screen = pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.front_face = 'ccw'
        self.aspect_ratio = win_size[0] / win_size[1]
        self.camera = Camera(self)
        self.puff_system = PuffSystem(self.ctx, self.camera)

        # Barra de infección
        self.infection_bar = InfectionBar(win_size[0], win_size[1])

        # Para renderizar la UI de pygame sobre OpenGL
        self.ui_surface = pg.Surface(self.WIN_SIZE, pg.SRCALPHA)

        # ...existing code...
        self.clock = pg.time.Clock()
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.delta_time = 0.016

        self.tick_duration = 0.2
        self.tick_timer = 0.0
        self.infection_probability = 1

        self.object = Object3D(self.ctx, obj_path, self.camera)
        self.object.app = self

        self.show_grid = False

        min_coords, max_coords = self.object.bounding_box
        print(f"Escenari carregat. Bounding Box:")
        print(f"  MIN: {min_coords}")
        print(f"  MAX: {max_coords}")

        self.pathfinding = PathfindingSystem()
        self.setup_waypoints(self.object.bounding_box)

        self.waypoint_visualizer = WaypointVisualizer(self.ctx, self.camera)
        self.waypoint_visualizer.build_from_system(self.pathfinding)

        self.people = []
        # número de persones
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
                    ground_y,
                    is_infected=is_infected
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
            print("Advertencia: No se encontró person_1.obj. No se crearan personas.")
            self.people = []
            self.person_vao_tri = None
            self.person_vao_line = None

    def setup_waypoints(self, bounding_box):
        """Configura una red de waypoints AUTOMÀTICAMENT basada en el Bounding Box."""

        min_coords, max_coords = bounding_box
        ground_y = min_coords.y + 0.2  # El terra

        wp_grid = {}
        spacing = 2.0  # Distància entre waypoints (pots ajustar-la)

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
                    for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                        neighbor = wp_grid.get((x + dx, z + dz))
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

    def render_ui_to_texture(self):
        """Renderiza la UI de pygame en una textura de OpenGL."""
        # Contar infectados
        num_infected = sum(1 for person in self.people if person.ring)
        total_people = len(self.people)

        # Limpiar superficie UI
        self.ui_surface.fill((0, 0, 0, 0))

        # Dibujar barra de infección
        self.infection_bar.render(self.ui_surface, num_infected, total_people)

        # Convertir superficie de pygame a textura OpenGL
        texture_data = pg.image.tostring(self.ui_surface, 'RGBA', True)

        # Crear/actualizar textura
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
                -1, 1, 0, 1,
                1, 1, 1, 1,
            ], dtype='f4')

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
                '''
            )

            self.ui_vao = self.ctx.vertex_array(
                self.ui_shader,
                [(self.ui_vbo, '2f 2f', 'in_position', 'in_texcoord')]
            )

        # Renderizar UI
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA

        self.ui_texture.use(0)
        self.ui_shader['ui_texture'] = 0
        self.ui_vao.render(mode=mgl.TRIANGLE_STRIP)

        self.ctx.disable(mgl.BLEND)
        self.ctx.enable(mgl.DEPTH_TEST)

    def run(self):
        last_frame_time = time.time()

        while True:
            current_frame_time = time.time()
            self.delta_time = current_frame_time - last_frame_time
            if self.delta_time == 0:
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

            self.camera.move(self.delta_time)

            for person in self.people:
                person.update(self.delta_time)

            self.puff_system.update(self.delta_time)

            self.tick_timer += self.delta_time

            if self.tick_timer >= self.tick_duration:
                self.tick_timer -= self.tick_duration
                self.check_infections()

            self.camera.update_matrices()

            # --- RENDERITZAT ---
            self.ctx.clear(0.07, 0.07, 0.09)

            self.object.render()

            if self.show_grid:
                self.waypoint_visualizer.render()

            if self.people and self.person_vao_tri:
                light_pos = self.object.update_light_position()
                for person in self.people:
                    person.render(self.object.shader, self.person_vao_tri,
                                  self.person_vao_line, light_pos)

            self.puff_system.render()

            # Renderizar UI (barra de infección)
            self.render_ui_to_texture()

            # --- Càlcul FPS ---
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time
                pg.display.set_caption(f"3D Viewer - FPS: {self.fps:.1f} - WASD moverte, TAB soltar ratón")

            pg.display.flip()
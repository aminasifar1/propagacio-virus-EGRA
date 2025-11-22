import os
import json
import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import random
from facultat import Sala, Clase, Pasillo
from escenario import Escenario
from camera import Camera
from person import Person
from marker import Marker
from virus import Virus
from infectionbar import InfectionBar

# =====================================================
#                    CARREGAR OBJ
# =====================================================
def load_obj(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[glm.vec3, glm.vec3]]:
    vertices = []
    faces = []
    edges = set()

    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    _, x, y, z = line.strip().split()
                    vertices.append((float(x), float(y), float(z)))
                elif line.startswith('f '):
                    parts = line.strip().split()[1:]
                    face_indices = [int(p.split('/')[0]) - 1 for p in parts]
                    # Triangles i edges
                    for i in range(1, len(face_indices) - 1):
                        faces.append((face_indices[0], face_indices[i], face_indices[i + 1]))
                    for i in range(len(face_indices)):
                        p1 = face_indices[i]
                        p2 = face_indices[(i + 1) % len(face_indices)]
                        edges.add(tuple(sorted((p1, p2))))
    except FileNotFoundError:
        print(f"Error: El fitxer '{path}' no s'ha trobat.")
        raise
    except Exception as e:
        print(f"Error processant el fitxer '{path}': {e}")
        raise

    if not vertices:
        return np.array([]), np.array([]), np.array([]), (glm.vec3(0), glm.vec3(0))

    np_vertices = np.array(vertices, dtype='f4')
    min_coords = np.min(np_vertices, axis=0)
    max_coords = np.max(np_vertices, axis=0)
    bounding_box = (glm.vec3(min_coords), glm.vec3(max_coords))

    # Normals per vertex
    vertex_normals = [np.zeros(3) for _ in range(len(vertices))]
    for face in faces:
        v0, v1, v2 = (np.array(vertices[i]) for i in face)
        face_normal = np.cross(v1 - v0, v2 - v0)
        for vertex_index in face:
            vertex_normals[vertex_index] += face_normal
    vertex_normals = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0, 1, 0]) for v in vertex_normals]

    # Dades per render de triangles
    tri_vertices_data = []
    normals_data = []
    for face in faces:
        for vertex_index in face:
            tri_vertices_data.extend(vertices[vertex_index])
            normals_data.extend(vertex_normals[vertex_index])

    # Dades per render de línies
    line_vertices_data = []
    for edge in edges:
        line_vertices_data.extend(vertices[edge[0]])
        line_vertices_data.extend(vertices[edge[1]])

    return (
        np.array(tri_vertices_data, dtype='f4'),
        np.array(normals_data, dtype='f4'),
        np.array(line_vertices_data, dtype='f4'),
        bounding_box
    )

# =====================================================
#                    MOTOR GRÀFIC
# =====================================================
class MotorGrafico:
    def __init__(self, scene_path, person_path, facultad, win_size=(640, 360)):
        pg.init()
        pg.display.set_caption("3D Viewer - WASD moverte, TAB soltar ratón")
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        self.screen = pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.front_face = 'ccw'
        self.aspect_ratio = win_size[0] / win_size[1]

        # Camara
        self.camera = Camera(self)

        # Mundo
        self.mundo = facultad

        # Marker
        self.marker = Marker(self.ctx, self.camera)

        # InfectionBar
        #self.infection_bar = InfectionBar(self.WIN_SIZE[0], self.WIN_SIZE[1])
        self.ui_surface = pg.Surface(self.WIN_SIZE, pg.SRCALPHA)

        # Clock y control de FPS
        self.clock = pg.time.Clock()
        self.last_time = time.time()
        self.delta_time = 0.016
        self.frame_count = 0
        self.fps = 0

        # Virus
        self.tick_duration = 0.2
        self.tick_timer = 0.0
        self.infection_probability = 0.2
        self.virus = Virus(self, self.tick_duration, self.tick_timer, self.infection_probability, 1, 0.9)

        # Escenari
        tri_data, normals, line_data, bounding_box = load_obj(scene_path)
        self.object = Escenario(self.ctx, self.camera, tri_data, normals, line_data, bounding_box)
        self.object.app = self

        # Persones
        self.p_tri_data, self.p_normals, self.p_line_data, bounding_box = load_obj(person_path)
        self.people = []
        self.simulando = False
        self.tiempo_persona = 0.0
        self.intervalo_spawn = 4.0
        self.max_people = 50

        # Creem la primera persona només per obtenir el VAO
        first_person = Person(self.ctx, self.camera, self.p_tri_data, self.p_normals, self.p_line_data, facultad, ['aula1'], 'pasillo')
        self.person_vao_tri = self.ctx.vertex_array(self.object.shader, [(first_person.tri_vbo, '3f', 'in_position'), (first_person.nrm_vbo, '3f', 'in_normal')])
        self.person_vao_line = self.ctx.vertex_array(self.object.shader, [(first_person.line_vbo, '3f', 'in_position')])

        min_coords, max_coords = self.object.bounding_box
        print(f"Escenari carregat. Bounding Box: MIN {min_coords}, MAX {max_coords}")

    # Crear persona
    def create_person(self, schedule=[], spawn='pasillo'):
        persona = Person(self.ctx, self.camera, self.p_tri_data, self.p_normals, self.p_line_data, self.mundo, schedule, spawn)
        self.people.append(persona)
        return persona

    # UI overlay
    def _render_ui_overlay(self):
        ui_string = pg.image.tostring(self.ui_surface, 'RGBA', True)
        if not hasattr(self, 'ui_texture'):
            self.ui_texture = self.ctx.texture(self.WIN_SIZE, 4, ui_string)
            self.ui_texture.filter = (mgl.LINEAR, mgl.LINEAR)
            ui_vertex_shader = """
            #version 330 core
            in vec2 in_position;
            in vec2 in_texcoord;
            out vec2 v_texcoord;
            void main() {
                gl_Position = vec4(in_position, 0.0, 1.0);
                v_texcoord = in_texcoord;
            }"""
            ui_fragment_shader = """
            #version 330 core
            uniform sampler2D ui_texture;
            in vec2 v_texcoord;
            out vec4 fragColor;
            void main() { fragColor = texture(ui_texture, v_texcoord); }"""
            self.ui_program = self.ctx.program(vertex_shader=ui_vertex_shader, fragment_shader=ui_fragment_shader)
            vertices = np.array([-1, -1, 0,0, 1,-1, 1,0, -1,1,0,1, 1,1,1,1], dtype='f4')
            self.ui_vbo = self.ctx.buffer(vertices)
            self.ui_vao = self.ctx.vertex_array(self.ui_program, [(self.ui_vbo, '2f 2f', 'in_position', 'in_texcoord')])
        else:
            self.ui_texture.write(ui_string)
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(mgl.DEPTH_TEST)
        self.ui_texture.use(0)
        self.ui_program['ui_texture'] = 0
        self.ui_vao.render(mgl.TRIANGLE_STRIP)
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.BLEND)

    # =====================================================
    #                  CICLO PRINCIPAL
    # =====================================================
    def start(self):
        print("[MOTOR] Inicializando recursos...")
        self.activo = True
        print("[MOTOR] Motor listo.")

    def run(self):
        if not self.activo:
            print("[MOTOR] Motor no inicializado.")
            return

        rooms = list(self.mundo.keys())
        if 'pasillo' in rooms: rooms.remove('pasillo')
        clean_rooms = {room:0 for room in rooms}

        print("[MOTOR] Iniciando ciclo principal...")
        last_frame_time = time.time()

        # ==========================
        # Funció de col·lisió AABB
        # ==========================
        def aabb_collision(pos1, bb1_half, pos2, bb2_half):
            """
            Retorna True si hi ha intersecció entre dos AABB centrats a pos1 i pos2.
            bb_half: semi-dimensions del bounding box.
            """
            return (abs(pos1.x - pos2.x) <= (bb1_half.x + bb2_half.x) and
                    abs(pos1.y - pos2.y) <= (bb1_half.y + bb2_half.y) and
                    abs(pos1.z - pos2.z) <= (bb1_half.z + bb2_half.z))

        while True:
            dt = self.clock.tick(60)/1000.0
            current_frame_time = time.time()
            self.delta_time = current_frame_time - last_frame_time
            if self.delta_time == 0: self.delta_time = 1e-6
            last_frame_time = current_frame_time
            keys = pg.key.get_pressed()

            # ==========================
            # Gestió d'events
            # ==========================
            for e in pg.event.get():
                self.marker.handle_input(keys)
                self.camera.handle_mouse(e)
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    pg.event.set_grab(False)
                    pg.quit()
                    sys.exit()
                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_p:
                        self.simulando = not self.simulando
                        print("▶️ Simulación:", "Iniciada" if self.simulando else "Pausada")
                    elif e.key == pg.K_r:
                        self.people.clear()
                        self.tiempo_persona = 0.0
                        print("🔄 Simulación reiniciada")

            # Actualitzar càmera
            self.camera.move(self.delta_time)
            self.camera.update_matrices()
            self.ctx.clear(0.07,0.07,0.09)

            # ==========================
            # Spawn de persones
            # ==========================
            if self.simulando:
                self.tiempo_persona += dt
                if self.tiempo_persona >= self.intervalo_spawn:
                    selection = random.choice(rooms)
                    p = self.create_person([selection])
                    if clean_rooms[selection] == 0:
                        self.virus.infectar(p)
                    clean_rooms[selection] += 1
                    self.tiempo_persona = 0.0

                # Tick virus
                self.tick_timer += self.delta_time
                if self.tick_timer >= self.tick_duration:
                    self.tick_timer -= self.tick_duration
                    self.virus.check_infections(self.mundo)

            # ==========================
            # Render escenari i marker
            # ==========================
            self.object.render()
            self.marker.render()
            light_pos = self.object.update_light_position()
            self.virus.render(light_pos)

            # ==========================
            # Actualitzar persones amb col·lisions
            # ==========================
            for p in self.people:
                if self.simulando:
                    old_pos = glm.vec3(p.position)  # guardem posició antiga
                    p.update(self.delta_time)
                    # Comprovem col·lisions amb altres persones
                    for other in self.people:
                        if other is p: continue
                        if aabb_collision(p.position, p.bb_half, other.position, other.bb_half):
                            # Revertim posició si hi ha col·lisió
                            p.position = old_pos
                            p.m_model = glm.translate(glm.mat4(1.0), old_pos)
                            break
                # Render de la persona
                p.render(self.object.shader, self.person_vao_tri, self.person_vao_line, light_pos)

            # ==========================
            # Render UI
            # ==========================
            self.ui_surface.fill((0,0,0,0))
            num_infected = sum(1 for p in self.people if hasattr(p,'ring') and p.ring is not None)
            total_people = len(self.people)
            if total_people>0:
                pass
                #self.infection_bar.render(self.ui_surface, num_infected, total_people)
            self._render_ui_overlay()
    
            # ==========================
            # FPS
            # ==========================
            self.frame_count += 1
            if time.time()-self.last_time>=1.0:
                self.fps = self.frame_count/(time.time()-self.last_time)
                self.frame_count=0
                self.last_time=time.time()
                pg.display.set_caption(f"3D Viewer - FPS: {self.fps:.1f} - WASD moverte, TAB soltar ratón")

            pg.display.flip()

# =====================================================
#                     MAIN
# =====================================================
if __name__ == "__main__":
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH,"DEMO","data","salas")
    SCENE_PATH = os.path.join(ROOT_PATH,"DEMO","Models","OBJ.obj")
    PERSON_PATH = os.path.join(ROOT_PATH,"DEMO","Models","person.obj")
    print(f"[MAIN] Ruta base: {ROOT_PATH}")

    # ==========================
    # Carregar totes les sales
    # ==========================
    facultad = {}
    for archivo in os.listdir(DATA_PATH):
        if not archivo.endswith(".json"): continue
        ruta = os.path.join(DATA_PATH, archivo)
        nombre = os.path.splitext(archivo)[0]
        with open(ruta,"r",encoding="utf-8") as f:
            data = json.load(f)
        tipo = data.get("tipo")
        if tipo=="clase": sala = Clase.from_json_struct(data)
        elif tipo=="pasillo": sala = Pasillo.from_json_struct(data)
        else: sala = Sala.from_json_struct(data)
        facultad[nombre] = sala
        print(f"[MAIN] Cargada sala '{nombre}' ({tipo}) con {len(sala.waypoints)} waypoints.")

    print(f"[MAIN] Total salas: {len(facultad)}\n")

    motor = MotorGrafico(SCENE_PATH, PERSON_PATH, facultad)
    motor.start()
    motor.run()

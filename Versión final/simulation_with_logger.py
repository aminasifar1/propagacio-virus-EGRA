"""
Standalone simulation runner with integrated logging.

This script wraps the existing DEMO components to track and export
contacts and infections without modifying the original files.

Usage:
    python DEMO\simulation_with_logger.py

Outputs CSV to: DEMO/exports/
"""
import os
import json
import sys
import time
import random
import csv
import pygame as pg
import moderngl as mgl
import numpy as np
import glm

from facultat import Sala, Clase, Pasillo
from escenario import Escenario
from camera import Camera
from person import Person
from marker import Marker
from virus import Virus
from infectionbar import InfectionBar


class SimulationLogger:
    """Tracks contacts and infections during simulation and exports to CSV."""

    def __init__(self, out_dir: str = None):
        self.out_dir = out_dir or os.path.join(os.getcwd(), "DEMO", "exports")
        os.makedirs(self.out_dir, exist_ok=True)
        self.start_time = time.time()
        self.contacts = []
        self.infections = []
        self.states = []
        # maps person object id to a person identifier for CSV output
        self.person_map = {}

    def _now(self):
        return time.time() - self.start_time

    def get_person_id(self, person_obj):
        """Get or create a unique ID for a person."""
        obj_id = id(person_obj)
        if obj_id not in self.person_map:
            self.person_map[obj_id] = len(self.person_map) + 1
        return self.person_map[obj_id]

    def log_contact(self, p1, p2, distance, method="proximity"):
        """Log a contact between two people."""
        try:
            p1_id = self.get_person_id(p1)
            p2_id = self.get_person_id(p2)
            self.contacts.append({
                "time": self._now(),
                "p1": p1_id,
                "p2": p2_id,
                "distance": float(distance),
                "method": method,
            })
        except Exception:
            pass

    def log_infection(self, source, target, method="direct"):
        """Log an infection event."""
        try:
            src_id = self.get_person_id(source) if source else None
            tgt_id = self.get_person_id(target)
            self.infections.append({
                "time": self._now(),
                "source": src_id,
                "target": tgt_id,
                "method": method,
            })
        except Exception:
            pass

    def log_state(self, total_people, infected_count):
        """Log population state (total and infected)."""
        self.states.append({
            "time": self._now(),
            "total_people": int(total_people),
            "infected": int(infected_count),
        })

    def dump_csv(self):
        """Write CSV files and return output directory."""
        if self.contacts:
            path = os.path.join(self.out_dir, "contacts.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time", "p1", "p2", "distance", "method"])
                writer.writeheader()
                writer.writerows(self.contacts)
            print(f"Wrote {len(self.contacts)} contacts to {path}")

        if self.infections:
            path = os.path.join(self.out_dir, "infections.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time", "source", "target", "method"])
                writer.writeheader()
                writer.writerows(self.infections)
            print(f"Wrote {len(self.infections)} infections to {path}")

        if self.states:
            path = os.path.join(self.out_dir, "states.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time", "total_people", "infected"])
                writer.writeheader()
                writer.writerows(self.states)
            print(f"Wrote {len(self.states)} state snapshots to {path}")

        return self.out_dir


class LoggingVirusWrapper:
    """Wraps Virus to intercept infectar calls and log them."""

    def __init__(self, virus, logger):
        self.virus = virus
        self.logger = logger
        self._inside_check_infections = False

    def __getattr__(self, name):
        return getattr(self.virus, name)

    def infectar(self, person):
        """Intercept infectar to log infection."""
        # Log the infection (source unknown without more context, use None)
        self.logger.log_infection(source=None, target=person, method="infectar")
        # Call original
        self.virus.infectar(person)

    def check_infections(self, mundo):
        """Intercept check_infections to log contacts and infections."""
        self._inside_check_infections = True

        # Get infected/uninfected people
        infected_people = []
        uninfected_people = []
        for nombre in mundo:
            for p in mundo[nombre].personas:
                if p.ring:
                    infected_people.append(p)
                else:
                    uninfected_people.append(p)

        # Log proximity contacts and infections (direct method)
        for infected in infected_people:
            for uninfected in uninfected_people:
                dist = glm.length(infected.position - uninfected.position)
                # Log all proximity contacts
                self.logger.log_contact(infected, uninfected, distance=float(dist), method="direct_radius")

                # Check if infection happens
                if dist < infected.ring.contagion_radius:
                    if random.random() < self.virus.infection_probability:
                        self.logger.log_infection(source=infected, target=uninfected, method="direct_radius")

        # Call original check_infections
        self.virus.check_infections(mundo)
        self._inside_check_infections = False


def load_obj(path: str):
    """Load OBJ file (copied from DEMO.py) ."""
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
                    for i in range(1, len(face_indices) - 1):
                        faces.append((face_indices[0], face_indices[i], face_indices[i + 1]))
                    for i in range(len(face_indices)):
                        p1 = face_indices[i]
                        p2 = face_indices[(i + 1) % len(face_indices)]
                        edges.add(tuple(sorted((p1, p2))))
    except FileNotFoundError:
        print(f"Error: El fitxer '{path}' no s'ha trobat.")
        raise

    if not vertices:
        return np.array([]), np.array([]), np.array([]), (glm.vec3(0), glm.vec3(0))

    np_vertices = np.array(vertices, dtype='f4')
    min_coords = np.min(np_vertices, axis=0)
    max_coords = np.max(np_vertices, axis=0)
    bounding_box = (glm.vec3(min_coords), glm.vec3(max_coords))

    vertex_normals = [np.zeros(3) for _ in range(len(vertices))]
    for face in faces:
        v0, v1, v2 = (np.array(vertices[i]) for i in face)
        face_normal = np.cross(v1 - v0, v2 - v0)
        for vertex_index in face:
            vertex_normals[vertex_index] += face_normal
    vertex_normals = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0, 1, 0]) for v in vertex_normals]

    tri_vertices_data = []
    normals_data = []
    for face in faces:
        for vertex_index in face:
            tri_vertices_data.extend(vertices[vertex_index])
            normals_data.extend(vertex_normals[vertex_index])

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


class MotorGraficoWithLogger:
    """Modified MotorGrafico that uses LoggingVirusWrapper."""

    def __init__(self, scene_path, person_path, facultad, win_size=(640, 360), logger=None):
        pg.init()
        pg.display.set_caption("3D Viewer + Logger - WASD moverte, TAB soltar ratón")
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        self.screen = pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.front_face = 'ccw'
        self.aspect_ratio = win_size[0] / win_size[1]

        self.camera = Camera(self)
        self.mundo = facultad
        self.marker = Marker(self.ctx, self.camera)
        self.ui_surface = pg.Surface(self.WIN_SIZE, pg.SRCALPHA)

        self.clock = pg.time.Clock()
        self.last_time = time.time()
        self.delta_time = 0.016
        self.frame_count = 0
        self.fps = 0
        self.show_bboxes = False

        # Virus with logger
        self.logger = logger
        virus = Virus(self, 0.2, 0.0, 0.2, 1, 0.9)
        self.virus = LoggingVirusWrapper(virus, logger) if logger else virus

        tri_data, normals, line_data, bounding_box = load_obj(scene_path)
        self.object = Escenario(self.ctx, self.camera, tri_data, normals, line_data, bounding_box)
        self.object.app = self

        self.p_tri_data, self.p_normals, self.p_line_data, bounding_box = load_obj(person_path)
        self.people = []
        self.simulando = False
        self.tiempo_persona = 0.0
        self.intervalo_spawn = 4.0
        self.max_people = 100  # Máximo de personas en la simulación

        first_person = Person(self.ctx, self.camera, self.p_tri_data, self.p_normals, self.p_line_data, facultad, ['aula1'], 'pasillo')
        self.person_vao_tri = self.ctx.vertex_array(self.object.shader, [(first_person.tri_vbo, '3f', 'in_position'), (first_person.nrm_vbo, '3f', 'in_normal')])
        self.person_vao_line = self.ctx.vertex_array(self.object.shader, [(first_person.line_vbo, '3f', 'in_position')])

    def create_person(self, schedule=[], spawn='pasillo'):
        persona = Person(self.ctx, self.camera, self.p_tri_data, self.p_normals, self.p_line_data, self.mundo, schedule, spawn)
        self.people.append(persona)
        return persona

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
            vertices = np.array([-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1], dtype='f4')
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

    def start(self):
        print("[MOTOR] Inicializando recursos...")
        self.activo = True
        print("[MOTOR] Motor listo.")

    def run(self):
        if not self.activo:
            print("[MOTOR] Motor no inicializado.")
            return

        rooms = list(self.mundo.keys())
        if 'pasillo' in rooms:
            rooms.remove('pasillo')
        clean_rooms = {room: 0 for room in rooms}

        print("[MOTOR] Iniciando ciclo principal...")
        last_frame_time = time.time()

        def aabb_collision(pos1, bb1_half, pos2, bb2_half):
            return (abs(pos1.x - pos2.x) <= (bb1_half.x + bb2_half.x) and
                    abs(pos1.y - pos2.y) <= (bb1_half.y + bb2_half.y) and
                    abs(pos1.z - pos2.z) <= (bb1_half.z + bb2_half.z))

        try:
            while True:
                dt = self.clock.tick(60) / 1000.0
                current_frame_time = time.time()
                self.delta_time = current_frame_time - last_frame_time
                if self.delta_time == 0:
                    self.delta_time = 1e-6
                last_frame_time = current_frame_time

                for e in pg.event.get():
                    self.marker.handle_input(pg.key.get_pressed())
                    self.camera.handle_mouse(e)
                    if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                        return
                    elif e.type == pg.KEYDOWN:
                        if e.key == pg.K_p:
                            self.simulando = not self.simulando
                            print("▶️ Simulación:", "Iniciada" if self.simulando else "Pausada")
                        elif e.key == pg.K_b:
                            self.show_bboxes = not self.show_bboxes
                            print(f"Mostrar bounding boxes: {self.show_bboxes}")
                        elif e.key == pg.K_r:
                            self.people.clear()
                            self.tiempo_persona = 0.0
                            print("🔄 Simulación reiniciada")

                self.camera.move(self.delta_time)
                self.camera.update_matrices()
                self.ctx.clear(0.07, 0.07, 0.09)

                if self.simulando:
                    self.tiempo_persona += dt
                    if self.tiempo_persona >= self.intervalo_spawn and len(self.people) < self.max_people:
                        selection = random.choice(rooms)
                        p = self.create_person([selection])
                        # Infect first person in each room to seed the virus
                        if clean_rooms[selection] == 0:
                            self.virus.virus.infectar(p)
                            print(f"🦠 Initial infection in room: {selection}")
                        clean_rooms[selection] += 1
                        self.tiempo_persona = 0.0

                    self.virus.virus.tick_timer += self.delta_time
                    if self.virus.virus.tick_timer >= self.virus.virus.tick_duration:
                        self.virus.virus.tick_timer -= self.virus.virus.tick_duration
                        self.virus.check_infections(self.mundo)
                        
                        # Log state every tick
                        num_infected = sum(1 for p in self.people if hasattr(p, 'ring') and p.ring is not None)
                        total_people = len(self.people)
                        if self.logger:
                            self.logger.log_state(total_people, num_infected)

                self.object.render()
                self.marker.render()
                light_pos = self.object.update_light_position()
                self.virus.virus.render(light_pos)

                for p in self.people:
                    if self.simulando:
                        old_pos = glm.vec3(p.position)
                        p.update(self.delta_time)
                        for other in self.people:
                            if other is p:
                                continue
                            if aabb_collision(p.position, p.bb_half, other.position, other.bb_half):
                                p.position = old_pos
                                p.m_model = glm.translate(glm.mat4(1.0), old_pos)
                                break
                    p.render(self.object.shader, self.person_vao_tri, self.person_vao_line, light_pos)

                self.ui_surface.fill((0, 0, 0, 0))
                self._render_ui_overlay()

                self.frame_count += 1
                if time.time() - self.last_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_time)
                    self.frame_count = 0
                    self.last_time = time.time()
                    pg.display.set_caption(f"3D Viewer - FPS: {self.fps:.1f} - Logger active")

                pg.display.flip()

        finally:
            pg.event.set_grab(False)
            pg.quit()


def main():
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH, "DEMO", "data", "salas")
    SCENE_PATH = os.path.join(ROOT_PATH, "DEMO", "Models", "OBJ.obj")
    PERSON_PATH = os.path.join(ROOT_PATH, "DEMO", "Models", "person.obj")

    facultad = {}
    for archivo in os.listdir(DATA_PATH):
        if not archivo.endswith(".json"):
            continue
        ruta = os.path.join(DATA_PATH, archivo)
        nombre = os.path.splitext(archivo)[0]
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        tipo = data.get("tipo")
        if tipo == "clase":
            sala = Clase.from_json_struct(data)
        elif tipo == "pasillo":
            sala = Pasillo.from_json_struct(data)
        else:
            sala = Sala.from_json_struct(data)
        facultad[nombre] = sala

    print("Starting simulation with logging...")
    logger = SimulationLogger()
    motor = MotorGraficoWithLogger(SCENE_PATH, PERSON_PATH, facultad, logger=logger)
    motor.start()
    motor.run()

    # Export logs on exit
    print("\n" + "=" * 50)
    print("Exporting logs...")
    out_dir = logger.dump_csv()
    print(f"Logs saved to: {out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()

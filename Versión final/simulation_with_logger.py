"""
Standalone simulation runner with integrated logging.

This script wraps the existing DEMO components to track and export
contacts and infections without modifying the original files.

Usage:
    python "Versión final/simulation_with_logger.py"

Outputs CSV to: Versión final/exports/
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
from simclock import SimClock
from DEMO import load_obj  # Import load_obj from DEMO.py


class SimulationLogger:
    """Tracks contacts and infections during simulation and exports to CSV."""

    def __init__(self, out_dir: str = None):
        self.out_dir = out_dir or os.path.join(os.getcwd(), "Versión final", "exports")
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

    def log_infection(self, source, target, method="direct", room=None):
        """Log an infection event."""
        try:
            src_id = self.get_person_id(source) if source else None
            tgt_id = self.get_person_id(target)
            # Get room from target person if not provided
            if room is None:
                if hasattr(target, 'sala'):
                    room = target.sala
                elif hasattr(target, 'location'):
                    room = target.location
            self.infections.append({
                "time": self._now(),
                "source": src_id,
                "target": tgt_id,
                "method": method,
                "room": room or "unknown",
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
                writer = csv.DictWriter(f, fieldnames=["time", "source", "target", "method", "room"])
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
        # Get room from person
        room = getattr(person, 'sala', 'unknown')
        # Log the infection (source unknown without more context, use None)
        self.logger.log_infection(source=None, target=person, method="infectar", room=room)
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
                        # Get room from infected or uninfected person
                        room = getattr(infected, 'sala', None) or getattr(uninfected, 'sala', 'unknown')
                        self.logger.log_infection(source=infected, target=uninfected, method="direct_radius", room=room)

        # Call original check_infections
        self.virus.check_infections(mundo)
        self._inside_check_infections = False


class MotorGraficoWithLogger:
    """Modified MotorGrafico that uses LoggingVirusWrapper."""

    def __init__(self, scene_path, person_path, facultad, horaris_path=None, win_size=(1820, 980), logger=None):
        pg.init()
        pg.display.set_caption("Epidemiological Simulator + Logger - WASD moverte, TAB soltar ratón")
        self.WIN_SIZE = win_size
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)
        pg.display.gl_set_attribute(pg.GL_STENCIL_SIZE, 8)
        self.screen = pg.display.set_mode(self.WIN_SIZE, pg.OPENGL | pg.DOUBLEBUF)
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.front_face = 'ccw'
        self.aspect_ratio = win_size[0] / win_size[1]
        self.speed = 1

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
        
        # Infection bar
        self.infection_bar = InfectionBar(self.WIN_SIZE[0], self.WIN_SIZE[1])

        # SimClock and Scheduler
        self.day_sim_seconds = 20 * 60  # 20 minutos para un día lectivo
        self.sim_clock = SimClock(day_sim_seconds=self.day_sim_seconds, speed_mult=self.speed)
        self.sim_time_in_day = 0.0
        self.current_day = 0  # 0=monday, 1=tuesday, etc.
        
        # Load schedule from CSV
        self.schedule_events = []
        if horaris_path and os.path.exists(horaris_path):
            with open(horaris_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.schedule_events.append(row)
            print(f"[SCHEDULER] Loaded {len(self.schedule_events)} schedule entries")
        
        self.spawned_events = set()  # Track which events have been spawned

        # Load scene
        scene_data, bounding_box, texture_file = load_obj(scene_path)
        self.object = Escenario(self.ctx, self.camera, scene_data, bounding_box, texture_file)
        self.object.app = self

        # Load person model
        p_data, p_bbox, p_texture = load_obj(person_path)
        self.p_data = p_data
        self.p_texture = p_texture
        self.people = []
        self.simulando = False

        # Create person VBO and shader
        from person import Person
        temp_person = Person(self, self.ctx, self.camera, self.p_data, facultad, ['Q1-0013'], 'pasillo')
        self.person_shader = temp_person.get_shader(self.ctx)
        
        # Create shared VBO and VAO
        self.person_vbo = self.ctx.buffer(self.p_data)
        self.person_vao_tri = self.ctx.vertex_array(
            self.person_shader, 
            [(self.person_vbo, '3f 12x 2f 3f 3f', 'in_position', 'in_texcoord', 'in_color', 'in_smooth_normal')]
        )
        self.person_vao_line = None

    def create_person(self, schedule=[], spawn='pasillo'):
        persona = Person(self, self.ctx, self.camera, self.p_data, self.mundo, schedule, spawn)
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

        print("[MOTOR] Iniciando ciclo principal...")
        last_frame_time = time.time()
        grupo_personas = {}  # group_id -> list of Person
        dt_sim = 0.0  # delta time for simulation

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
                            grupo_personas.clear()
                            self.sim_time_in_day = 0.0
                            self.current_day = 0
                            self.sim_clock.world_time = 0.0
                            self.spawned_events.clear()
                            print("🔄 Simulación reiniciada")
                        elif e.key == pg.K_1:
                            self.speed = 1
                        elif e.key == pg.K_2:
                            self.speed = 3
                        elif e.key == pg.K_3:
                            self.speed = 10

                self.camera.move(self.delta_time)
                self.camera.update_matrices()
                self.ctx.clear(0.07, 0.07, 0.09)

                if self.simulando:
                    scaled_dt = dt * self.speed
                    dt_sim = self.sim_clock.step(scaled_dt)
                    self.sim_time_in_day += dt_sim
                    
                    # Calculate current slot (30 min intervals, starting at 8:00)
                    slot_now = int(self.sim_clock.minute_of_day() // 30)
                    
                    # Map slot to day_name for scheduler
                    day_names = ['mon', 'tue', 'wed', 'thu', 'fri']
                    day_name = day_names[self.current_day % 5]

                    # Check schedule for spawns
                    if self.schedule_events:
                        # Convert slot to hour:minute
                        total_minutes = self.sim_clock.minute_of_day()
                        hour = 8 + (total_minutes // 60)
                        minute = total_minutes % 60
                        
                        for event in self.schedule_events:
                            # Parse event time (format: "HH:MM")
                            event_start = event['start']
                            event_hour, event_minute = map(int, event_start.split(':'))
                            
                            # Check if this event matches current day and time (with 1-minute tolerance)
                            event_key = f"{event['group']}_{event['day']}_{event_start}"
                            if (event['day'] == day_name and 
                                event_hour == hour and 
                                abs(event_minute - minute) <= 1 and
                                event_key not in self.spawned_events):
                                
                                group_id = event['group']
                                room = event['room']
                                
                                if group_id not in grupo_personas:
                                    grupo_personas[group_id] = []
                                
                                # Spawn 10-20 people for this group
                                num_people = random.randint(10, 20)
                                for _ in range(num_people):
                                    persona = self.create_person([room], spawn='pasillo')
                                    grupo_personas[group_id].append(persona)
                                    
                                # Infect 1 random person in the group
                                if grupo_personas[group_id]:
                                    patient_zero = random.choice(grupo_personas[group_id])
                                    self.virus.virus.infectar(patient_zero)
                                    print(f"🦠 Initial infection in group {group_id} at {hour:02d}:{minute:02d}")
                                
                                self.spawned_events.add(event_key)

                    # Virus tick
                    self.virus.virus.tick_timer += dt_sim
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
                        p.update(dt_sim)
                        for other in self.people:
                            if other is p:
                                continue
                            if aabb_collision(p.position, p.bb_half, other.position, other.bb_half):
                                p.position = old_pos
                                p.m_model = glm.translate(glm.mat4(1.0), old_pos)
                                break
                    p.render(self.person_shader, self.person_vao_tri, self.person_vao_line, light_pos)

                # Render UI
                self.ui_surface.fill((0, 0, 0, 0))
                
                # Infection bar
                num_infected = sum(1 for p in self.people if hasattr(p, 'ring') and p.ring is not None)
                total_people = len(self.people)
                self.infection_bar.render(self.ui_surface, num_infected, total_people)
                
                # Clock and speed text
                font = pg.font.Font(None, 24)
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                total_minutes = self.sim_clock.minute_of_day()
                hour = 8 + (total_minutes // 60)
                minute = total_minutes % 60
                clock_text = f"{day_names[self.current_day % 5]} {hour:02d}:{minute:02d}"
                speed_text = f"Speed: x{self.speed}"
                text_surf = font.render(clock_text, True, (255, 255, 255))
                speed_surf = font.render(speed_text, True, (255, 255, 255))
                self.ui_surface.blit(text_surf, (10, 10))
                self.ui_surface.blit(speed_surf, (10, 40))
                
                self._render_ui_overlay()

                self.frame_count += 1
                if time.time() - self.last_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_time)
                    self.frame_count = 0
                    self.last_time = time.time()
                    pg.display.set_caption(f"Epidemiological Simulator - FPS: {self.fps:.1f} - Logger active")

                pg.display.flip()

        finally:
            pg.event.set_grab(False)
            pg.quit()


def main():
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH, "Versión final", "data", "salas")
    SCENE_PATH = os.path.join(ROOT_PATH, "Versión final", "Models", "DEF.obj")
    PERSON_PATH = os.path.join(ROOT_PATH, "Versión final", "Models", "person.obj")
    HORARIS_PATH = os.path.join(ROOT_PATH, "Versión final", "Horarios.csv")
    
    print(f"[MAIN] Ruta base: {ROOT_PATH}")

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
        print(f"[MAIN] Cargada sala '{nombre}' ({tipo}) con {len(sala.waypoints)} waypoints.")

    print(f"[MAIN] Total salas: {len(facultad)}\n")
    print("Starting simulation with logging...")
    
    logger = SimulationLogger()
    motor = MotorGraficoWithLogger(SCENE_PATH, PERSON_PATH, facultad, HORARIS_PATH, logger=logger)
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

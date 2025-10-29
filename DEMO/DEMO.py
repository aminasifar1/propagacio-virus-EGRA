import os
import json
import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import math
import random
from facultat import Sala, Clase, Pasillo
from escenario import Escenario
from camera import Camera

def load_obj(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[glm.vec3, glm.vec3]]:
    """
    Carrega vèrtexs, cares (triangles) i arestes d'un fitxer OBJ, calculant normals per vèrtex
    per a un ombrejat suau (Phong).

    Args:
        path (str): La ruta al fitxer .obj.

    Returns:
        tuple: Una tupla contenint:
            - np.ndarray: Vèrtexs dels triangles per al renderitzat.
            - np.ndarray: Normals per vèrtex per a la il·luminació.
            - np.ndarray: Vèrtexs de les arestes per al renderitzat de línies.
            - tuple: El bounding box (min_coords, max_coords) de l'objecte.
    """
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
                    # Gestiona cares amb 3 o més vèrtexs (triangulació simple)
                    parts = line.strip().split()[1:]
                    face_indices = [int(p.split('/')[0]) - 1 for p in parts]
                    
                    # Triangula la cara si té més de 3 vèrtexs
                    for i in range(1, len(face_indices) - 1):
                        faces.append((face_indices[0], face_indices[i], face_indices[i + 1]))

                    # Afegeix les arestes de la cara
                    for i in range(len(face_indices)):
                        p1 = face_indices[i]
                        p2 = face_indices[(i + 1) % len(face_indices)]
                        edge = tuple(sorted((p1, p2)))
                        edges.add(edge)
    except FileNotFoundError:
        print(f"Error: El fitxer '{path}' no s'ha trobat.")
        raise
    except Exception as e:
        print(f"Error en processar el fitxer OBJ '{path}': {e}")
        raise

    if not vertices:
        return np.array([]), np.array([]), np.array([]), (glm.vec3(0), glm.vec3(0))

    # --- Càlcul del Bounding Box ---
    np_vertices = np.array(vertices, dtype='f4')
    min_coords = np.min(np_vertices, axis=0)
    max_coords = np.max(np_vertices, axis=0)
    bounding_box = (glm.vec3(min_coords), glm.vec3(max_coords))

    # --- Càlcul de Normals per Vèrtex (Phong Shading) ---
    vertex_normals = [np.zeros(3) for _ in range(len(vertices))]
    for face in faces:
        v0, v1, v2 = (np.array(vertices[i]) for i in face)
        face_normal = np.cross(v1 - v0, v2 - v0)
        
        # Acumula la normal de la cara a cada vèrtex que la compon
        for vertex_index in face:
            vertex_normals[vertex_index] += face_normal

    # Normalitza totes les normals acumulades
    vertex_normals = [
        v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0, 1, 0])
        for v in vertex_normals
    ]

    # --- Preparació de Dades per a OpenGL ---
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

# =====================================================
#                   MOTOR GRÁFICO
# =====================================================
class MotorGrafico:
    """
    Se encarga del ciclo de vida de la simulación:
    - inicialización de recursos
    - actualización de objetos
    - renderizado
    """

    def __init__(self, scene_path, win_size=(1536, 864)):
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

        # Camera
        self.camera = Camera(self)

        # Para renderizar la UI de pygame sobre OpenGL
        self.ui_surface = pg.Surface(self.WIN_SIZE, pg.SRCALPHA)

        # Clock
        self.clock = pg.time.Clock()
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.delta_time = 0.016

        self.tick_duration = 0.2
        self.tick_timer = 0.0
        self.infection_probability = 1

        # Scenario
        tri_data, normals, line_data, bounding_box = load_obj(scene_path)
        self.object = Escenario(self.ctx, self.camera, tri_data, normals, line_data, bounding_box)
        self.object.app = self

        min_coords, max_coords = self.object.bounding_box
        print(f"Escenari carregat. Bounding Box:")
        print(f"  MIN: {min_coords}")
        print(f"  MAX: {max_coords}")

    def start(self):
        """Aquí inciaremos y cargaremos a todas las personas que vayamos a simular"""
        print("[MOTOR] Inicializando recursos gráficos...")
        self.activo = True
        print("[MOTOR] Motor iniciado correctamente.")

    def run(self):
        """Inicia el bucle principal de simulación (simplificado)."""
        if not self.activo:
            print("[MOTOR] No se ha inicializado el motor. Llamar a iniciar().")
            return
        print("[MOTOR] Iniciando ciclo principal...")

        last_frame_time = time.time()
        while True:
            keys = pg.key.get_pressed()

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

            self.camera.move(self.delta_time)

            self.tick_timer += self.delta_time

            if self.tick_timer >= self.tick_duration:
                self.tick_timer -= self.tick_duration
            
            self.camera.update_matrices()
            
            # --- RENDERITZAT ---
            self.ctx.clear(0.07, 0.07, 0.09)

            self.object.render()

            # --- CONTROL FPS ---
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time
                pg.display.set_caption(f"3D Viewer - FPS: {self.fps:.1f} - WASD moverte, TAB soltar ratón")

            pg.display.flip()
        print("[MOTOR] Ejecución finalizada (placeholder).")


# =====================================================
#                     MAIN
# =====================================================
if __name__ == "__main__":
    # Establecer la ruta base del proyecto
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH, "DEMO" ,"data", "salas")
    SCENE_PATH = os.path.join(ROOT_PATH, "Models" ,"OBJ.obj")
    print(f"[MAIN] Ruta base: {ROOT_PATH}")

    # Crear diccionario global 'facultad'
    facultad = {}

    # Cargar cada sala desde JSON
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

    print(f"[MAIN] Total de salas cargadas: {len(facultad)}\n")

    # Crear motor gráfico y registrar elementos que se deban renderizar
    motor = MotorGrafico(SCENE_PATH)

    # Inicializar y ejecutar motor
    motor.start()
    motor.run()

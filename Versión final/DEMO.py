import os
import json
import pygame as pg
import moderngl as mgl
import numpy as np
import glm
import sys
import time
import random
from facultat import Sala, Clase, Pasillo, WaypointVisualizer
from escenario import Escenario
from camera import Camera
from person import Person
from marker import Marker
from virus import Virus
from infectionbar import InfectionBar
import menu

# =====================================================
#                   GRAFO PUNTOS
# =====================================================

import re

def parse_graph_from_text(raw: str):
    nodes = {}
    adj = {}

    id_re = re.compile(r"\)\s*(\d+)\s*:")
    coord_re = re.compile(r":\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*-\s*\(([^)]*)\)?")

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue

        m_id = id_re.search(line)
        m_c = coord_re.search(line)
        if not (m_id and m_c):
            continue

        node_id = int(m_id.group(1))
        x, y, z = map(float, m_c.group(1, 2, 3))
        nei_txt = m_c.group(4).strip()

        nodes[node_id] = (x, y, z)
        adj.setdefault(node_id, set())

        for t in nei_txt.split(","):
            t = t.strip()
            if t.isdigit():
                nb = int(t)
                adj[node_id].add(nb)
                adj.setdefault(nb, set()).add(node_id)

    edges = set()
    for u, nbs in adj.items():
        for v in nbs:
            if u != v:
                edges.add((min(u, v), max(u, v)))

    return nodes, edges, adj

GRAPH_VERT = """
#version 330
in vec3 in_pos;
uniform mat4 mvp;
uniform float point_size;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    gl_PointSize = point_size;   // para puntos
}
"""

GRAPH_FRAG = """
#version 330
uniform vec3 color;
out vec4 fragColor;
void main() {
    fragColor = vec4(color, 1.0);
}
"""

def build_graph_drawables(ctx: mgl.Context, nodes: dict, edges: set):
    # ---- buffers de posiciones ----
    pts = np.array([nodes[i] for i in sorted(nodes.keys())], dtype="f4")

    segs = []
    for u, v in sorted(edges):
        if u in nodes and v in nodes:
            segs.append(nodes[u])
            segs.append(nodes[v])
    segs = np.array(segs, dtype="f4")

    prog = ctx.program(vertex_shader=GRAPH_VERT, fragment_shader=GRAPH_FRAG)

    vbo_pts = ctx.buffer(pts.tobytes())
    vao_pts = ctx.vertex_array(prog, [(vbo_pts, "3f", "in_pos")])

    vbo_lines = ctx.buffer(segs.tobytes())
    vao_lines = ctx.vertex_array(prog, [(vbo_lines, "3f", "in_pos")])

    return {
        "prog": prog,
        "vao_pts": vao_pts,
        "vao_lines": vao_lines,
        "n_points": len(pts),
        "n_line_verts": len(segs),
    }

def render_graph(ctx: mgl.Context, graph, mvp: np.ndarray,
                 point_size=10.0,
                 color_points=(1.0, 0.2, 0.2),
                 color_lines=(0.2, 0.8, 1.0),
                 overlay=False):

    prog = graph["prog"]
    prog["mvp"].write(mvp.astype("f4").tobytes())
    prog["point_size"].value = float(point_size)

    # Para que gl_PointSize funcione en core profile:
    ctx.enable(mgl.PROGRAM_POINT_SIZE)

    if overlay:
        # siempre visible (sin oclusión)
        ctx.disable(mgl.DEPTH_TEST)

    # Líneas
    prog["color"].value = color_lines
    graph["vao_lines"].render(mode=mgl.LINES, vertices=graph["n_line_verts"])

    # Puntos
    prog["color"].value = color_points
    graph["vao_pts"].render(mode=mgl.POINTS, vertices=graph["n_points"])

    if overlay:
        ctx.enable(mgl.DEPTH_TEST)

RAW_POINTS = r"""
	- (pitagoras) 1: 4.90, -0.10, -15.15 - (0,2,
	- (escalera principal) 2: 7.70, -0.10, -13.65 - (0,1,3,5
	- (pasillo izquierda abajo) 3: 10.60, -0.10, -13.65 - (2,6
	- (pasillo medio abajo) 4: 13.90, -0.10, -13.65 - (3,6
	- (principio escalera) 5: 7.70, -0.10, -14.80 - (2,
	- (principio primera rampa izquierda) 6: 10.60, -0.10, -18.00 - (3,4,7,8,9)
	- (principio primera rampa medio) 7: 13.90, -0.10, -18.00 - (3,4,6,8,9)
	- (final primera rampa izquierda) 8: 10.60, 0.45, -30.10 - (6,7,9,10,12,13)
	- (final primera rampa medio) 9: 13.90, 0.45, -30.10 - (6,7,8,10,12)
	- (final pitagoras) 10: 10.60, 0.45, -31.25 - (8,9,11,13,12)
	- (elbow pitagoras) 11: 4.90, 0.45, -31.25 - (1,10)
	- (inicio Q1/0) 12: 13.90, 0.45, -34.10 - (10,9,8,
	- (principio segunda rampa) 13: 13.90, 0.45, -46.95 - (10,8,12,14)
	- (final segunda rampa) 14: 13.90, 1.45, -58.75 - (13,15
	- (inicio Q2/0) 15: 13.90, 1.45, -62.70 - (14,16
	- (inicio tercera rampa) 16: 13.90, 1.45, -76.10 - (15,17,
	- (final tercera rampa) 17: 13.90, 2.00, -87.85 - (16,18,
	- (inicio Q3/0) 18: 13.90, 2.00, -91.85 - (17,19,
	- (inicio cuarta rampa) 19: 13.90, 2.00, -104.20 - (18,20
	- (final cuarta rampa) 20: 13.90, 2.60, -117.00 - (19,21
	- (incio Q4/0) 21: 13.90, 2.60, -121.05 - (20,22
	- (entrada rampa Q4/0) 22: 10.60, 2.60, -118.25 - (20,21,23)
	- (abajo escaleras Q4) 23: 4.85, 2.60, -118.25 - (22,24)
	- (inicio escaleras Q4) 24: 4.85, 2.60, -115.50 - (23,
	- (entrada rampa Q3/0) 25: 10.60, 2.00, -89.10 - (17,18,19,26)
	- (abajo escaleras Q3) 26: 4.85, 2.00, -89.10 - (25,27)
	- (inicio escaleras Q3) 27: 4.85, 2.00, -86.20 - (26,
	- (entrada rampa Q2/0) 28: 10.60, 1.45, -60.00 - (14,15,16,29)
	- (abajo escaleras Q2) 29: 4.85, 1.45, -60.00 - (28,30)
	- (inicio escaleras Q2) 30: 4.85, 1.45, -57.20 - (29,
    """

nodes, edges, adj = parse_graph_from_text(RAW_POINTS)

# =====================================================
#                    CARREGAR OBJ
# =====================================================
def load_obj(path: str, default_color=(0.1, 0.1, 0.1)) -> tuple[np.ndarray, np.ndarray, tuple[glm.vec3, glm.vec3], str]:
    vertices = []
    normals = []
    texcoords = []
    faces = []
    
    # Diccionario de materiales: nombre -> (Color RGB, Ruta Textura)
    materials = {}
    current_material_name = None
    
    # Ruta base
    base_dir = os.path.dirname(path)
    texture_file = None # Guardaremos la primera textura que encontremos para pasarsela a Escenario

    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.startswith('mtllib '):
                    # --- PARSEO DEL ARCHIVO MTL ---
                    parts = line.split(maxsplit=1)
                    mtl_filename = parts[1].strip() if len(parts) > 1 else ""

                    # Candidatos: el indicado por el OBJ y fallback al "<obj_name>.mtl"
                    candidates = []
                    if mtl_filename:
                        candidates.append(os.path.join(base_dir, mtl_filename))
                    candidates.append(os.path.join(base_dir, os.path.splitext(os.path.basename(path))[0] + ".mtl"))

                    mtl_path = next((p for p in candidates if os.path.exists(p)), None)
                    if not mtl_path:
                        print(f"Advertencia: No se encontró el archivo MTL. Probados: {candidates}")
                        continue

                    try:
                        with open(mtl_path, 'r') as mtl_f:
                            mat_name = None
                            mat_color = default_color
                            mat_texture = None

                            for m_line in mtl_f:
                                m_line = m_line.strip()
                                if m_line.startswith('newmtl '):
                                    if mat_name:
                                        materials[mat_name] = {'color': mat_color, 'texture': mat_texture}
                                    mat_name = m_line.split()[1]
                                    mat_color = default_color
                                    mat_texture = None

                                elif m_line.startswith('Kd '):
                                    parts = m_line.split()
                                    mat_color = (float(parts[1]), float(parts[2]), float(parts[3]))

                                elif m_line.startswith('map_Kd '):
                                    parts = m_line.split()
                                    tex_name = parts[-1]
                                    mat_texture = os.path.join(base_dir, tex_name)
                                    if not texture_file:
                                        texture_file = mat_texture

                            if mat_name:
                                materials[mat_name] = {'color': mat_color, 'texture': mat_texture}

                    except FileNotFoundError:
                        print(f"Advertencia: No se encontró el archivo MTL: {mtl_path}")

                elif line.startswith('usemtl '):
                    # Cambiamos el material actual
                    current_material_name = line.split()[1]

                elif line.startswith('v '):
                    parts = line.split()
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith('vn '):
                    parts = line.split()
                    normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith('vt '):
                    parts = line.split()
                    texcoords.append((float(parts[1]), float(parts[2])))
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    face_verts = []
                    for p in parts:
                        vals = p.split('/')
                        v_idx = int(vals[0]) - 1
                        vt_idx = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else -1
                        vn_idx = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else -1
                        face_verts.append((v_idx, vt_idx, vn_idx))
                    
                    # Triangulación fan y asignar material actual a la cara
                    for i in range(1, len(face_verts) - 1):
                        faces.append({
                            'verts': (face_verts[0], face_verts[i], face_verts[i + 1]),
                            'material': current_material_name
                        })

    except FileNotFoundError:
        print(f"Error: El fichero '{path}' no existe.")
        raise

    if not vertices:
        return np.array([]), np.array([]), (glm.vec3(0), glm.vec3(0)), None
    
    smooth_sum = {}  # v_idx -> glm.vec3
    for face in faces:
        (a, b, c) = face['verts']

        # Normal de la cara como fallback si no hay vn
        pa = glm.vec3(vertices[a[0]])
        pb = glm.vec3(vertices[b[0]])
        pc = glm.vec3(vertices[c[0]])
        face_n = glm.cross(pb - pa, pc - pa)
        if glm.length(face_n) > 0:
            face_n = glm.normalize(face_n)
        else:
            face_n = glm.vec3(0, 1, 0)

        for (v_idx, vt_idx, vn_idx) in (a, b, c):
            if vn_idx >= 0:
                n = glm.vec3(normals[vn_idx])
            else:
                n = face_n

            if v_idx not in smooth_sum:
                smooth_sum[v_idx] = glm.vec3(0, 0, 0)
            smooth_sum[v_idx] += n

    # Normalizar
    smooth_normals = {}
    for v_idx, n_vec in smooth_sum.items():
        if glm.length(n_vec) > 0:
            smooth_normals[v_idx] = glm.normalize(n_vec)
        else:
            smooth_normals[v_idx] = glm.vec3(0, 1, 0)

    # --- Construcción del Buffer (Pos + Normal + UV + COLOR) ---
    vertex_data = []
    
    np_verts = np.array(vertices, dtype='f4')
    min_coords = np.min(np_verts, axis=0)
    max_coords = np.max(np_verts, axis=0)
    bounding_box = (glm.vec3(min_coords), glm.vec3(max_coords))

    for face in faces:
        mat_name = face['material']
        # Obtener color del material (o blanco si falla)
        color = (1.0, 1.0, 1.0)
        if mat_name in materials:
            color = materials[mat_name]['color']
        
        for v_idx, vt_idx, vn_idx in face['verts']:
            # 1. Posición (3f)
            pos = vertices[v_idx]
            vertex_data.extend(pos)
            
            # 2. Normal Plana (3f) - Para Iluminación
            if vn_idx >= 0: vertex_data.extend(normals[vn_idx])
            else: vertex_data.extend([0, 1, 0])
            
            # 3. UV (2f)
            if vt_idx >= 0: vertex_data.extend(texcoords[vt_idx])
            else: vertex_data.extend([0.0, 0.0])
            
            # 4. Color (3f)
            vertex_data.extend(color)

            # 5. NORMAL SUAVE (3f) - ¡NUEVO! - Para Contorno
            # Buscamos la normal promediada usando la posición como clave
            smooth_n = smooth_normals.get(v_idx, glm.vec3(0, 1, 0))
            vertex_data.extend([smooth_n.x, smooth_n.y, smooth_n.z])

    buffer_data = np.array(vertex_data, dtype='f4')
    return buffer_data, bounding_box, texture_file

def cargar_diccionarios_desde_carpeta(ruta):
    diccionario_final = {}

    for archivo in os.listdir(ruta):
        if archivo.endswith(".json"):
            ruta_archivo = os.path.join(ruta, archivo)
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    diccionario_final.update(data)
                else:
                    print(f"Advertencia: {archivo} no contiene un diccionario válido")

    return diccionario_final

# =====================================================
#                    MOTOR GRÀFIC
# =====================================================
class MotorGrafico:
    def __init__(self, scene_path, person_path, facultad, win_size=(1640, 1024)):
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
        self.speed = 1

        # Multi Vista
        self.menu_width = 400   # anchura del menú
        self.menu_surface = pg.Surface((self.menu_width, self.WIN_SIZE[1]), pg.SRCALPHA)

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

        self.show_bboxes = False

        # Virus
        self.tick_duration = 0.2
        self.tick_timer = 0.0
        self.tick_global = 0  # --- Contador global de ticks ---
        self.infection_probability = 0.2
        self.virus = Virus(self, self.tick_duration, self.tick_timer, self.infection_probability, 1, 0.9)

        # Waypoint Visualizer
        self.waypoint_visualizer = WaypointVisualizer(self.ctx, self.camera)
        self.show_waypoints = False
        for nombre, sala in self.mundo.items():
            self.waypoint_visualizer.build_from_sala(nombre, sala)

        # Escenari
        scene_data, bounding_box, texture_file = load_obj(scene_path)
        self.object = Escenario(self.ctx, self.camera, scene_data, bounding_box, texture_file)
        self.object.app = self

        # Persones
        self.person_shader = Person.get_shader(self.ctx)
        self.p_data, p_bbox, self.p_tex_path = load_obj(person_path, default_color=(0.6, 0.6, 0.7))
        self.person_texture = None
        if self.p_tex_path:
            try:
                img = pg.image.load(self.p_tex_path).convert()
                img = pg.transform.flip(img, False, True)
                self.person_texture = self.ctx.texture(img.get_size(), 3, pg.image.tostring(img, 'RGB'))
                self.person_texture.build_mipmaps()
            except Exception as e: print(e)
        self.people = []
        self.simulando = False
        self.tiempo_persona = 0.0
        self.intervalo_spawn = 4.0
        self.people_type = cargar_diccionarios_desde_carpeta(HORARIS_PATH)
        for i in ["Q1-0007","Q1-0013"]:
            for j in range(50):
                p = self.create_person([i], i)

        self.people[0].infectar(1)  # Infectem la primera persona

        # Creem la primera persona només per obtenir el VAO
        # first_person = Person(self, self.ctx, self.camera, self.p_data, facultad, ['aula1'], 'pasillo', position=glm.vec3(1000,1000,1000))
        self.person_vbo = self.ctx.buffer(self.p_data)

        # Usamos el shader compilado
        self.person_vao_tri = self.ctx.vertex_array(
            self.person_shader, 
            [(self.person_vbo, '3f 12x 2f 3f 3f', 'in_position', 'in_texcoord', 'in_color', 'in_smooth_normal')]
        )
        self.person_vao_line = None

        min_coords, max_coords = self.object.bounding_box
        print(f"Escenari carregat. Bounding Box: MIN {min_coords}, MAX {max_coords}")

        self.graph = build_graph_drawables(self.ctx, nodes, edges)

    # Crear persona
    def create_person(self, schedule=[], spawn='pasillo'):
        # persona = Person(self.ctx, self.camera, self.p_tri_data, self.p_normals, self.p_line_data, self.mundo, schedule, spawn)
        persona = Person(self, self.ctx, self.camera, self.p_data, self.mundo, schedule, spawn)
        persona.infection_tick = None  # --- Manté el tick d’infecció ---
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

        # # ==========================
        # # Funció de col·lisió AABB
        # # ==========================
        # def aabb_collision(pos1, bb1_half, pos2, bb2_half):
        #     """
        #     Retorna True si hi ha intersecció entre dos AABB centrats a pos1 i pos2.
        #     bb_half: semi-dimensions del bounding box.
        #     """
        #     return (abs(pos1.x - pos2.x) <= (bb1_half.x + bb2_half.x) and
        #             abs(pos1.y - pos2.y) <= (bb1_half.y + bb2_half.y) and
        #             abs(pos1.z - pos2.z) <= (bb1_half.z + bb2_half.z))

        while True:
            dt = self.clock.tick(60)/1000.0
            dt *= self.speed
            current_frame_time = time.time()
            self.delta_time = current_frame_time - last_frame_time
            if self.delta_time == 0: self.delta_time = 1e-6
            last_frame_time = current_frame_time
            keys = pg.key.get_pressed()
            if (keys[pg.K_LALT] or keys[pg.K_RALT]):
                self.marker.handle_input(keys)

            # ==========================
            # Gestió d'events
            # ==========================
            for e in pg.event.get():
                mx, my = pg.mouse.get_pos()
                if mx < self.menu_width:
                    menu.handle_menu_event(e)
                self.camera.handle_mouse(e)
                if not keys[pg.K_LALT] and not keys[pg.K_RALT]:
                    self.marker.handle_input(keys)
                
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    pg.event.set_grab(False)
                    pg.quit()
                    sys.exit()
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
                    elif e.key == pg.K_g:
                        self.show_waypoints = not self.show_waypoints
                        print(f"📍 Mostrar waypoints: {self.show_waypoints}")
                    elif e.key == pg.K_q:
                        # Cámara anterior
                        self.camera.prev_preset()
                    elif e.key == pg.K_e:
                        # Cámara siguiente
                        self.camera.next_preset()
                if e.type == pg.KEYDOWN:
                    if e.key == pg.K_1:
                        self.speed = 1.0
                        print("Velocidad x1")
                    if e.key == pg.K_2:
                        self.speed = 3.0
                        print("Velocidad x3")
                    if e.key == pg.K_3:
                        self.speed = 10.0
                        print("Velocidad x10")

            # Actualitzar càmera
            self.camera.move(self.delta_time)
            self.camera.update_matrices()
            self.ctx.clear(0.01, 0.8, 0.9, 1.0)

            # ==========================
            # Spawn de persones
            # ==========================
            if self.simulando:
                # self.tiempo_persona += dt
                # if self.tiempo_persona >= self.intervalo_spawn:
                #     selection = random.choice(rooms)
                #     p = self.create_person([selection])
                #     if clean_rooms[selection] == 0:
                #         self.virus.infectar(p)
                #     clean_rooms[selection] += 1
                #     self.tiempo_persona = 0.0

                # Tick virus
                self.tick_timer += self.delta_time * self.speed
                if self.tick_timer >= self.tick_duration:
                    self.tick_timer -= self.tick_duration
                    self.tick_global += 1  # --- Incrementa tick global
                    self.virus.check_infections(self.mundo)

            # ==========================
            # Render escenari i marker
            # ==========================
            self.object.render()
            self.marker.render()
            light_pos = self.object.update_light_position()

            # Actualizar partículas de rastros a FPS de simulación
            self.virus.update_particles(self.delta_time * self.speed)

            self.virus.render(light_pos)

            # Mostrar grafo de waypoints si está activado
            if self.show_waypoints:
                self.waypoint_visualizer.render(self.mundo)

            # ==========================
            # Actualitzar persones amb col·lisions
            # ==========================
            for p in self.people:
                if self.simulando:
                    # old_pos = glm.vec3(p.position)  # guardem posició antiga
                    p.update(self.delta_time * self.speed)
                    # # Comprovem col·lisions amb altres persones
                    # for other in self.people:
                    #     if other is p: continue
                    #     if aabb_collision(p.position, p.bb_half, other.position, other.bb_half):
                    #         # Revertim posició si hi ha col·lisió
                    #         p.position = old_pos
                    #         p.m_model = glm.translate(glm.mat4(1.0), old_pos)
                    #         break
                # Render de la persona
                # p.render(self.object.shader, self.person_vao_tri, self.person_vao_line, light_pos)
                p.render(self.person_shader, self.person_vao_tri, self.person_vao_line, light_pos, self.person_texture)

            # ==========================
            # Render UI
            # ==========================
            self.ui_surface.fill((0,0,0,0))
            # num_infected = sum(1 for p in self.people if hasattr(p,'ring') and p.ring is not None)
            total_people = len(self.people)
            if total_people > 0:
                pass
                # self.infection_bar.render(self.ui_surface, num_infected, total_people)
            menu.render_menu(self.menu_surface)
            self.ui_surface.blit(self.menu_surface, (0,0))
            self._render_ui_overlay()

            render_graph(
                self.ctx,
                self.graph,
                np.array(self.camera.m_proj * self.camera.m_view, dtype="f4").T,
                point_size=12.0,
                overlay=False
            )
    
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
    DATA_PATH = os.path.join(ROOT_PATH,"DEMO2","data","salas")
    SCENE_PATH = os.path.join(ROOT_PATH,"DEMO2","Models","prova.obj")
    PERSON_PATH = os.path.join(ROOT_PATH,"DEMO2","Models","person.obj")
    TEXURE_PATH = os.path.join(ROOT_PATH,"DEMO2","Models","uni_mala.mtl")
    HORARIS_PATH = os.path.join(ROOT_PATH,"DEMO2","data","horaris")
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

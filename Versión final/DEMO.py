import os
os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
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
from simclock import SimClock
from scheduler import SimCalendar
from collections import defaultdict

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

import json

def export_pasillo_json_keep_ids(nodes, adj, out_path, *, sala_id=0, salida=666, alpha=2.0):
    """
    Exporta a un JSON con la estructura que has mostrado, manteniendo los IDs originales:

    {
      "id": <int>,
      "tipo": "pasillo",
      "entrada": null,
      "salida": 666,
      "pos": { "<id>": [x,y,z], ... },
      "con": { "<id>": ["<id_nb>", ...], ... },
      "rutas": {},
      "alpha": 2.0
    }
    """

    node_ids = sorted(nodes.keys())

    # pos: "<id>" -> [x,y,z]
    pos = {
        str(nid): [float(nodes[nid][0]), float(nodes[nid][1]), float(nodes[nid][2])]
        for nid in node_ids
    }

    # con: "<id>" -> ["<id_nb1>", ...] (solo vecinos existentes)
    con = {}
    for u in node_ids:
        nbs = []
        for v in adj.get(u, []):
            if v in nodes:  # filtrar ids no definidos (p.ej. 0 si no existe)
                nbs.append(str(v))
        con[str(u)] = sorted(nbs, key=lambda s: int(s))

    data = {
        "id": int(sala_id),
        "tipo": "pasillo",
        "entrada": None,
        "salida": int(salida),   # <- nodo 666 como salida (aunque no exista como waypoint)
        "pos": pos,
        "con": con,
        "rutas": {},
        "alpha": float(alpha),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return data

RAW_POINTS = r"""
	- (puerta) 340001: 54.55, 2.00, -92.55 - (340002)
    - (dentro) 340002: 54.55, 2.00, -93.80 - (340001,340003,340004,340005,340006,340007,340008,340009,340010,340011,340012,340013)
    - (asiento) 340003: 37.80, 6.05, -123.275 - (340004)
    - (asiento) 340004: 37.80, 6.05, -124.375 - (340005)
    - (asiento) 340005: 37.80, 6.05, -125.30 - (340006
    - (asiento) 340006: 37.80, 6.05, -126.25 - (340007)
    - (asiento) 340007: 37.80, 6.05, -127.20 - (340008)
    - (asiento) 340008: 37.80, 6.05, -128.10 - (340009)
    - (asiento) 340009: 37.80, 6.05, -129.15 - (340010)
    - (asiento) 340010: 37.80, 6.05, -130.05 - (340011)
    - (asiento) 340011: 37.80, 6.05, -130.95 - (340012)
    - (asiento) 340012: 37.80, 6.05, -131.90 - (340013)
    - (pasillo) 340013: 37.80, 6.05, -132.80 - (340014, 340001)
    - (pasillo) 340014: 36.15, 6.05, -132.80 - (340013, 340015)
    - (pasillo) 340015: 34.20, 6.05, -132.80 - (340014, 340016)
    - (pasillo) 340016: 32.20, 6.05, -132.80 - (340015, 340017)
    - (pasillo) 340017: 30.40, 6.05, -132.80 - (340016, 340018)
    - (asiento) 340027: 30.40, 6.05, -123.275 - (340026)
    - (asiento) 340026: 30.40, 6.05, -124.375 - (340027)
    - (asiento) 340025: 30.40, 6.05, -125.30 - (340026)
    - (asiento) 340024: 30.40, 6.05, -126.25 - (340025)
    - (asiento) 340023: 30.40, 6.05, -127.20 - (340024)
    - (asiento) 340022: 30.40, 6.05, -128.10 - (340023)
    - (asiento) 340021: 30.40, 6.05, -129.15 - (340022)
    - (asiento) 340020: 30.40, 6.05, -130.05 - (340021)
    - (asiento) 340019: 30.40, 6.05, -130.95 - (340020)
    - (asiento) 340018: 30.40, 6.05, -131.90 - (340019)
    - (asiento) 340028: 36.55, 6.05, -123.275 - (340029,340038,340039)
    - (asiento) 340029: 36.55, 6.05, -124.375 - (340030,340038,340039,340040)
    - (asiento) 340030: 36.55, 6.05, -125.30 - (340031,340039,340040,340041)
    - (asiento) 340031: 36.55, 6.05, -126.25 - (340032,340042,340041,340040)
    - (asiento) 340032: 36.55, 6.05, -127.20 - (340033,340042,340041,340043)
    - (asiento) 340033: 36.55, 6.05, -128.10 - (340034,340042,340044,340043)
    - (asiento) 340034: 36.55, 6.05, -129.15 - (340035,340045,340044,340043)
    - (asiento) 340035: 36.55, 6.05, -130.05 - (340036,340044,340045,340046)
    - (asiento) 340036: 36.55, 6.05, -130.95 - (340037,340045,340046,340047)
    - (asiento) 340037: 36.55, 6.05, -131.90 - (340014,340046,340047)
    - (asiento) 340038: 35.80, 6.05, -123.275 - (340039)
    - (asiento) 340039: 35.80, 6.05, -124.375 - (340040)
    - (asiento) 340040: 35.80, 6.05, -125.30 - (340041)
    - (asiento) 340041: 35.80, 6.05, -126.25 - (340042)
    - (asiento) 340042: 35.80, 6.05, -127.20 - (340043)
    - (asiento) 340043: 35.80, 6.05, -128.10 - (340044)
    - (asiento) 340044: 35.80, 6.05, -129.15 - (340045)
    - (asiento) 340045: 35.80, 6.05, -130.05 - (340046)
    - (asiento) 340046: 35.80, 6.05, -130.95 - (340047)
    - (asiento) 340047: 35.80, 6.05, -131.90 - (340014)
    - (asiento) 340048: 34.55, 6.05, -123.275 - (340049,340058,340059)
    - (asiento) 340049: 34.55, 6.05, -124.375 - (340050,340058,340059,340060)
    - (asiento) 340050: 34.55, 6.05, -125.30 - (340051,340059,340060,340061)
    - (asiento) 340051: 34.55, 6.05, -126.25 - (340052,340062,340061,340060)
    - (asiento) 340052: 34.55, 6.05, -127.20 - (340053,340062,340061,340063)
    - (asiento) 340053: 34.55, 6.05, -128.10 - (340054,340062,340064,340063)
    - (asiento) 340054: 34.55, 6.05, -129.15 - (340055,340065,340064,340063)
    - (asiento) 340055: 34.55, 6.05, -130.05 - (340056,340064,340065,340066)
    - (asiento) 340056: 34.55, 6.05, -130.95 - (340057,340065,340066,340067)
    - (asiento) 340057: 34.55, 6.05, -131.90 - (340015,340066,340067)
    - (asiento) 340058: 33.80, 6.05, -123.275 - (340059)
    - (asiento) 340059: 33.80, 6.05, -124.375 - (340060)
    - (asiento) 340060: 33.80, 6.05, -125.30 - (340061)
    - (asiento) 340061: 33.80, 6.05, -126.25 - (340062)
    - (asiento) 340062: 33.80, 6.05, -127.20 - (340063)
    - (asiento) 340063: 33.80, 6.05, -128.10 - (340064)
    - (asiento) 340064: 33.80, 6.05, -129.15 - (340065)
    - (asiento) 340065: 33.80, 6.05, -130.05 - (340066)
    - (asiento) 340066: 33.80, 6.05, -130.95 - (340067)
    - (asiento) 340067: 33.80, 6.05, -131.90 - (340015)
    - (asiento) 340068: 32.50, 6.05, -123.275 - (340069,340078,340079)
    - (asiento) 340069: 32.50, 6.05, -124.375 - (340070,340078,340079,340080)
    - (asiento) 340070: 32.50, 6.05, -125.30 - (340071,340079,340080,340081)
    - (asiento) 340071: 32.50, 6.05, -126.25 - (340072,340082,340081,340080)
    - (asiento) 340072: 32.50, 6.05, -127.20 - (340073,340082,340081,340083)
    - (asiento) 340073: 32.50, 6.05, -128.10 - (340074,340082,340084,340083)
    - (asiento) 340074: 32.50, 6.05, -129.15 - (340075,340085,340084,340083)
    - (asiento) 340075: 32.50, 6.05, -130.05 - (340076,340084,340085,340086)
    - (asiento) 340076: 32.50, 6.05, -130.95 - (340077,340085,340086,340087)
    - (asiento) 340077: 32.50, 6.05, -131.90 - (340016,340086,340087)
    - (asiento) 340078: 31.80, 6.05, -123.275 - (340079)
    - (asiento) 340079: 31.80, 6.05, -124.375 - (340080)
    - (asiento) 340080: 31.80, 6.05, -125.30 - (340081)
    - (asiento) 340081: 31.80, 6.05, -126.25 - (340082)
    - (asiento) 340082: 31.80, 6.05, -127.20 - (340083)
    - (asiento) 340083: 31.80, 6.05, -128.10 - (340084)
    - (asiento) 340084: 31.80, 6.05, -129.15 - (340085)
    - (asiento) 340085: 31.80, 6.05, -130.05 - (340086)
    - (asiento) 340086: 31.80, 6.05, -130.95 - (340087)
    - (asiento) 340087: 31.80, 6.05, -131.90 - (340016)
    """

nodes, edges, adj = parse_graph_from_text(RAW_POINTS)
export_pasillo_json_keep_ids(nodes, adj, "pasillo_grafo.json", salida=666, alpha=2.0)

# =====================================================
#                    TIEMPO
# =====================================================

def weekday_name(i):
    return ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"][i]

def slot_to_hhmm(slot_index):
    # slot 0 => 08:00, slot 1 => 08:30 ...
    minutes = 8 * 60 + slot_index * 30
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


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

def repartir_por_grupos(data: dict, total_por_grupo: int, sep: str = "-") -> dict:
    """
    data: dict cargado del JSON (keys tipo 'GED1-81', 'GED1-82', ...)
    total_por_grupo: entero a repartir dentro de cada grupo (p.ej. 50)
    sep: separador para definir el grupo (por defecto '-')

    Devuelve: dict con mismas keys y valor int asignado.
    Ej: si GED1 tiene 2 keys y total_por_grupo=50 -> cada una recibe 25.
    """
    grupos = defaultdict(list)

    for k in data.keys():
        grupo = k.split(sep, 1)[0]   # 'GED1-81' -> 'GED1'
        grupos[grupo].append(k)

    out = {}
    for grupo, keys in grupos.items():
        n = len(keys)
        if n == 0:
            continue
        asignacion = total_por_grupo // n  # redondeo hacia abajo
        for k in keys:
            out[k] = asignacion

    return out

# =====================================================
#                    MOTOR GRÀFIC
# =====================================================
class MotorGrafico:
    def __init__(self, scene_path, person_path, facultad, win_size=(1820, 980), parets_path=None):
        pg.init()
        pg.display.set_caption("Epidemiological Simulator - WASD moverte, TAB soltar ratón")
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

        # Multi Vista
        self.menu_width = 400   # anchura del menú
        self.menu_surface = pg.Surface((self.menu_width, self.WIN_SIZE[1]), pg.SRCALPHA)

        # Camara
        self.camera = Camera(self)

        # Mundo
        self.mundo = facultad

        # Marker
        self.marker = Marker(self.ctx, self.camera)
        self.marker.position = glm.vec3(37.80, 6.05, -124.60)

        # InfectionBar
        self.infection_bar = InfectionBar(self.WIN_SIZE[0], self.WIN_SIZE[1])
        self.infection_bar.bar_x = self.menu_width + 20
        self.infection_bar.bar_width = self.WIN_SIZE[0] - self.menu_width - 40
        self.ui_surface = pg.Surface(self.WIN_SIZE, pg.SRCALPHA)

        # Clock y control de FPS
        self.clock = pg.time.Clock()
        self.last_time = time.time()
        self.delta_time = 0.016
        self.frame_count = 0
        self.fps = 0

        self.show_bboxes = False

        # SimClock (tiempo del mundo)
        self.day_sim_seconds = 0.05 * 60  # cámbialo a 5*60 o 30*60 cuando quieras
        self.sim_clock = SimClock(day_sim_seconds=self.day_sim_seconds, speed_mult=self.speed)
        self.exposure_scale = 0       # pasillo por defecto
        self.class_exposure_scale = 6.0 # ejemplo: 20 min sim = 120 min real -> 6x
        self.calendar = SimCalendar(slot_sim_minutes=5, start_weekday=0)

        # --- Tiempo "académico" para UI (slots de 30 min) ---
        self.slot_sim_minutes = 5          # 1 slot (30 min real) dura 5 min simulados visibles
        self.slot_sim_seconds = self.slot_sim_minutes * 60.0
        self.slots_per_day = 28            # 08:00 -> 22:00 son 14h -> 28 slots
        self.sim_weekday = 0               # 0=Lun ... 4=Vie
        self.sim_time_in_day = 0.0         # segundos simulados desde las 08:00


        # Virus
        self.tick_duration = 1
        self.tick_timer = 0.0
        self.tick_global = 0  # --- Contador global de ticks ---
        self.infection_probability = 0.02
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

        # PARETS (obj opcional con su .mtl)
        self.parets_object = None
        self.show_parets = False
        if parets_path:
            parets_data, parets_bbox, parets_texture = load_obj(parets_path)
            self.parets_object = Escenario(self.ctx, self.camera, parets_data, parets_bbox, parets_texture)
            self.parets_object.app = self
            self.show_parets = True

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
        reparto = repartir_por_grupos(self.people_type, total_por_grupo=20)
        for i in reparto:
            for j in range(reparto[i]):
                p = self.create_person(grupo=i)

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
    def create_person(self, schedule=[], spawn='pasillo', grupo=None):
        # persona = Person(self.ctx, self.camera, self.p_tri_data, self.p_normals, self.p_line_data, self.mundo, schedule, spawn)
        persona = Person(self, self.ctx, self.camera, self.p_data, self.mundo, schedule, spawn, group=grupo)
        persona.infection_tick = None  # --- Manté el tick d’infecció ---
        self.people.append(persona)
        return persona
    
    def get_active_and_next(self, day_sessions, slot_now):
        active = None
        next_sess = None
        for s in day_sessions:
            if s["start_slot"] <= slot_now < s["end_slot"]:
                active = s
                break
            if s["start_slot"] > slot_now:
                next_sess = s
                break
        return active, next_sess

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
            dt_wall = self.clock.tick(60) / 1000.0
            if dt_wall <= 0: dt_wall = 1e-6
            self.delta_time = dt_wall  # dt_wall para cámara/UI

            dt_sim = dt_wall * self.speed

            if self.simulando:
                self.calendar.step(dt_sim)
                self.sim_time_in_day += dt_sim
                day_len = self.slots_per_day * self.slot_sim_seconds

                while self.sim_time_in_day >= day_len:
                    self.sim_time_in_day -= day_len
                    self.sim_weekday = (self.sim_weekday + 1) % 5  # saltamos finde

            dt_real_eq = dt_sim * self.exposure_scale

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
                    elif e.key == pg.K_h:
                        self.virus.debug_grid = not self.virus.debug_grid
                        print(f"Debug grid: {self.virus.debug_grid}")
                    elif e.key == pg.K_v:
                        self.show_parets = not self.show_parets
                        print(f"Parets: {'ON' if self.show_parets else 'OFF'}")
                    elif e.key == pg.K_F5:
                        # Guarda posición + target (hacia donde mira) y lo imprime para copiar/pegar
                        self.camera.capture_current_preset(distance=1.0, append=True)
                    elif e.key == pg.K_1:
                        self.speed = 1.0
                        print("Velocidad x1")
                    elif e.key == pg.K_2:
                        self.speed = 3.0
                        print("Velocidad x3")
                    elif e.key == pg.K_3:
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
                day_key = self.calendar.weekday_key()
                slot_now = self.calendar.current_slot()
                sesions = {}

                for g in self.people_type:
                    day_sessions = self.people_type[g][day_key]
                    sesions[g] = self.get_active_and_next(day_sessions, slot_now)

                # self.tiempo_persona += dt
                # if self.tiempo_persona >= self.intervalo_spawn:
                #     selection = random.choice(rooms)
                #     p = self.create_person([selection])
                #     if clean_rooms[selection] == 0:
                #         self.virus.infectar(p)
                #     clean_rooms[selection] += 1
                #     self.tiempo_persona = 0.0

                # Tick virus
                self.tick_timer += dt_sim
                while self.tick_timer >= self.tick_duration:
                    self.tick_timer -= self.tick_duration
                    self.tick_global += 1
                    self.virus.check_infections(self.mundo)

            # ==========================
            # Render escenari i marker
            # ==========================
            light_pos = self.object.update_light_position()
            self.object.render(light_pos=light_pos)
            if self.show_parets and self.parets_object is not None:
                self.parets_object.render(light_pos=light_pos)
            self.marker.render()

            # Actualizar partículas de rastros a FPS de simulación
            self.virus.update_particles(dt_sim)

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
                    # p.update(dt_sim)
                    # # Comprovem col·lisions amb altres persones
                    # for other in self.people:
                    #     if other is p: continue
                    #     if aabb_collision(p.position, p.bb_half, other.position, other.bb_half):
                    #         # Revertim posició si hi ha col·lisió
                    #         p.position = old_pos
                    #         p.m_model = glm.translate(glm.mat4(1.0), old_pos)
                    #         break
                    group = getattr(p, "group_id", None)
                    if not group: 
                        p.update(dt_sim)  
                    else:
                        # asegúrate que está ordenada una vez al cargar (mejor)
                        active, next_sess = sesions[group]

                        # 1) si hay clase activa, fuerza objetivo a esa clase (si no está ya)
                        if active:
                            p.preclass_plan = None  # ya no aplica
                            p.schedule = [active["room"]]  # o método "go_to_room"
                        else:
                            # 2) si no hay clase: ejecutar plan si toca
                            started = p.maybe_execute_preclass_plan(self.calendar)

                            # 3) si no hay plan y la próxima empieza en el siguiente slot, planificar
                            if (not started) and (p.preclass_plan is None) and next_sess and (slot_now + 1 == next_sess["start_slot"]):
                                # aquí deberías asegurarte de que p está "libre" (no en clase)
                                # según tu estado actual, lo más simple: si no active, asumimos libre
                                p.plan_preclass_departure(self.calendar, next_sess["room"])

                            # 4) si no hay nada, wander
                            if (p.preclass_plan is None) and (not next_sess):
                                # no quedan clases: ir salida / wander
                                p.schedule = ["pasillo"]  # o salida

                        p.update(dt_sim)
                # Render de la persona
                # p.render(self.object.shader, self.person_vao_tri, self.person_vao_line, light_pos)
                if p.present:
                    p.render(self.person_shader, self.person_vao_tri, self.person_vao_line, light_pos, self.person_texture)

            # ==========================
            # Render UI
            # ==========================
            self.ui_surface.fill((0,0,0,0))
            num_infected = sum(1 for p in self.people if hasattr(p,'ring') and p.ring is not None)
            total_people = len(self.people)
            if total_people > 0:
                pass
                # self.infection_bar.render(self.ui_surface, num_infected, total_people)
            self.infection_bar.render(self.ui_surface, num_infected, total_people)
            # Render menú (sliders)
            menu.render_menu(self.menu_surface)

            # Construir stats (rápido y robusto)
            total = len(self.people)
            present = sum(1 for p in self.people if getattr(p, "present", False))
            infected = sum(1 for p in self.people if getattr(p, "ring", None) is not None)
            moving = sum(1 for p in self.people if getattr(p, "en_movimiento", False))
            seated = sum(1 for p in self.people if getattr(p, "sentado", False))
            slot_now = int(self.sim_time_in_day // self.slot_sim_seconds)
            sim_day = weekday_name(self.sim_weekday)
            sim_hhmm = slot_to_hhmm(slot_now)

            # Top salas por infección (si existe algún atributo numérico)
            top_rooms = []
            for name, sala in self.mundo.items():
                # cambia "contaminacion" por tu nombre real si es distinto
                v = getattr(sala, "contaminacion", None)
                if isinstance(v, (int, float)):
                    top_rooms.append((name, float(v)))
            top_rooms.sort(key=lambda x: x[1], reverse=True)

            stats = {
                "total": total,
                "present": present,
                "infected": infected,
                "healthy": max(0, total - infected),
                "moving": moving,
                "seated": seated,
                "speed": self.speed,
                "sim_day": sim_day,
                "sim_time": sim_hhmm,
                # si luego tienes calendar, aquí podrás meter hora real del lunes etc.
                # "sim_time": ...
                "top_rooms": top_rooms,
            }

            # Pintar panel debajo de sliders
            y0 = menu.get_content_bottom() if hasattr(menu, "get_content_bottom") else 260
            menu.render_stats_panel(self.menu_surface, y=y0 + 10, stats=stats, width=self.menu_width, height=260)

            # Blit a UI
            self.ui_surface.blit(self.menu_surface, (0,0))
            self._render_ui_overlay()

            # render_graph(
            #     self.ctx,
            #     self.graph,
            #     np.array(self.camera.m_proj * self.camera.m_view, dtype="f4").T,
            #     point_size=12.0,
            #     overlay=False
            # )
    
            # ==========================
            # FPS
            # ==========================
            self.frame_count += 1
            if time.time()-self.last_time>=1.0:
                self.fps = self.frame_count/(time.time()-self.last_time)
                self.frame_count=0
                self.last_time=time.time()
                pg.display.set_caption(f"Epidemiological Simulator - FPS: {self.fps:.1f} - WASD moverte, TAB soltar ratón")

            # si debug grid activat
            if self.virus.debug_grid:
                mvp = np.array(self.camera.m_proj * self.camera.m_view, dtype="f4").T
                self.virus.render_debug_grid(mvp=mvp)

            pg.display.flip()

# =====================================================
#                     MAIN
# =====================================================
if __name__ == "__main__":
    ROOT_PATH = os.getcwd()
    DATA_PATH = os.path.join(ROOT_PATH,"Versión final","data","salas")
    SCENE_PATH = os.path.join(ROOT_PATH,"Versión final","Models","DEF.obj")
    PERSON_PATH = os.path.join(ROOT_PATH,"Versión final","Models","person.obj")
    TEXURE_PATH = os.path.join(ROOT_PATH,"Versión final","Models","DEF.mtl")
    HORARIS_PATH = os.path.join(ROOT_PATH,"Versión final","data","horaris")
    PARETS_PATH = os.path.join(ROOT_PATH, "Versión final", "Models", "parets.obj")
    PARETS_TEXTURE_PATH = os.path.join(ROOT_PATH, "Versión final", "Models", "parets.mtl")

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
        if tipo=="clase": 
            print(data.get("id"))
            sala = Clase.from_json_struct(data)
        elif tipo=="pasillo": sala = Pasillo.from_json_struct(data)
        else: sala = Sala.from_json_struct(data)
        facultad[nombre] = sala
        print(f"[MAIN] Cargada sala '{nombre}' ({tipo}) con {len(sala.waypoints)} waypoints.")

    print(f"[MAIN] Total salas: {len(facultad)}\n")

    motor = MotorGrafico(SCENE_PATH, PERSON_PATH, facultad, parets_path=PARETS_PATH)
    motor.start()
    motor.run()

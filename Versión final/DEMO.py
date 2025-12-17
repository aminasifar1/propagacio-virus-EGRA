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
from simclock import SimClock
from scheduler import SimCalendar


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
	- (puerta) 0: -0.50, -0.10, -15.15 - (1,2,666
	- (pitagoras) 1: 4.90, -0.10, -15.15 - (0,2,
	- (escalera principal) 2: 7.70, -0.10, -13.65 - (0,1,3,5
	- (pasillo izquierda abajo) 3: 10.60, -0.10, -13.65 - (2,6
	- (pasillo medio abajo) 4: 13.90, -0.10, -13.65 - (3,6
	- (principio escalera) 5: 7.70, -0.10, -14.80 - (2,39)
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
	- (abajo escaleras Q3) 26   : 4.85, 2.00, -89.10 - (25,27)
	- (inicio escaleras Q3) 27: 4.85, 2.00, -86.20 - (26,
	- (entrada rampa Q2/0) 28: 10.60, 1.45, -60.00 - (14,15,16,29)
	- (abajo escaleras Q2) 29: 4.85, 1.45, -60.00 - (28,30)
	- (inicio escaleras Q2) 30: 4.85, 1.45, -57.20 - (29,
	- (final Q1/0) 31: 74.30, 0.45, -34.10 - (12,32
	- (incio escaleras final Q1/0) 32: 76.15, 0.45, -38.95 - (31,
	- (final Q2/0) 33: 74.30, 1.45, -62.70 - (15,34
	- (incio escaleras final Q2/0) 34: 76.15, 1.45, -68.15 - (33,
	- (final Q3/0) 35: 74.30, 2.00, -91.85 - (18,36
	- (incio escaleras final Q3/0) 36: 76.15, 2.00, -97.70 - (35,
	- (final Q4/0) 37: 74.30, 2.60, -121.05 - (21,38
	- (incio escaleras final Q4/0) 38: 76.15, 2.60, -126.80 - (37,
	- (medio 1 escalera) 39: 7.70, 2.00, -20.70 - (5,40)
	- (medio 2 escalera) 40: 7.70, 2.00, -23.50 - (39,41)
	- (final escalera) 41: 7.70, 4.30, -29.90 - (40,42)
	- (inicio pasillo 1) 42: 10.60, 4.30, -31.25 - (41,43,44)
	- (inicio Q1/1) 43: 13.90, 4.30, -34.10 - (42
	- (principio segunda rampa 1) 44: 10.60, 4.30, -46.95 - (44,45)
	- (final segunda rampa 1) 45: 10.60, 4.85, -58.60 - (44,46,47)
	- (entrada rampa Q2/1) 46: 10.60, 4.85, -60.00 - (45,47
	- (inicio Q2/1) 47: 13.90, 4.85, -62.70 - (45,46
	- (inicio tercera rampa 1) 48: 10.60, 4.85, -76.25 - (46,49
	- (final tercera rampa 1) 49: 10.60, 5.40, -87.80 - (48,50,51
	- (entrada rampa Q3/1) 50: 10.60, 5.40, -89.10 - (49,51,52
	- (inicio Q3/1) 51: 13.90, 5.40, -91.85 - (50,49
	- (inicio cuarta rampa 1) 52: 10.60, 5.40, -104.30 - (50,53
	- (final cuarta rama 1) 53: 10.60, 6.05, -117.00 - (52,54,55
	- (entrada rampa Q4/1) 54:10.60, 6.05, -118.25 - (53,55
	- (incio Q4/1) 55: 13.90, 6.05, -121.05 - (54,
	- (final Q1/1) 56: 74.30, 4.30, -34.10 - (43,
	- (final Q2/1) 57: 74.30, 4.85, -62.70 - (47,
	- (final Q3/1) 58: 74.30, 5.40, -91.85 - (51,
	- (final Q4/1) 59: 74.30, 6.05, -121.05 - (55
	- (medio escaleras Q4) 60: 4.85, 3.25, -53.00 - (61,30)
	- (medio 1 escaleras Q2) 61: 6.50, 3.25, -52.15 - (60,62)
	- (medio 2 escaleras Q2) 62: 8.40, 3.25, -52.95 - (61,63)
	- (final escaleras Q2) 63: 8.40, 4.85, -58.05 - (62,46)
	- (final escaleras Q3) 64: 8.40, 5.45, -87.25 - (50,65)
	- (medio 2 escaleras Q3) 65: 8.40, 3.80, -82.10 - (64,66)
	- (medio 1 escaleras Q3) 66: 6.50, 3.80, -81.40 - (65,67)
	- (medio escaleras Q3) 67: 4.85, 3.80, -82.10 - (66,27)
	- (medio escaleras Q4) 68: 4.85, 4.45, -111.30 - (24,69)
	- (medio 1 escaleras Q4) 69: 6.50, 4.45, -110.350 - (68,70)
	- (medio 2 escaleras Q4) 70: 8.40, 4.45, -111.30 - (69,71)
	- (final escaleras Q4) 71: 8.40, 6.05, -116.45 - (70,54)
	- (medio escaleras final Q1) 72: 76.15, 2.65, -44.30 - (73,32)
	- (medio 1 escaleras final Q1) 73: 75.15, 2.65, -44.80 - (72,74)
	- (medio 2 escaleras final Q1) 74: 74.15, 2.65, -44.30 - (73,75)
	- (final escaleras final Q1) 75: 74.15, 4.30, -39.15 - (74,56)
	- (final escaleras final Q2) 76: 74.15, 4.85, -67.45 - (77,57)
	- (medio 2 escaleras final Q2) 77: 74.15, 3.20, -72.60 - (76,78)
	- (medio 1 escaleras final Q2) 78: 75.15, 3.20, -73.10 - (77,79)
	- (medio escaleras final Q2) 79: 76.15, 3.20, -72.60 - (78,34)
	- (final escaleras final Q3) 80: 74.15, 5.40, -96.95 - (81,58)
	- (medio 2 escaleras final Q3) 81: 74.15, 3.75, -102.10 - (80,82)
	- (medio 1 escaleras final Q3) 82: 75.15, 3.75, -102.60 - (81,83)
	- (medio escaleras final Q3) 83: 76.15, 3.75, -102.10 - (82,36)
	- (medio escaleras final Q4) 84: 76.15, 4.40, -131.25 - (85,38)
	- (medio 1 escaleras final Q4) 85: 75.15, 4.40, -131.75 - (84,86)
	- (medio 2 escaleras final Q4) 86: 74.15, 4.40, -131.25 - (85,87)
	- (final escaleras final Q4) 87: 74.15, 6.10, -126.15 - (86,59)
	- (inicio pasillo para Q5-Q6) 89: 79.05, 4.30, -34.10 - (56,90)
	- (punto entrada pasillo Q5) 90: 79.05, 4.40, -44.70 - (89,91)
	- (punto pasillo Q5-Q6 1a rampa inicio) 91: 79.05, 4.40, -47.90 - (90,92)
	- (punto pasillo Q5-Q6 1a rampa fin) 92: 79.05, 5.00, -58.45 - (91,93)
	- (pasillo 2 despues del punto final Q5-Q6) 93: 79.05, 5.00, -62.75 - (92,94)
	- (punto pasillo Q5-Q6 2a rampa inicio) 94: 79.05, 5.00, -76.65 - (93,95)
	- (punto pasillo entrada Q6) 95: 79.05, 5.45, -84.40 - (94,96)
	- (punto pasillo Q5-Q6 2a rampa final) 96: 79.05, 5.55, -87.60 - (95,97)
	- (punto pasillo despues rampa mitad) 97: 79.05, 5.55, -96.45 - (96,98)
	- (punto rampa 3a inicio Q5-Q6) 98: 79.05, 5.55, -104.80 - (97,99)
	- (punto rampa 3a final Q5-Q6) 99: 79.05, 6.20, -116.80 - (98,100)
	- (punto final pasillo Q5-Q6) 100: 79.05, 6.20, -121.10 - (99,59)
	- (final Q5/1) 101: 145.90, 4.40, -44.85 - (102)
	- (inicio Q5/1) 102: 92.20, 4.40, -44.85 - (90,
	- (inicio Q5/0) 103: 92.20, 0.95, -44.85 - ()
	- (inicio Q5/2) 104: 92.05, 7.80, -44.85  - ()
	- (inici escales Q5/2 arriba) 105: 148.40, 7.80, -50.45 - (112,
	- (entemig escales Q5/2) 106: 148.40, 6.10, -55.10 - (105,107)
	- (entremig escales q5/2 2) 107: 149.05, 6.10, -55.75 - (106,108)
	- (entremig escales q5/2 3) 108: 149.85, 6.10, -55.80 - (107,109)
	- (entremig escales q5/2 4) 109: 150.65, 6.10, -55.00 - (108,110)
	- (final escales q5/1) 110: 150.65, 4.30, -51.00 - (109,111)
	- (inici escales q5/1) 111: 150.65, 4.30, -44.85 - (110,101)
	- (final Q5/2) 112: 148.40, 7.80, -44.85 - (104,
    - (inicio escaleras derecha Q5/2) 113: 90.40, 7.80, -45.00 - (104,114)
	- (escaleras derecha Q5/2 2do punto) 114: 90.40, 7.80, -50.20 - (113,115)
	- (escaleras derecha Q5/2 3er punto) 115: 90.40, 6.10, -55.05 - (114,116)
	- (escaleras derecha Q5/2 4to punto) 116: 90.85, 6.10, -55.80 - (115,117)
	- (escaleras derecha Q5/2 5to punto) 117: 91.65, 6.10, -55.80 - (116,118)
	- (escaleras derecha Q5/2 6to punto) 118: 92.20, 6.10, -55.35 - (117,119)
	- (final escaleras derecha Q5/2 abajo) 119: 92.20, 4.40, -51.10 - (118,102)
	- (inicio escaleras derecha Q2/1) 120: 90.30, 4.35, -49.95 - (102,121)
	- (escaleras derecha Q2/1 2do punto) 121: 90.30, 2.70, -55.10 - (120,122)
	- (escaleras derecha Q2/1 3er punto) 122: 90.75, 2.70, -55.80 - (121,123)
	- (escaleras derecha Q2/1 4to punto) 123: 91.85, 2.70, -55.80 - (122,124)
	- (escaleras derecha Q2/1 5to punto) 124: 92.30, 2.70, -55.05 - (123,125)
	- (final escaleras derecha Q2/1) 125: 92.30, 0.90, -51.00 - (124,103)
    - (inicio entrada mini rampa Q6/1) 126: 84.25, 5.30, -84.40 - (95,127)
	- (final mini rampa Q6/1) 127: 88.20, 5.25, -84.00 - (126,128)
	- (punto entrada inicio Q6/1) 128: 91.05, 5.25, -84.00 - (127,129, 137,138)
	- (punto final Q6/1) 129: 149.35, 5.25, -84.00 - (128)
	- (punto final Q6/2) 130: 149.35, 8.70, -84.00 - (131)
	- (punto inicio Q6/2) 131: 91.05, 8.70, -84.00 - (130,132)
    - (inicio escaleras derecha Q6/2 arriba) 132: 90.30, 8.70, -89.70 - (131,133)
	- (escaleras Q6/2 2do punto) 133: 90.30, 7.05, -94.55 - (132,134)
	- (escaleras Q6/2 3er punto) 134: 90.70, 7.05, -95.55 - (133,135)
	- (escaleras Q6/2 4to punto) 135: 91.85, 7.05, -95.55 - (134,136)
	- (escaleras Q6/2 5to punto) 136: 92.30, 7.05, -94.25 - (135,137)
	- (final escalera Q6/2 hacia Q6/1) 137: 92.30, 5.25, -90.65 - (136,138,128)
    - (inicio escaleras derecha Q6/1 arriba) 138: 90.25, 5.25, -89.95 - (137,139,128)
	- (escaleras Q6/1 2do punto) 139: 90.25, 3.60, -94.50 - (138,140)
	- (escaleras Q6/1 3er punto) 140: 90.80, 3.60, -95.50 - (139,141)
	- (escaleras Q6/1 4to punto) 141: 91.75, 3.65, -95.50 - (140,142)
	- (escaleras Q6/1 5to punto) 142: 92.40, 3.60, -94.40 - (141,143)
	- (final escaleras Q6/1 hacia Q6/0) 143: 92.40, 1.80, -90.65 - (142,144)
	- (inicio Q6/0) 144: 92.40, 1.80, -84.20 - (143)
    - (inicio escaleras izquierda Q6/2 final pasillo) 145: 148.65, 8.70, -90.05 - (130,146)
	- (escaleras izquierda Q6/2 2do punto) 146: 148.65, 7.05, -94.60 - (145,147)
	- (escaleras izquierda Q6/2 3er punto) 147: 149.10, 7.05, -95.35 - (146,148)
	- (escaleras izquierda Q6/2 4to punto) 148: 150.05, 7.05, -95.35 - (147,149)
	- (escaleras izquierda Q6/2 5to punto) 149: 150.65, 7.05, -94.25 - (148,150)
	- (final escaleras izquierda Q6/1) 150: 150.65, 5.25, -90.50 - (149,129)
	- (salida) 666: -0.50, -0.10, -18.70 - (0)
    """

nodes, edges, adj = parse_graph_from_text(RAW_POINTS)
export_pasillo_json_keep_ids(nodes, adj, "pasillo_grafo.json", salida=666, alpha=2.0)

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
        self.marker.position = glm.vec3(148.65, 8.65, -90.05)

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

        # SimClock (tiempo del mundo)
        self.day_sim_seconds = 5 * 60  # cámbialo a 5*60 o 30*60 cuando quieras
        self.sim_clock = SimClock(day_sim_seconds=self.day_sim_seconds, speed_mult=self.speed)
        self.exposure_scale = 1.0       # pasillo por defecto
        self.class_exposure_scale = 6.0 # ejemplo: 20 min sim = 120 min real -> 6x
        self.calendar = SimCalendar(slot_sim_minutes=5, start_weekday=0)

        # Virus
        self.tick_duration = 1
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
            for j in range(1):
                p = self.create_person([i])

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
                        print(f"🧩 Debug grid: {self.virus.debug_grid}")
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

    motor = MotorGrafico(SCENE_PATH, PERSON_PATH, facultad)
    motor.start()
    motor.run()

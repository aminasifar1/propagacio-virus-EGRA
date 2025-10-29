from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import json
import math
import random
import glm
import os

Vec3 = glm.vec3

class Waypoint:
    """
    Son los nodos por los que se moveran las personas
    """

    def __init__(self, position):
        """
        position: tuple (x, y, z)
        conexiones: lista de claves de otros waypoints conectados
        """
        self.position = glm.vec3(*position)
        self.conexiones = []

    def conectar(self, otro):
        """Crea una conexion entre dos waypoints"""
        self.conexiones.append(otro)
        otro.conexiones.append(self)

    def __repr__(self):
        return f"Waypoint(pos={tuple(self.position)}, con={len(self.conexiones)})"
    

class Sala:
    """
    Base común para cualquier sala/zona con un grafo de waypoints.
    - Mantiene un diccionario {id_wp: Waypoint} para indexado/serialización.
    - Las conexiones en Waypoint son referencias (no claves).
    """

    def __init__(self, id_sala: int, tipo: str):
        self.id_sala = id_sala
        self.tipo = tipo
        self.waypoints: Dict[int, Waypoint] = {}

        # Si la sala tiene “puerta” (dos nodos especiales):
        # entrada = nodo fuera (en pasillo); salida = nodo dentro (en la sala)
        self.entrada_id: Optional[int] = None
        self.salida_id: Optional[int] = None

    # ---------- Gestión de nodos ----------
    def add_waypoint(self, id_wp: int, position: Tuple[float, float, float]) -> None:
        if id_wp in self.waypoints:
            raise ValueError(f"[Sala {self.id_sala}] Waypoint {id_wp} ya existe.")
        self.waypoints[id_wp] = Waypoint(position)

    def connect_ids(self, a_id: int, b_id: int) -> None:
        a = self.waypoints.get(a_id); b = self.waypoints.get(b_id)
        if a is None or b is None:
            raise KeyError(f"[Sala {self.id_sala}] Waypoint inexistente {a_id} o {b_id}.")
        # IMPORTANTE: tu Waypoint.conectar NO evita duplicados; no llamarla 2 veces
        a.conectar(b)

    def get_wp(self, id_wp: int) -> Waypoint:
        return self.waypoints[id_wp]

    # ---------- Pathfinding genérico (A*) ----------
    # Reutilizable. Las subclases pueden:
    # - Pasar un filtro de “traversable” (p.ej. ignorar asientos ocupados)
    # - O directamente sobrescribir get_path()
    def _a_star(
        self,
        start_id: int,
        goal_id: int,
        traversable: Optional[Callable[[int], bool]] = None,
    ) -> List[int]:
        """
        A* sobre el grafo de esta sala. Devuelve lista de ids (ruta).
        traversable(id) -> False para nodos bloqueados. El goal siempre es permitido.
        """
        if start_id not in self.waypoints or goal_id not in self.waypoints:
            return []

        if traversable is None:
            traversable = lambda _id: True

        start = start_id
        goal = goal_id

        open_set = {start}
        came_from: Dict[int, int] = {}
        g: Dict[int, float] = {start: 0.0}
        f: Dict[int, float] = {start: self._h(start, goal)}

        while open_set:
            current = min(open_set, key=lambda nid: f.get(nid, float("inf")))
            if current == goal:
                return self._reconstruct_path(came_from, current)

            open_set.remove(current)
            current_wp = self.waypoints[current]

            for neigh in current_wp.conexiones:
                # necesitamos el id del vecino (reverse lookup)
                neigh_id = self._id_of(neigh)
                if neigh_id is None:
                    continue

                # El goal siempre es permitible; el resto pasa por “traversable”
                if neigh_id != goal and not traversable(neigh_id):
                    continue

                tentative_g = g[current] + glm.distance(current_wp.position, neigh.position)
                if tentative_g < g.get(neigh_id, float("inf")):
                    came_from[neigh_id] = current
                    g[neigh_id] = tentative_g
                    f[neigh_id] = tentative_g + self._h(neigh_id, goal)
                    open_set.add(neigh_id)

        return []  # no hay ruta

    def _h(self, a_id: int, b_id: int) -> float:
        A = self.waypoints[a_id].position
        B = self.waypoints[b_id].position
        return glm.distance(A, B)

    def _reconstruct_path(self, came_from: Dict[int, int], current: int) -> List[int]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _id_of(self, wp_obj: Waypoint) -> Optional[int]:
        # Búsqueda lineal: para grafos pequeños/medianos es suficiente.
        # Si la sala crece mucho, se puede mantener un dict {Waypoint: id}
        for _id, _wp in self.waypoints.items():
            if _wp is wp_obj:
                return _id
        return None

    # Interfaz pública; las subclases pueden sobrescribir
    def get_path(self, start_id: int, goal_id: int) -> List[int]:
        return self._a_star(start_id, goal_id)

    # ---------- Utilidades puerta ----------
    def set_puerta(self, entrada_id: int, salida_id: int) -> None:
        self.entrada_id = entrada_id
        self.salida_id = salida_id

    # ---------- (Opcional) serialización simple de la sala ----------
    def to_json_struct(self) -> dict:
        # Exporta posiciones y conexiones como ids (para guardar en disco)
        pos = {str(i): (float(wp.position.x), float(wp.position.y), float(wp.position.z))
               for i, wp in self.waypoints.items()}
        con = {str(i): [str(self._id_of(n)) for n in wp.conexiones]
               for i, wp in self.waypoints.items()}
        return {
            "id": self.id_sala,
            "tipo": self.tipo,
            "entrada": self.entrada_id,
            "salida": self.salida_id,
            "pos": pos,
            "con": con,
        }

    @classmethod
    def from_json_struct(cls, data: dict) -> "Sala":
        sala = cls(data["id"], data["tipo"])
        # Crear nodos
        for sid, p in data["pos"].items():
            sala.add_waypoint(int(sid), tuple(p))
        # Conectar (una sola pasada, respetando tu conectar sin dedupe)
        for sid, vecinos in data["con"].items():
            a = int(sid)
            for svec in vecinos:
                b = int(svec)
                if a < b:  # evitar doble llamada
                    sala.connect_ids(a, b)
        sala.entrada_id = data.get("entrada")
        sala.salida_id = data.get("salida")
        return sala

class Clase(Sala):
    """
    Sala tipo aula con asientos (nodos) que pueden ocuparse.
    A* evita nodos ocupados salvo el goal.
    """

    def __init__(self, id_sala: int):
        super().__init__(id_sala, tipo="clase")
        self.asientos: List[int] = []            # ids de waypoints que son asientos
        self.ocupado: Dict[int, bool] = {}       # id -> ocupado

    # Marcar qué nodos son asientos
    def marcar_asiento(self, id_wp: int) -> None:
        if id_wp not in self.waypoints:
            raise KeyError(f"Asiento {id_wp} no existe en sala {self.id_sala}.")
        if id_wp not in self.asientos:
            self.asientos.append(id_wp)
        self.ocupado.setdefault(id_wp, False)

    def ocupar_asiento(self, id_wp: int) -> bool:
        if id_wp not in self.ocupado:
            return False
        if self.ocupado[id_wp]:
            return False
        self.ocupado[id_wp] = True
        return True

    def liberar_asiento(self, id_wp: int) -> None:
        if id_wp in self.ocupado:
            self.ocupado[id_wp] = False

    def asiento_libre_aleatorio(self) -> Optional[int]:
        libres = [i for i in self.asientos if not self.ocupado.get(i, False)]
        return random.choice(libres) if libres else None

    # Pathfinding: evita asientos ocupados y cualquier nodo que quieras bloquear.
    def get_path(self, start_id: int, goal_id: int) -> List[int]:
        def traversable(nid: int) -> bool:
            # Permitimos pasar por NO-asientos o por asientos libres.
            # (y el goal siempre será permitido por el A* base)
            if nid in self.ocupado:
                return not self.ocupado[nid]
            return True

        return self._a_star(start_id, goal_id, traversable=traversable)
    
    def to_json_struct(self) -> dict:
        base = super().to_json_struct()
        base["asientos"] = self.asientos
        # No hace falta guardar 'ocupado' ya que se reinicia a False siempre
        return base

    @classmethod
    def from_json_struct(cls, data: dict) -> "Clase":
        # Crear una instancia base
        clase = cls(data["id"])
        # Crear nodos y conexiones
        for sid, p in data["pos"].items():
            clase.add_waypoint(int(sid), tuple(p))
        for sid, vecinos in data["con"].items():
            a = int(sid)
            for svec in vecinos:
                b = int(svec)
                if a < b:
                    clase.connect_ids(a, b)

        clase.entrada_id = data.get("entrada")
        clase.salida_id = data.get("salida")

        # Marcar los asientos (reinicia ocupación a vacía)
        for aid in data.get("asientos", []):
            clase.marcar_asiento(aid)

        return clase

    def save_json(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_struct(), f, indent=4)

    @classmethod
    def load_json(cls, path: str) -> "Clase":
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_struct(data)

class Pasillo(Sala):
    """
    Sala tipo pasillo:
    - No calcula rutas con A* en tiempo real.
    - Usa rutas precomputadas entre ciertos pares de nodos.
    - Para variabilidad: elige una de las K mejores rutas con probabilidad
      ponderada por distancia (más corta = más probable).
    """

    def __init__(self, id_sala: int, alpha: float = 2.0):
        super().__init__(id_sala, tipo="pasillo")
        # cache: (a_id, b_id) (ordenado) -> lista de rutas (cada ruta = [ids])
        self.rutas = {}
        self.alpha = alpha  # exponente de penalización (1 suave, 2+ fuerte)

    @staticmethod
    def _key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    def set_rutas(self, a: int, b: int, rutas_ids: List[List[int]]) -> None:
        self.rutas[self._key(a, b)] = rutas_ids

    def _ruta_longitud(self, ruta_ids: List[int]) -> float:
        if len(ruta_ids) < 2:
            return 0.0
        total = 0.0
        for i in range(len(ruta_ids) - 1):
            A = self.waypoints[ruta_ids[i]].position
            B = self.waypoints[ruta_ids[i + 1]].position
            total += glm.distance(A, B)
        return total

    def _seleccionar_ponderado(self, rutas_info):
        lens = [r["longitud"] for r in rutas_info]
        pesos = [1.0 / (L ** self.alpha) for L in lens]
        total = sum(pesos)
        probs = [w / total for w in pesos]
        idx = random.choices(range(len(rutas_info)), weights=probs, k=1)[0]
        return rutas_info[idx]["ruta"]

    def get_path(self, start_id: int, goal_id: int) -> List[int]:
        key = self._key(start_id, goal_id)
        rutas = self.rutas.get(key)
        if not rutas:
            # Fallback: en caso de no tener precálculo, usa A* genérico
            return super().get_path(start_id, goal_id)
        # Elegir una ruta entre las K mejores (ya precalculadas)
        return self._seleccionar_ponderado(rutas)

    # --------- I/O de rutas precalculadas (ids) ---------
    def to_json_struct(self) -> dict:
        base = super().to_json_struct()

        # Convertir las claves (a,b) -> "a-b"
        rutas_str = {f"{a}-{b}": rutas for (a, b), rutas in self.rutas.items()}
        base["rutas"] = rutas_str
        base["alpha"] = self.alpha
        return base

    @classmethod
    def from_json_struct(cls, data: dict) -> "Pasillo":
        pasillo = cls(data["id"], alpha=data.get("alpha", 2.0))

        # Crear nodos y conexiones
        for sid, p in data["pos"].items():
            pasillo.add_waypoint(int(sid), tuple(p))
        for sid, vecinos in data["con"].items():
            a = int(sid)
            for svec in vecinos:
                b = int(svec)
                if a < b:
                    pasillo.connect_ids(a, b)

        pasillo.entrada_id = data.get("entrada")
        pasillo.salida_id = data.get("salida")

        # Cargar rutas precalculadas: convertir "a-b" -> (a,b)
        rutas_dict = {}
        for k, rutas in data.get("rutas", {}).items():
            a, b = map(int, k.split("-"))
            rutas_dict[(a, b)] = rutas
        pasillo.rutas = rutas_dict

        return pasillo

    def save_json(self, path: str) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_struct(), f, indent=4)

    @classmethod
    def load_json(cls, path: str) -> "Pasillo":
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_json_struct(data)

if __name__ == "__main__":
    import os
    import glm

    # Obtener la ruta de trabajo actual
    path_actual = os.getcwd()
    print(f"Guardando y cargando archivos de prueba en: {path_actual}\n")

    # ---------- Crear una CLASE ----------
    aula = Clase(11003)
    for i in range(5):
        aula.add_waypoint(i, (i, 0.0, i * 0.5))
    for i in range(4):
        aula.connect_ids(i, i + 1)

    # Marcar algunos como asientos
    aula.marcar_asiento(2)
    aula.marcar_asiento(3)
    # Definir puerta
    aula.set_puerta(entrada_id=0, salida_id=1)

    # Guardar a JSON con el nuevo método
    aula_path = os.path.join(path_actual, "aula_test.json")
    aula.save_json(aula_path)
    print(f"✅ Clase guardada en {aula_path}")

    # Cargar la clase desde JSON
    aula_cargada = Clase.load_json(aula_path)
    print("✅ Clase cargada correctamente\n")

    # ---------- Crear un PASILLO ----------
    pasillo = Pasillo(21001)
    for i in range(5):
        pasillo.add_waypoint(i, (i * 2.0, 0.0, 0.0))
    for i in range(4):
        pasillo.connect_ids(i, i + 1)
    pasillo.set_puerta(entrada_id=None, salida_id=4)

    # Añadir rutas precalculadas de ejemplo (ya con longitud incluida)
    pasillo.set_rutas(0, 4, [
        {"ruta": [0, 1, 2, 3, 4],
         "longitud": float(glm.distance(pasillo.get_wp(0).position, pasillo.get_wp(4).position))},
        {"ruta": [0, 1, 2, 4],
         "longitud": float(glm.distance(pasillo.get_wp(0).position, pasillo.get_wp(4).position) * 1.1)}
    ])

    # Guardar pasillo a JSON
    pasillo_path = os.path.join(path_actual, "pasillo_test.json")
    pasillo.save_json(pasillo_path)
    print(f"✅ Pasillo guardado en {pasillo_path}")

    # Cargar pasillo desde JSON
    pasillo_cargado = Pasillo.load_json(pasillo_path)
    print("✅ Pasillo cargado correctamente\n")

    # ---------- Mostrar resultados ----------
    print("---- VERIFICACIÓN ----")
    print("Aula cargada -> ID:", aula_cargada.id_sala)
    print("Asientos marcados:", aula_cargada.asientos)
    print("Puerta (entrada, salida):", aula_cargada.entrada_id, aula_cargada.salida_id)
    print("Waypoints aula:", len(aula_cargada.waypoints))

    print("\nPasillo cargado -> ID:", pasillo_cargado.id_sala)
    print("Puerta (entrada, salida):", pasillo_cargado.entrada_id, pasillo_cargado.salida_id)
    print("Rutas precalculadas:", list(pasillo_cargado.rutas.keys()))
    print("Ejemplo de ruta 0–4:", pasillo_cargado.rutas.get("(0, 4)") or pasillo_cargado.rutas.get((0, 4)))
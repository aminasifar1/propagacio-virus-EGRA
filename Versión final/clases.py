import json
from collections import defaultdict
import matplotlib.pyplot as plt


class GraphBuilder:
    def __init__(
        self,
        piso,
        pasillo,
        clase,
        punto_izq,
        punto_med,
        punto_der,
        punto_fila_4,
        punto_fila_3,
        dist_columnas,
        dist_filas,
        n_filas,
    ):

        # Identificadores base
        self.piso = piso
        self.pasillo = pasillo
        self.clase = clase

        # Contador interno para puntos
        self.node_counter = 1   # empieza en 1 → 001, 002, 003...

        # Puntos generadores
        self.p_izq = punto_izq
        self.p_med = punto_med
        self.p_der = punto_der
        self.pf4 = punto_fila_4
        self.pf3 = punto_fila_3

        # Parámetros geométricos
        self.dist_c = dist_columnas   # separación entre puntos de una fila (eje Z)
        self.dist_f = dist_filas      # separación entre filas/columnas (eje X)
        self.n_filas = n_filas

        self.pos = {}
        self.con = defaultdict(list)

        self.puertas_externas = []   # IDs 1, 2, 3...
        self.puertas_internas = []   # IDs normales
        self.puertas_coords = []     # coordenadas interiores para conexiones posteriores
        self.puerta_counter = 1      # para IDs 1,2,3...

        self.asientos = []

    def _new_id(self):
        """Devuelve un ID con formato pasillo_clase_pisoXXX."""
        id_num = f"{self.pasillo}{self.clase}{self.piso}{self.node_counter:03d}"
        self.node_counter += 1
        return id_num

    # ------------------------------------------------------------------
    # COLUMNAS
    # ------------------------------------------------------------------
    def generar_columnas(self):
        """
        Genera columnas desplazando en X (dist_filas) y manteniendo:
        - Y constante
        - Z fijo para cada columna
        Además crea un punto "cabeza" para cada columna y los conecta entre sí.
        """
        self.col_izq = []
        self.col_med = []
        self.col_der = []

        # --- Cabezas de columna (puntos generadores) ---
        self.head_izq = self._new_id()
        self.head_med = self._new_id()
        self.head_der = self._new_id()

        self.pos[self.head_izq] = (self.p_izq[0], self.p_izq[1], self.p_izq[2])
        self.pos[self.head_med] = (self.p_med[0], self.p_med[1], self.p_med[2])
        self.pos[self.head_der] = (self.p_der[0], self.p_der[1], self.p_der[2])

        # Conexiones entre cabezas: izquierda ↔ medio ↔ derecha
        self.con[self.head_izq].append(self.head_med)
        self.con[self.head_med].append(self.head_izq)

        self.con[self.head_med].append(self.head_der)
        self.con[self.head_der].append(self.head_med)

        # --- Puntos de las columnas por debajo de la cabeza ---
        # COLUMNAS: X avanza con dist_f, Z queda fijo para cada columna
        for i in range(1, self.n_filas + 1):
            x = self.p_izq[0] + i * self.dist_f
            self.col_izq.append((x, self.p_izq[1], self.p_izq[2]))

        for i in range(1, self.n_filas + 1):
            x = self.p_med[0] + i * self.dist_f
            self.col_med.append((x, self.p_med[1], self.p_med[2]))

        for i in range(1, self.n_filas + 1):
            x = self.p_der[0] + i * self.dist_f
            self.col_der.append((x, self.p_der[1], self.p_der[2]))

        # Convertimos a IDs (solo los puntos "de asiento", no las cabezas)
        self.col_izq_ids = []
        self.col_med_ids = []
        self.col_der_ids = []

        for p in self.col_izq:
            nid = self._new_id()
            self.pos[nid] = p
            self.col_izq_ids.append(nid)

        for p in self.col_med:
            nid = self._new_id()
            self.pos[nid] = p
            self.col_med_ids.append(nid)

        for p in self.col_der:
            nid = self._new_id()
            self.pos[nid] = p
            self.col_der_ids.append(nid)
        
        # Conectar cabezas con el primer nodo de su columna
        self.con[self.head_izq].append(self.col_izq_ids[0])
        self.con[self.col_izq_ids[0]].append(self.head_izq)

        self.con[self.head_med].append(self.col_med_ids[0])
        self.con[self.col_med_ids[0]].append(self.head_med)

        self.con[self.head_der].append(self.col_der_ids[0])
        self.con[self.col_der_ids[0]].append(self.head_der)

    def conectar_columna(self, ids):
        """Conexión vertical (en X) dentro de cada columna."""
        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]
            self.con[a].append(b)
            self.con[b].append(a)

    # ------------------------------------------------------------------
    # FILAS
    # ------------------------------------------------------------------
    def generar_filas(self):
        """
        Genera las filas horizontales:
        - X se mantiene fija por fila
        - Z cambia (dist_columnas)
        """
        self.filas_ids = []

        for i in range(self.n_filas):

            # FILA DE 4+1 PUNTOS (entre izquierda y medio)
            fila4 = []
            for j in range(5):
                z = self.pf4[2] + j * self.dist_c   # Z cambia
                x = self.pf4[0] + i * self.dist_f   # X fija dentro de la fila, distinta por fila
                nid = self._new_id()
                self.pos[nid] = (x, self.pf4[1], z)
                fila4.append(nid)
                self.asientos.append(nid)

            # FILA DE 3+1 PUNTOS (entre medio y derecha)
            fila3 = []
            for j in range(4):
                z = self.pf3[2] + j * self.dist_c
                x = self.pf3[0] + i * self.dist_f
                nid = self._new_id()
                self.pos[nid] = (x, self.pf3[1], z)
                fila3.append(nid)
                self.asientos.append(nid)

            self.filas_ids.append((fila4, fila3))

            # Conexiones internas dentro de cada fila
            self._conectar_linea(fila4)
            self._conectar_linea(fila3)

            # Conexión a columnas
            self.conectar_extremos(fila4, fila3, i)

    def _conectar_linea(self, ids):
        """Conecta cada punto con el siguiente en una misma fila."""
        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i + 1]
            self.con[a].append(b)
            self.con[b].append(a)

    def conectar_extremos(self, fila4, fila3, idx):
        """
        Conecta los extremos de las filas con las columnas correspondientes:
        - Fila de 4 → extremo izquierdo con columna izquierda
        - Fila de 4 → extremo derecho con columna central
        - Fila de 3 → extremo derecho con columna derecha
        - Fila de 3 → extremo izquierdo con columna central (nuevo)
        """
        # Fila de 4 → extremo izquierdo ↔ columna izquierda
        self.con[fila4[0]].append(self.col_izq_ids[idx])
        self.con[self.col_izq_ids[idx]].append(fila4[0])

        # Fila de 4 → extremo derecho ↔ columna central
        self.con[fila4[-1]].append(self.col_med_ids[idx])
        self.con[self.col_med_ids[idx]].append(fila4[-1])

        # Fila de 3 → extremo derecho ↔ columna derecha
        self.con[fila3[-1]].append(self.col_der_ids[idx])
        self.con[self.col_der_ids[idx]].append(fila3[-1])

        # Fila de 3 → extremo izquierdo ↔ columna central (lo que faltaba)
        self.con[fila3[0]].append(self.col_med_ids[idx])
        self.con[self.col_med_ids[idx]].append(fila3[0])
    
    def agregar_puertas(self, lista_puertas):
        """
        Crea puertas a partir de posiciones base.
        - Punto exterior: ID especial 1, 2, 3...
        - Punto interior: ID normal del grafo
        """

        for (x, y, z) in lista_puertas:

            # ---------- PUNTO EXTERIOR ----------
            id_exterior = str(self.puerta_counter)
            self.puerta_counter += 1
            pos_exterior = (x, y, z + 0.8)

            self.pos[id_exterior] = pos_exterior
            self.puertas_externas.append(id_exterior)

            # ---------- PUNTO INTERIOR ----------
            id_interior = self._new_id()
            pos_interior = (x, y, z - 0.8)

            self.pos[id_interior] = pos_interior
            self.puertas_internas.append(id_interior)
            self.puertas_coords.append(pos_interior)

            # ---------- Conexiones entre exterior ↔ interior ----------
            self.con[id_exterior].append(id_interior)
            self.con[id_interior].append(id_exterior)
    
    def conectar_puertas(self, col_der_ids):
        """
        Conecta cada punto interior de puertas:
        - Con todos los demás puntos interiores
        - Con todos los puntos de la columna derecha
        """

        # Conectar interiores entre ellos
        for i in range(len(self.puertas_internas)):
            a = self.puertas_internas[i]
            for j in range(i + 1, len(self.puertas_internas)):
                b = self.puertas_internas[j]

                self.con[a].append(b)
                self.con[b].append(a)

        # Conectar interiores con columna derecha
        for interior in self.puertas_internas:
            for nodo_der in col_der_ids:
                self.con[interior].append(nodo_der)
                self.con[nodo_der].append(interior)

    # ------------------------------------------------------------------
    def exportar(self):
        return {
            "id": f"{self.pasillo}{self.clase}{self.piso}",
            "tipo": "clase",
            "entrada": self.puertas_externas,   
            "salida": self.puertas_internas,    
            "pos": self.pos,
            "con": self.con,
            "asientos": self.asientos,
        }


# ----------------------------------------------------------------------
# DIBUJO DEL GRAFO
# ----------------------------------------------------------------------
def dibujar_grafo(pos, con, figsize=(12, 6)):
    plt.figure(figsize=figsize)

    # DIBUJAR ARISTAS (Z en eje horizontal, X en vertical)
    for a, vecinos in con.items():
        xa, _, za = pos[a]
        for b in vecinos:
            xb, _, zb = pos[b]
            plt.plot([za, zb], [xa, xb], linewidth=1, color="black", alpha=0.5)

    # DIBUJAR NODOS
    xs = []
    zs = []
    ids = []

    for nid, (x, _, z) in pos.items():
        xs.append(x)
        zs.append(z)
        ids.append(nid)

    plt.scatter(zs, xs, s=40, color="dodgerblue")

    # Etiquetas
    for nid, z, x in zip(ids, zs, xs):
        plt.text(z, x, nid, fontsize=0, ha="center", va="center")

    plt.title("Visualización del grafo (Z horizontal, X vertical)")
    plt.xlabel("Z")
    plt.ylabel("X")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# =====================================================================
# EJEMPLO DE USO
# =====================================================================
if __name__ == "__main__":
    builder = GraphBuilder(
	    punto_izq=(49.50, 2.05, -103.80),   # CAMBIAR
        punto_med=(49.50, 2.05, -98.65),   # CAMBIAR
        punto_der=(49.50, 2.05, -94.10),   # CAMBIAR
        punto_fila_4=(47.75, 2.05, -103.20),   # CAMBIAR
        punto_fila_3=(47.75, 2.05, -97.80),   # CAMBIAR
        dist_columnas=0.9,   
        dist_filas=-1.4,
        n_filas=4, # CAMBIAR
        pasillo=3,  # CAMBIAR
        clase=3,    # CAMBIAR
        piso=0,
    )

    builder.generar_columnas()
    builder.conectar_columna(builder.col_izq_ids)
    builder.conectar_columna(builder.col_med_ids)
    builder.conectar_columna(builder.col_der_ids)

    builder.generar_filas()

    builder.agregar_puertas([   # CAMBIAR
        (46.75, 2.05, -93.25)
    ])
    builder.conectar_puertas(builder.col_der_ids)

    data = builder.exportar()

    dibujar_grafo(data["pos"], data["con"])

    # Guardar JSON
    with open("Versión Final/data/salas/Q3-0009.json", "w") as f:   # CAMBIAR
        json.dump(data, f, indent=4)

    print("Grafo generado en grafo_generado.json")

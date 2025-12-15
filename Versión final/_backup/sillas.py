import json

def generar_cuadricula_filas(ancho_x: int, largo_z: int,
                             inicio: tuple[float, float],
                             espaciado_x: float = 1.0,
                             espaciado_z: float = 1.0) -> dict:
    """
    Genera un grafo de nodos distribuidos en una cuadrícula rectangular en el plano XZ,
    donde las conexiones solo son horizontales (entre nodos de la misma fila).

    ancho_x: número de nodos en el eje X
    largo_z: número de nodos en el eje Z
    inicio: coordenadas (x0, z0) de la esquina superior izquierda
    espaciado_x: distancia entre nodos en X
    espaciado_z: distancia entre nodos en Z
    """
    x0, z0 = inicio
    pos = {}
    con = {}

    # Crear posiciones
    index = 300
    for i in range(largo_z):     # Z
        for j in range(ancho_x): # X
            x = x0 + j * espaciado_x
            z = z0 + i * espaciado_z
            pos[str(index)] = [x, 0.0, z]
            index += 1

    # Crear conexiones solo entre vecinos de la misma fila
    for i in range(largo_z):
        for j in range(ancho_x):
            idx = i * ancho_x + j + 300
            vecinos = []
            # Conectar con el de la izquierda
            if j > 0:
                vecinos.append(str(idx - 1))
            # Conectar con el de la derecha
            if j < ancho_x - 1:
                vecinos.append(str(idx + 1))
            con[str(idx)] = vecinos

    return {"pos": pos, "con": con}

"""
Con este codigo si sacais la distancia entre sillas horizontalmente (la silla con la de al lado), 
la posicion de la silla superior izquierda (la de alante a la izquierda)
y la distancia entre filas (la silla con la de detrás), 
podeis sacar facil la posicion y conexiones de cada silla del aula para hacer el grafo
"""

data = generar_cuadricula_filas(
    ancho_x=4, # numero de sillas por fila
    largo_z=11, # numero de filas
    inicio=(-3.4, 4.5), # Posicion de la silla de alante a la izquierda
    espaciado_x=0.7, # Distancia entre una silla y la de su derecha
    espaciado_z=0.9 # Distancia entre una silla y la de detras
)

print(json.dumps(data, indent=4))
print(list(data["con"].keys()))
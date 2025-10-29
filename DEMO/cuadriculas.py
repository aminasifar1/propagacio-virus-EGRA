import json

def generar_cuadricula_rectangular_8(ancho_x: int, largo_z: int,
                                     inicio: tuple[float, float],
                                     espaciado_x: float = 1.0,
                                     espaciado_z: float = 1.0) -> dict:
    """
    Genera un grafo de nodos distribuidos en una cuadrícula rectangular en el plano XZ.
    Cada nodo se conecta con sus 8 vecinos (verticales, horizontales y diagonales).

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
    index = 200
    for i in range(largo_z):     # Z
        for j in range(ancho_x): # X
            x = x0 + j * espaciado_x
            z = z0 + i * espaciado_z
            pos[str(index)] = [x, 0.0, z]
            index += 1

    # Crear conexiones (8 direcciones)
    for i in range(largo_z):
        for j in range(ancho_x):
            idx = i * ancho_x + j + 200
            vecinos = []

            # Recorremos los desplazamientos de las 8 direcciones
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dz == 0:
                        continue  # ignorar el propio nodo
                    ni, nj = i + dz, j + dx
                    if 0 <= ni < largo_z and 0 <= nj < ancho_x:
                        vecino_idx = ni * ancho_x + nj
                        vecinos.append(str(vecino_idx + 200))

            con[str(idx)] = vecinos

    return {"pos": pos, "con": con}

data = generar_cuadricula_rectangular_8(
    ancho_x=6, 
    largo_z=9, 
    inicio=(-8.5, 15.4), 
    espaciado_x=1.6, 
    espaciado_z=1.6
)

print(json.dumps(data, indent=4))

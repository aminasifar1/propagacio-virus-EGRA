import json
import matplotlib.pyplot as plt

def borrar():
    nodos_a_borrar = ["458", "457", "456", "455", "391", "344", "345", "346", "347", "371", "459", "471"]

    with open(r"DEMO\data\salas\aula3.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Borrar de pos
    for nodo in nodos_a_borrar:
        if nodo in data["pos"]:
            del data["pos"][nodo]
            print(f"Eliminado de pos: {nodo}")

    # Borrar de con y sus referencias
    for nodo in nodos_a_borrar:
        if nodo in data["con"]:
            del data["con"][nodo]
            print(f"Eliminado de con: {nodo}")

    # Eliminar referencias en las conexiones de otros nodos
    for nodo_key, vecinos in data["con"].items():
        data["con"][nodo_key] = [v for v in vecinos if v not in nodos_a_borrar]

    # Borrar de asientos
    data["asientos"] = [a for a in data["asientos"] if a not in nodos_a_borrar]

    # Borrar de pasillos
    data["pasillos"] = [p for p in data["pasillos"] if p not in nodos_a_borrar]

    # Guardar
    with open(r"DEMO\data\salas\aula3.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print("\n✅ Nodos eliminados correctamente")

def mostrar():
    with open(r"DEMO/data/salas/aula3.json", encoding="utf-8") as f:
        data = json.load(f)

    fig, ax = plt.subplots()
    for node, (x, _, z) in data["pos"].items():
        ax.scatter(x, z, color="skyblue")
        ax.text(x, z, node, fontsize=8, ha="center", va="bottom")
        for vecino in data["con"].get(node, []):
            x2, _, z2 = data["pos"][vecino]
            ax.plot([x, x2], [z, z2], color="gray")

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    plt.show()
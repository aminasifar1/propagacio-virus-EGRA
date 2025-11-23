import json
import os

def main():
    # Carpeta destino
    save_dir = "DEMO2/data/horaris"

    # Crear carpeta si no existe
    os.makedirs(save_dir, exist_ok=True)

    # 1. Input del nombre de la clase
    class_name = input("Introduce el nombre de la clase: ").strip()

    # 2. Crear diccionario con una lista vacía
    data = {class_name: []}

    print("\nIntroduce valores para añadir a la lista.")
    print("Escribe 'FIN' para terminar.\n")

    # 3. Rellenar la lista con inputs
    while True:
        value = input("Añadir valor: ").strip()
        
        if value.upper() == "FIN" or value.upper() == "":
            break
        
        data[class_name].append(value)

    # 4. Guardar en JSON en la carpeta indicada
    filename = os.path.join(save_dir, f"{class_name}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nDiccionario guardado correctamente en '{filename}'")
    print("Contenido final:")
    print(data)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

def _as_str(x) -> str:
    return str(x)

def _ensure_list(d, key):
    if key not in d or d[key] is None:
        d[key] = {}
    return d[key]

def merge_clase_entradas_into_pasillo(
    pasillo_json_path: str | Path,
    clases_dir: str | Path,
    prefix: str,
    pasillo_start_id,
    pasillo_end_id,
    out_path: str | Path | None = None,
):
    pasillo_json_path = Path(pasillo_json_path)
    clases_dir = Path(clases_dir)
    out_path = Path(out_path) if out_path else pasillo_json_path.with_name(pasillo_json_path.stem + "_merged.json")

    start_id = _as_str(pasillo_start_id)
    end_id = _as_str(pasillo_end_id)

    # --- cargar pasillo ---
    pasillo = json.loads(pasillo_json_path.read_text(encoding="utf-8"))

    if pasillo.get("tipo") != "pasillo":
        raise ValueError(f"El fichero no parece un pasillo: tipo={pasillo.get('tipo')}")

    pos_pas = _ensure_list(pasillo, "pos")
    con_pas = _ensure_list(pasillo, "con")

    # Validación mínima: los dos nodos del pasillo deben existir
    if start_id not in pos_pas:
        raise KeyError(f"El nodo start_id={start_id} no existe en pasillo['pos']")
    if end_id not in pos_pas:
        raise KeyError(f"El nodo end_id={end_id} no existe en pasillo['pos']")

    # Asegurar que existan sus listas de conexiones
    con_pas.setdefault(start_id, [])
    con_pas.setdefault(end_id, [])

    # --- recorrer clases cuyo nombre empiece por prefix ---
    class_files = sorted(clases_dir.glob(f"{prefix}*.json"))
    if not class_files:
        raise FileNotFoundError(f"No se encontraron JSON en {clases_dir} que empiecen por '{prefix}'")

    added = 0
    skipped_missing_pos = 0

    for fp in class_files:
        data = json.loads(fp.read_text(encoding="utf-8"))

        entradas = data.get("entrada") or []
        pos_cls = data.get("pos") or {}

        # entradas puede ser lista de strings; nos quedamos solo con las que existan en pos
        for eid in entradas:
            eid = _as_str(eid)
            if eid not in pos_cls:
                skipped_missing_pos += 1
                continue

            # 1) añadir/actualizar posición en el pasillo
            pos_pas[eid] = pos_cls[eid]

            # 2) definir conexiones del nodo entrada: exclusivamente a start y end
            con_pas[eid] = [start_id, end_id]

            # 3) añadir el nodo entrada a las listas de start y end (sin duplicados)
            if eid not in con_pas[start_id]:
                con_pas[start_id].append(eid)
            if eid not in con_pas[end_id]:
                con_pas[end_id].append(eid)

            added += 1

    # Ordenar listas de conexiones (opcional, pero queda limpio)
    for k, lst in con_pas.items():
        # Mantener como strings; ordenar por valor numérico si se puede
        try:
            con_pas[k] = sorted(set(map(str, lst)), key=lambda s: int(s))
        except ValueError:
            con_pas[k] = sorted(set(map(str, lst)))

    out_path.write_text(json.dumps(pasillo, ensure_ascii=False, indent=4), encoding="utf-8")

    return {
        "out_path": str(out_path),
        "classes_matched": len(class_files),
        "entradas_added": added,
        "entradas_skipped_missing_pos": skipped_missing_pos,
    }


if __name__ == "__main__":
    # Ejemplo:
    # - pasillo: fichero del pasillo
    # - clases_dir: carpeta donde están Q1-0007.json, Q1-0013.json, ...
    # - prefix: "Q1-0"
    # - start/end: 12 y 30
    report = merge_clase_entradas_into_pasillo(
        pasillo_json_path="Versión final/data/salas/pasillo.json",
        clases_dir="Versión final/data/salas",
        prefix="Q4-1",
        pasillo_start_id=55,
        pasillo_end_id=59,
        out_path="pasillo_grafo_merged.json",
    )
    print(report)

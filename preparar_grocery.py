"""
Convierte GroceryStoreDataset al formato de YOLOv8-cls y genera catalogo.json con las descripciones.

Uso (desde la carpeta del proyecto):
  git clone https://github.com/marcusklasson/GroceryStoreDataset.git
  python preparar_grocery.py --origen GroceryStoreDataset/dataset [--traducir]

Crea:
  dataset_cls/train/<clase>/*.jpg   dataset_cls/val/<clase>/*.jpg   dataset_cls/test/<clase>/*.jpg
  catalogo.json  (clase, categoría, descripción de classes.csv; conserva tu sección "voz")
"""
import argparse, csv, json, re, shutil
from pathlib import Path

BASE = Path(__file__).parent


def leer_clases(csv_path):
    """Lee classes.csv de forma tolerante (busca columnas por palabra clave)."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.reader(f, skipinitialspace=True))
    cab = [c.strip().lower() for c in filas[0]]

    def col(*claves, excluir=()):
        for i, c in enumerate(cab):
            if all(k in c for k in claves) and not any(x in c for x in excluir):
                return i
        return None

    i_nom = col("class", "name", excluir=("coarse",))
    i_id = col("class", "id", excluir=("coarse",))
    i_cnom = col("coarse", "name")
    i_desc = col("description")
    if None in (i_nom, i_id):
        raise SystemExit(f"No reconozco las columnas de classes.csv: {filas[0]}")
    clases = {}
    for r in filas[1:]:
        if len(r) <= max(i_nom, i_id):
            continue
        desc = ", ".join(x.strip() for x in r[i_desc:]).strip() if i_desc is not None else ""  # por si la descripción trae comas
        clases[int(r[i_id])] = {
            "nombre": r[i_nom].strip(),
            "categoria": r[i_cnom].strip() if i_cnom is not None and len(r) > i_cnom else "",
            "descripcion": desc,
        }
    return clases


def copiar_split(origen, split, clases, destino):
    n = 0
    for linea in (origen / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        partes = [p.strip() for p in re.split(r"[,\s]+", linea.strip()) if p.strip()]
        if len(partes) < 2:
            continue
        ruta, id_fino = partes[0], int(partes[1])
        src = origen / ruta
        if not src.exists():
            continue
        carpeta = destino / split / clases[id_fino]["nombre"]
        carpeta.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, carpeta / src.name)
        n += 1
    return n


def traducir(textos):
    from deep_translator import GoogleTranslator  # pip install deep-translator
    tr = GoogleTranslator(source="en", target="es")
    return [tr.translate(t) if t else t for t in textos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="carpeta GroceryStoreDataset/dataset")
    ap.add_argument("--destino", default=str(BASE / "dataset_cls"))
    ap.add_argument("--traducir", action="store_true", help="traduce descripciones al español (requiere internet)")
    args = ap.parse_args()
    origen, destino = Path(args.origen), Path(args.destino)

    clases = leer_clases(origen / "classes.csv")
    print(f"{len(clases)} clases leídas")
    for split in ("train", "val", "test"):
        print(split, copiar_split(origen, split, clases, destino), "imágenes")

    ids = sorted(clases)
    descs = [clases[i]["descripcion"] for i in ids]
    if args.traducir:
        print("Traduciendo descripciones...")
        descs = traducir(descs)

    ruta_cat = BASE / "catalogo.json"
    voz = json.loads(ruta_cat.read_text(encoding="utf-8"))["voz"] if ruta_cat.exists() else {}
    productos = {}
    for i, d in zip(ids, descs):
        c = clases[i]
        productos[c["nombre"]] = {
            "nombre": c["nombre"].replace("-", " "),
            "categoria": c["categoria"],
            "descripcion": d or "Producto de supermercado.",
            "precio": "no disponible",
        }
    ruta_cat.write_text(json.dumps({"voz": voz, "productos": productos}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"catalogo.json actualizado con {len(productos)} productos")


if __name__ == "__main__":
    main()

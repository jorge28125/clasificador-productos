"""
Entrena YOLOv8 con las clases definidas en catalogo.json.

Estructura esperada del dataset (formato YOLO):
  dataset/images/train/*.jpg   dataset/labels/train/*.txt
  dataset/images/val/*.jpg     dataset/labels/val/*.txt
Cada .txt tiene una línea por objeto:  <id_clase> <x_centro> <y_centro> <ancho> <alto>  (normalizado 0-1)
El id_clase es la POSICIÓN del producto en catalogo.json -> "productos" (0 = primero, 1 = segundo, ...).

Uso:  python entrenar.py            (o: python entrenar.py --epocas 80 --tam 640)
"""
import argparse, json, shutil, sys
from pathlib import Path

BASE = Path(__file__).parent
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cargar_catalogo():
    with open(BASE / "catalogo.json", encoding="utf-8") as f:
        return json.load(f)


def validar_dataset(n_clases):
    ok = True
    for split in ("train", "val"):
        imgs = [p for p in (BASE / "dataset/images" / split).glob("*") if p.suffix.lower() in EXT]
        if not imgs:
            print(f"[ERROR] No hay imágenes en dataset/images/{split}/")
            ok = False
            continue
        sin_label, ids_malos = 0, 0
        for im in imgs:
            lab = BASE / "dataset/labels" / split / (im.stem + ".txt")
            if not lab.exists():
                sin_label += 1
                continue
            for linea in lab.read_text().splitlines():
                partes = linea.split()
                if partes and (not partes[0].isdigit() or int(partes[0]) >= n_clases):
                    ids_malos += 1
        print(f"[{split}] {len(imgs)} imágenes | sin etiqueta: {sin_label} | ids de clase inválidos: {ids_malos}")
        if ids_malos:
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=50)
    ap.add_argument("--tam", type=int, default=640)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--base", default="yolov8n.pt", help="modelo base para transfer learning")
    args = ap.parse_args()

    cat = cargar_catalogo()
    clases = list(cat["productos"].keys())
    sin_desc = [c for c in clases if not cat["productos"][c].get("descripcion", "").strip()]
    if sin_desc:
        print(f"[ERROR] Estos productos no tienen descripción en catalogo.json: {sin_desc}")
        sys.exit(1)

    # data.yaml se genera desde el catálogo: una sola fuente de verdad (clases + descripciones)
    yaml = [f"path: {(BASE / 'dataset').as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(clases)]
    (BASE / "dataset/data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"Clases ({len(clases)}): {clases}")

    if not validar_dataset(len(clases)):
        sys.exit("Corrige el dataset y vuelve a ejecutar.")

    from ultralytics import YOLO
    model = YOLO(args.base)
    model.train(data=str(BASE / "dataset/data.yaml"), epochs=args.epocas, imgsz=args.tam,
                batch=args.lote, project=str(BASE / "runs"), name="productos", exist_ok=True, verbose=False)

    best = BASE / "runs/productos/weights/best.pt"
    if best.exists():
        shutil.copy(best, BASE / "modelos/best.pt")
        print("Modelo copiado a modelos/best.pt (app.py lo usará automáticamente).")


if __name__ == "__main__":
    main()

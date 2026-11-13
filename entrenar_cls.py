"""Entrena YOLOv8-cls con dataset_cls/ (carpeta por clase). Genera modelos/best.pt"""
import argparse, shutil
from pathlib import Path

BASE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--epocas", type=int, default=20)
ap.add_argument("--tam", type=int, default=224)
ap.add_argument("--lote", type=int, default=32)
ap.add_argument("--base", default="yolov8n-cls.pt")
a = ap.parse_args()

from ultralytics import YOLO
model = YOLO(a.base)
model.train(data=str(BASE / "dataset_cls"), epochs=a.epocas, imgsz=a.tam, batch=a.lote,
            project=str(BASE / "runs"), name="productos_cls", exist_ok=True)
best = BASE / "runs/productos_cls/weights/best.pt"
(BASE / "modelos").mkdir(exist_ok=True)
shutil.copy(best, BASE / "modelos/best.pt")
print("Listo: modelos/best.pt")

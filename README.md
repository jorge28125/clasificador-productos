# Clasificador de Productos

App web (Flask + YOLOv8) que clasifica productos a partir de una foto.

## Ejecutar local
```bash
pip install -r requirements.txt
python app.py
```
Abre http://127.0.0.1:5000

## Desplegar en Render
1. Sube este repo a GitHub.
2. En Render, crea un "Web Service" conectado al repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. (Opcional) Agrega disco persistente si quieres guardar uploads.

## Notas
- El modelo entrenado está en `modelos/best.pt`.
- El catálogo está en `catalogo.json`.
- Para reentrenar: `python entrenar_cls.py --epocas 20` (clasificación) o `python entrenar.py` (detección).

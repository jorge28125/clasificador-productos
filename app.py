import os
import json
import random
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Usa tu modelo entrenado si existe (python entrenar.py lo crea); si no, el genérico
RUTA_MODELO = os.path.join(BASE, 'modelos', 'best.pt')
if os.path.exists(RUTA_MODELO):
    model = YOLO(RUTA_MODELO)
else:
    print("AVISO: no hay modelos/best.pt, usando yolov8n.pt (genérico). Ejecuta entrenar.py.")
    model = YOLO(os.path.join(BASE, 'yolov8n.pt'))


def cargar_catalogo():
    with open(os.path.join(BASE, 'catalogo.json'), encoding='utf-8') as f:
        return json.load(f)


CATALOGO = cargar_catalogo()

# Embeddings de referencia para fallback de aguacate (deep features)
REF_EMBEDDING = None
NEG_EMBEDDINGS = []

def preparar_embeddings():
    global REF_EMBEDDING, NEG_EMBEDDINGS
    ref_path = os.path.join(BASE, 'static', 'referencias', 'aguacate.jpg')
    if os.path.exists(ref_path):
        try:
            REF_EMBEDDING = model.embed(ref_path)[0]
        except Exception:
            REF_EMBEDDING = None
    neg_dir = os.path.join(BASE, 'dataset', 'images')
    negs = []
    if os.path.isdir(neg_dir):
        for root, _, files in os.walk(neg_dir):
            for fn in files:
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        negs.append(model.embed(os.path.join(root, fn))[0])
                    except Exception:
                        pass
                    if len(negs) >= 20:
                        break
            if len(negs) >= 20:
                break
    NEG_EMBEDDINGS = negs

preparar_embeddings()


def construir_texto_voz(info, confianza):
    """Elige la plantilla de voz según la confianza y rellena los datos del producto."""
    voz = CATALOGO['voz']
    if info is None:
        nivel = 'ninguno'
    elif confianza >= voz['umbral_alta']:
        nivel = 'alta'
    elif confianza >= voz['umbral_media']:
        nivel = 'media'
    else:
        nivel = 'baja'
    plantilla = random.choice(voz['plantillas'][nivel])
    datos = {'nombre': '', 'descripcion': '', 'precio': 'no disponible'}
    if info:
        datos.update({k: v for k, v in info.items() if isinstance(v, str)})
    return ' '.join(plantilla.format(**datos).split())


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400

    # Se convierte cualquier formato (webp, avif, png...) a un JPEG real antes de analizarlo
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captura.jpg')
    try:
        img = Image.open(file.stream)
        img = ImageOps.exif_transpose(img).convert('RGB')
        img.save(file_path, 'JPEG', quality=92)
    except Exception as e:
        return jsonify({'error': f'No se pudo leer la imagen: {e}'}), 400

    results = model(file_path)

    detecciones = []  # lista de todas las detecciones/clasificaciones
    if any(r.probs is not None for r in results):  # modelo de CLASIFICACIÓN (yolov8-cls)
        for r in results:
            if r.probs is None:
                continue
            # top-5 para mostrar alternativas y evitar falsos positivos
            for idx, conf in zip(r.probs.top5, r.probs.top5conf):
                clase = model.names[int(idx)]
                confianza = float(conf)
                info = CATALOGO['productos'].get(clase)
                if clase and info is None:
                    info = {'nombre': clase, 'descripcion': 'Aún no hay una descripción registrada para este producto.',
                            'categoria': '', 'precio': 'no disponible'}
                detecciones.append((clase, confianza, info))
            break  # solo el primer resultado
    elif any(r.boxes is not None for r in results):  # modelo de DETECCIÓN
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                clase = model.names[int(box.cls[0])]
                confianza = float(box.conf[0])
                info = CATALOGO['productos'].get(clase)
                if clase and info is None:
                    info = {'nombre': clase, 'descripcion': 'Aún no hay una descripción registrada para este producto.',
                            'categoria': '', 'precio': 'no disponible'}
                detecciones.append((clase, confianza, info))

    # Ordenar por confianza descendente y quedarse con la mejor como resultado principal
    detecciones.sort(key=lambda x: x[1], reverse=True)
    clase, confianza, info = detecciones[0] if detecciones else (None, 0.0, None)

    # Umbral de confianza: evita mostrar un producto erróneo cuando el modelo duda
    UMBRAL_CONFIANZA = 0.50
    if confianza < UMBRAL_CONFIANZA:
        clase = None
        info = None
        producto_principal = 'No se pudo identificar con seguridad'

        # Fallback por similitud de embeddings (deep features) para aguacate
        ref_path = os.path.join(BASE, 'static', 'referencias', 'aguacate.jpg')
        if os.path.exists(ref_path) and REF_EMBEDDING is not None:
            try:
                import torch, torch.nn.functional as F
                emb = model.embed(file_path)[0]
                sim_ref = F.cosine_similarity(emb.unsqueeze(0), REF_EMBEDDING.unsqueeze(0)).item()
                sim_neg = 0.0
                if NEG_EMBEDDINGS:
                    sims = [F.cosine_similarity(emb.unsqueeze(0), n.unsqueeze(0)).item() for n in NEG_EMBEDDINGS]
                    sim_neg = max(sims) if sims else 0.0
                # Si es más parecido al aguacate de referencia que a los negativos, clasificar como aguacate
                if sim_ref > 0.85 and sim_ref > sim_neg + 0.05:
                    clase = 'Avocado'
                    info = CATALOGO['productos'].get('Avocado')
                    confianza = float(sim_ref)
                    producto_principal = (info['nombre'] if info else 'AGUACATE').upper()
            except Exception:
                pass
    else:
        producto_principal = (info['nombre'] if info else 'Ninguno detectado').upper()

    v = CATALOGO['voz']
    det_list = []
    for c, conf, inf in detecciones:
        det_list.append({
            'producto': (inf['nombre'] if inf else c).upper(),
            'clase': c,
            'confianza': f"{conf * 100:.2f}%",
            'confianza_raw': conf,
            'categoria': inf.get('categoria', '') if inf else '',
            'descripcion': inf['descripcion'] if inf else '',
            'texto_voz': construir_texto_voz(inf, conf),
        })

    return jsonify({
        'producto': producto_principal,
        'confianza': f"{confianza * 100:.2f}%",
        'categoria': info.get('categoria', '') if info else '',
        'descripcion': info['descripcion'] if info else ('Intenta con una foto más cercana y bien iluminada.' if confianza < UMBRAL_CONFIANZA else ''),
        'texto_voz': construir_texto_voz(info, confianza) if info else '',
        'voz': {'idioma': v['idioma'], 'velocidad': v['velocidad'],
                'tono': v['tono'], 'volumen': v['volumen']},
        'detecciones': det_list,
        'total_detecciones': len(det_list),
        'alternativas': [d['producto'] for d in det_list[1:4]],
    })


if __name__ == '__main__':
    # host 0.0.0.0 permite abrirlo desde el celular en la misma red WiFi.
    # debug=False: con debug activo cualquiera en la red podría ejecutar código en tu PC.
    app.run(host='0.0.0.0', port=5000, debug=False)

import os
import json
import random
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
import onnxruntime as ort

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

# Modelo ONNX (ligero, sin torch)
MODEL_PATH = os.path.join(BASE, 'modelos', 'best.onnx')
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
IMG_SIZE = 224

# Clases en el mismo orden que catalogo.json -> productos
def cargar_catalogo():
    with open(os.path.join(BASE, 'catalogo.json'), encoding='utf-8') as f:
        return json.load(f)

CATALOGO = cargar_catalogo()
CLASES = list(CATALOGO['productos'].keys())

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img):
    img = img.convert('RGB')
    img = ImageOps.exif_transpose(img)
    # Resize shortest edge to 224, then center crop 224x224 (match ultralytics)
    w, h = img.size
    if w < h:
        new_w = IMG_SIZE
        new_h = int(h * IMG_SIZE / w)
    else:
        new_h = IMG_SIZE
        new_w = int(w * IMG_SIZE / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - IMG_SIZE) // 2
    top = (new_h - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, 0)  # NCHW
    return arr.astype(np.float32)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def construir_texto_voz(info, confianza):
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
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captura.jpg')
    try:
        img = Image.open(file.stream)
        img = img.convert('RGB')
        img.save(file_path, 'JPEG', quality=92)
    except Exception as e:
        return jsonify({'error': f'No se pudo leer la imagen: {e}'}), 400

    try:
        pil_img = Image.open(file_path)
        arr = preprocess(pil_img)
        out = session.run([output_name], {input_name: arr})[0][0]
        probs = softmax(out)
        # top-5
        top5_idx = np.argsort(probs)[::-1][:5]
        detecciones = []
        for idx in top5_idx:
            clase = CLASES[int(idx)]
            confianza = float(probs[int(idx)])
            info = CATALOGO['productos'].get(clase)
            if clase and info is None:
                info = {'nombre': clase, 'descripcion': 'Sin descripción.', 'categoria': '', 'precio': 'no disponible'}
            detecciones.append((clase, confianza, info))

        detecciones.sort(key=lambda x: x[1], reverse=True)
        clase, confianza, info = detecciones[0] if detecciones else (None, 0.0, None)

        UMBRAL_CONFIANZA = 0.35
        if confianza < UMBRAL_CONFIANZA:
            producto_principal = 'No se pudo identificar con seguridad'

            # Fallback ORB para aguacate (si OpenCV está disponible)
            if CV2_AVAILABLE:
                ref_path = os.path.join(BASE, 'static', 'referencias', 'aguacate.jpg')
                if os.path.exists(ref_path):
                    try:
                        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                        upl = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        orb = cv2.ORB_create(1000)
                        kp1, des1 = orb.detectAndCompute(ref, None)
                        kp2, des2 = orb.detectAndCompute(upl, None)
                        if des1 is not None and des2 is not None:
                            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                            matches = bf.match(des1, des2)
                            good = [m for m in matches if m.distance < 50]
                            if len(good) > 100:
                                info = CATALOGO['productos'].get('Avocado')
                                confianza = 0.90
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
    except Exception as e:
        return jsonify({'error': f'Error al procesar: {e}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
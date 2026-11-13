# Clasificador de productos

## Opción A: GroceryStoreDataset (clasificación, 81 productos con descripción) — recomendado
Ver `colab_grocery.ipynb` (Google Colab) o, en local:
1. `git clone https://github.com/marcusklasson/GroceryStoreDataset.git`
2. `python preparar_grocery.py --origen GroceryStoreDataset/dataset --traducir`
3. `python entrenar_cls.py --epocas 20`
4. `python app.py`

## Opción B: tus propios productos con cajas (detección)
Edita `catalogo.json`, pon fotos/etiquetas YOLO en `dataset/` y ejecuta `python entrenar.py`.

`catalogo.json` guarda las descripciones y el patrón de voz (sección "voz"). `app.py` detecta solo si el modelo es de clasificación o detección.

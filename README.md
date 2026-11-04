# violence-detection

Clasificación de imágenes en **violencia** / **no violencia** con deep learning (Módulo 7).

- **Entrenamiento y validación:** [Real Life Violence Situations – 11K images](https://www.kaggle.com/datasets/abdulmananraja/real-life-violence-situations) (Kaggle).
- **Prueba:** [A Dataset for Automatic Violence Detection in Videos](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos) (Bianculli et al.), del que se extraen fotogramas.

## Librerías

| Librería | Paquete pip | Se usa en |
|---|---|---|
| NumPy | `numpy` | todos |
| pandas | `pandas` | `understanding-preparation.ipynb`, `baseline.py` |
| Matplotlib | `matplotlib` | todos |
| Pillow | `pillow` | todos |
| TensorFlow / Keras | `tensorflow` | todos |
| scikit-learn | `scikit-learn` | `understanding-preparation.ipynb` (k-means, particiones por grupo), `baseline.py` (matriz de confusión) |
| SciPy | `scipy` | `understanding-preparation.ipynb` (componentes conexas) |
| OpenCV | `opencv-python` | `understanding-preparation.ipynb` (extracción de fotogramas de los videos) |
| tqdm | `tqdm` | `understanding-preparation.ipynb` |
| kagglehub | `kagglehub` | `understanding-preparation.ipynb`, `baseline.py` (descarga del dataset) |

Además se necesita `git` (para clonar el dataset de prueba) y, al correr en Colab, `google.colab` para montar Google Drive.

Instalación fuera de Colab:

```bash
pip install numpy pandas matplotlib pillow tensorflow scikit-learn scipy opencv-python tqdm kagglehub
```

En Google Colab casi todo viene instalado; si falta `kagglehub`: `!pip install -q kagglehub`.

## Archivos

| Archivo | Qué hace |
|---|---|
| `understanding-preparation.ipynb` | Comprensión y preparación de los datos. Descarga el dataset de Kaggle, revisa clases y formato, calcula embeddings con MobileNetV2, elimina duplicados y fotogramas oscuros, agrupa imágenes similares (componentes conexas, τ = 0.75) y hace la partición train/val por grupos para evitar fuga de fotogramas del mismo video. También clona el dataset de Bianculli y extrae 8 fotogramas por video para el conjunto de prueba. |
| `baseline.py` | Modelo base: CNN sencilla (1 capa convolucional + densa) con imágenes de 128 × 128. Carga la partición, entrena 20 épocas, grafica loss/accuracy, evalúa en validación y prueba (matriz de confusión y reporte de clasificación) y guarda el modelo. |
| `predict.ipynb` | Prueba del modelo (Colab): carga `baseline.keras` desde Drive, lee las imágenes de `MyDrive/violence-detection/images/` e imprime y muestra la clase predicha con su probabilidad. |
| `baseline.ipynb` | Notebook para Colab que clona (o actualiza) este repositorio y ejecuta `predict.py`. |
| `images/` | Imágenes propias para probar el modelo con `predict.ipynb`. |
| `Evidencia_modulo07_Kamila.pdf` | Documento de la entrega. |

### Archivos generados (se guardan en Google Drive, `MyDrive/violence-detection/`, no en el repo)

| Archivo | Lo genera | Lo usa |
|---|---|---|
| `emb_mobilenetv2.npy` | `understanding-preparation.ipynb` | el mismo notebook (caché para no recalcular embeddings) |
| `splits.csv` | `understanding-preparation.ipynb` | `baseline.py` |
| `bianculli_frames.csv` | `understanding-preparation.ipynb` | `baseline.py` |
| `bianculli_frames.zip` | `understanding-preparation.ipynb` | `baseline.py` (y el notebook en corridas posteriores) |
| `baseline.keras` | `baseline.py` | `predict.ipynb` |
| `baseline_history.csv` | `baseline.py` | — (historial de entrenamiento) |

## Dependencia entre archivos

```
understanding-preparation.ipynb
        │  genera splits.csv, bianculli_frames.csv, bianculli_frames.zip  (en Drive)
        ▼
baseline.py
        │  genera baseline.keras, baseline_history.csv  (en Drive)
        ▼
predict.ipynb  ◄── MyDrive/violence-detection/images/
        ▲
baseline.ipynb  (clona el repo y corre predict.py)
```

Se deben correr en este orden; cada paso necesita lo que produjo el anterior.

## Cómo correr cada archivo

Los tres pasos están pensados para **Google Colab** con Google Drive montado. Conviene activar GPU en *Entorno de ejecución → Cambiar tipo de entorno de ejecución*.

### 1. `understanding-preparation.ipynb`

1. Abrir el notebook en Colab.
2. Ejecutar todas las celdas (*Entorno de ejecución → Ejecutar todas*). Pedirá permiso para montar Drive.
3. Al terminar, en `MyDrive/violence-detection/` quedan `splits.csv`, `bianculli_frames.csv`, `bianculli_frames.zip` y `emb_mobilenetv2.npy`.

Fuera de Colab el notebook guarda los archivos en la carpeta actual (`USAR_DRIVE` se ignora).

### 2. `baseline.py`

Requiere los archivos del paso 1 en `MyDrive/violence-detection/`. En una celda de Colab:

```python
!git clone https://github.com/kamimartinez/violence-detection.git /content/violence-detection
%run /content/violence-detection/baseline.py
```

Descarga el dataset de Kaggle, descomprime los fotogramas de prueba, entrena y guarda `baseline.keras` y `baseline_history.csv` en Drive. Este script monta Drive siempre, así que solo corre en Colab.

### 3. `predict.ipynb`

Requiere `baseline.keras` en `MyDrive/violence-detection/`. Subir las imágenes a probar a `MyDrive/violence-detection/images/` (la carpeta se crea sola la primera vez que se corre el notebook), abrir el notebook en Colab y ejecutar todas las celdas.

Se clasifican todas las imágenes `.jpg`, `.jpeg`, `.png`, `.bmp` o `.webp` de esa carpeta.

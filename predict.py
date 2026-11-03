import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

IMG = 128
N_IMAGENES = 5
labels = ["non_violence", "violence"]

CARPETA = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(CARPETA, "images")


# Cargar el modelo
modelo_local = os.path.join(CARPETA, "baseline.keras")
if os.path.exists(modelo_local):
    ruta_modelo = modelo_local
else:
    from google.colab import drive
    drive.mount("/content/drive")
    ruta_modelo = "/content/drive/MyDrive/violence-detection/baseline.keras"

print("Cargando modelo:", ruta_modelo)
model = tf.keras.models.load_model(ruta_modelo)


# Leer las imágenes
extensiones = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
archivos = sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(extensiones))[:N_IMAGENES]
print("Imágenes encontradas:", len(archivos))

originales = []
imagenes = np.zeros((len(archivos), IMG, IMG, 3), dtype="float32")
for i, archivo in enumerate(archivos):
    imagen = Image.open(os.path.join(IMAGES_DIR, archivo)).convert("RGB")
    originales.append(imagen)
    imagenes[i] = np.array(imagen.resize((IMG, IMG))) / 255.0


# Clasificar
logits = model.predict(imagenes, verbose=0)
probabilidades = tf.nn.softmax(logits).numpy()

print()
for archivo, prob in zip(archivos, probabilidades):
    clase = labels[np.argmax(prob)]
    print(f"{archivo} -> {clase} ({prob.max():.0%})")


# Mostrar
fig, axes = plt.subplots(1, len(archivos), figsize=(4 * len(archivos), 4.5), squeeze=False)
for ax, imagen, archivo, prob in zip(axes[0], originales, archivos, probabilidades):
    clase = labels[np.argmax(prob)]
    ax.imshow(imagen)
    ax.set_title(f"{archivo}\n{clase} ({prob.max():.0%})",
                 color="red" if clase == "violence" else "green")
    ax.axis("off")
plt.tight_layout()
plt.show()

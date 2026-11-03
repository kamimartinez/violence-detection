# CNN sencilla para clasificar violencia / no violencia

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

print("TensorFlow", tf.__version__)

np.random.seed(42)
tf.random.set_seed(42)

IMG = 128          # tamaño de las imágenes (128 x 128)
EPOCHS = 20


# Rutas
from google.colab import drive
drive.mount("/content/drive")

DRIVE = "/content/drive/MyDrive/violence-detection"
FRAMES_DIR = "/content/bianculli_frames"

import kagglehub
path = kagglehub.dataset_download("abdulmananraja/real-life-violence-situations")
DATA_DIR = os.path.join(path, "new_violence")
print("Kaggle:", DATA_DIR)

if not os.path.exists(FRAMES_DIR):
    print("Descomprimiendo fotogramas de prueba...")
    shutil.unpack_archive(os.path.join(DRIVE, "bianculli_frames.zip"), FRAMES_DIR)
print("Bianculli:", FRAMES_DIR)


# Cargar los datos
labels = ["non_violence", "violence"]

splits = pd.read_csv(os.path.join(DRIVE, "splits.csv"))
prueba = pd.read_csv(os.path.join(DRIVE, "bianculli_frames.csv"))
print("splits.csv:", splits.shape)
print("bianculli_frames.csv:", prueba.shape)


def cargar_imagenes(carpeta, archivos):
    imagenes = np.zeros((len(archivos), IMG, IMG, 3), dtype=np.uint8)
    for i, archivo in enumerate(archivos):
        imagen = Image.open(os.path.join(carpeta, archivo)).convert("RGB").resize((IMG, IMG))
        imagenes[i] = np.array(imagen)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1} / {len(archivos)}")
    return imagenes


train = splits[splits["split"] == "train"]
val = splits[splits["split"] == "val"]

print("Cargando train...")
train_images = cargar_imagenes(DATA_DIR, train["archivo"])
train_labels = np.array([labels.index(c) for c in train["clase"]])

print("Cargando val...")
val_images = cargar_imagenes(DATA_DIR, val["archivo"])
val_labels = np.array([labels.index(c) for c in val["clase"]])

print("Cargando test...")
test_images = cargar_imagenes(FRAMES_DIR, prueba["archivo"])
test_labels = np.array([labels.index(c) for c in prueba["clase"]])

print("train:", train_images.shape, train_labels.shape)
print("val:  ", val_images.shape, val_labels.shape)
print("test: ", test_images.shape, test_labels.shape)

for nombre, y in [("train", train_labels), ("val", val_labels), ("test", test_labels)]:
    print(nombre, "->", dict(zip(labels, np.bincount(y).tolist())))

# Escalar los valores entre 0 y 1
train_images = train_images.astype("float32") / 255.0
val_images = val_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
print("Valores de pixel entre", train_images.min(), "y", train_images.max())

# Mostrar una imagen
plt.imshow(train_images[0])
plt.show()
print(f"label : {labels[train_labels[0]]}")


# Construir el modelo
model = Sequential([
    tf.keras.Input(shape=(IMG, IMG, 3)),
    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(2)
])

model.summary()


# Compilar
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

print("loss:", model.loss.name)
print("optimizer:", model.optimizer.name)
print("learning rate:", float(model.optimizer.learning_rate.numpy()))


# Entrenar
print("Entrenando...")
history = model.fit(train_images, train_labels, epochs=EPOCHS,
                    validation_data=(val_images, val_labels))

df = pd.DataFrame(history.history)
print(df)

df.plot(y=["loss", "val_loss"], title="Loss vs Epoch")
plt.show()

df.plot(y=["accuracy", "val_accuracy"], title="Accuracy vs Epoch")
plt.show()


# Evaluar
print("Evaluando en val...")
loss, accuracy = model.evaluate(val_images, val_labels)
print("val loss", loss)
print("val acc", accuracy)

print("Evaluando en test...")
loss, accuracy = model.evaluate(test_images, test_labels)
print("test loss", loss)
print("test acc", accuracy)


# Predicción de una imagen al azar del test
random_inx = np.random.choice(test_images.shape[0])
test_image = test_images[random_inx]

plt.imshow(test_image)
plt.show()
print(f"Label: {labels[test_labels[random_inx]]}")

pred = model.predict(test_image[np.newaxis, ...])
print("Model", labels[np.argmax(pred)])
print(pred)

pd.DataFrame(pred, columns=labels).plot.bar()
plt.show()


# Guardar
model.save(os.path.join(DRIVE, "baseline.keras"))
df.to_csv(os.path.join(DRIVE, "baseline_history.csv"), index=False)
print("Modelo guardado en", os.path.join(DRIVE, "baseline.keras"))

from sklearn.metrics import confusion_matrix, classification_report

for nombre, imagenes, y in [("val", val_images, val_labels), ("test", test_images, test_labels)]:
    pred_clase = np.argmax(model.predict(imagenes, verbose=0), axis=1)
    print(f"\nMatriz de confusión ({nombre}):")
    print(pd.DataFrame(confusion_matrix(y, pred_clase), index=labels, columns=labels))
    print(classification_report(y, pred_clase, target_names=labels, digits=3))
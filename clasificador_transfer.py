"""Etapa (b): clasificador binario con la parte convolucional congelada.

Se carga la base convolucional entrenada con CelebA, se congelan sus pesos y
se entrena solo el clasificador nuevo (una capa densa + la salida).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from metricas import evaluar, mejor_umbral

SEED = 42
SIZE = 64
EPOCHS = 30
BATCH = 32

Path("plots").mkdir(exist_ok=True)
Path("modelos").mkdir(exist_ok=True)
Path("resultados").mkdir(exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def construir_clasificador(base):
    """Congela la base convolucional y agrega el clasificador binario."""
    for capa in base.layers:
        capa.trainable = False

    entradas = layers.Input(shape=(SIZE, SIZE, 3), name="entrada")
    x = base(entradas, training=False)
    # Normalizar las caracteristicas congeladas facilita el ajuste del
    # clasificador (cada dimension queda en media 0 y varianza 1).
    x = layers.LayerNormalization(name="normalizacion")(x)
    x = layers.Dense(128, activation="relu", name="densa_1")(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(1, activation="sigmoid", name="salida")(x)

    modelo = models.Model(entradas, salida, name="clasificador_rostro")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return modelo


def main():
    d = np.load("data/binario.npz")
    base = tf.keras.models.load_model("modelos/conv_base.keras",
                                      compile=False)
    modelo = construir_clasificador(base)
    modelo.summary()

    gen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest")
    train = gen.flow(d["x_train"], d["y_train"], batch_size=BATCH, seed=SEED)

    val_gen = ImageDataGenerator(rescale=1 / 255.0)
    val = val_gen.flow(d["x_val"], d["y_val"], batch_size=BATCH,
                       shuffle=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "modelos/clasificador_frozen.keras", monitor="val_auc",
            mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=6,
            restore_best_weights=True),
    ]
    historia = modelo.fit(train, validation_data=val, epochs=EPOCHS,
                          callbacks=callbacks, verbose=2)

    df = pd.DataFrame(historia.history)
    df.index = df.index + 1
    df.index.name = "epoca"
    df.to_csv("resultados/transfer_historial.csv")

    prob_val = modelo.predict(d["x_val"], verbose=0).ravel()
    umbral = mejor_umbral(d["y_val"], prob_val)
    res_test = evaluar(modelo, d["x_test"], d["y_test"], "frozen", umbral)
    res_train = evaluar(modelo, d["x_train"], d["y_train"], "frozen_train",
                        umbral)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df.index, df["loss"], label="train")
    axes[0].plot(df.index, df["val_loss"], label="val")
    axes[0].set(title="Perdida", xlabel="Epoca", ylabel="loss")
    axes[1].plot(df.index, df["auc"], label="train")
    axes[1].plot(df.index, df["val_auc"], label="val")
    axes[1].set(title="AUC", xlabel="Epoca", ylabel="AUC")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("plots/transfer_entrenamiento.png", dpi=150)
    plt.close(fig)

    print("UMBRAL:", umbral)
    print("TEST :", json.dumps(res_test, indent=2))
    print("TRAIN:", json.dumps(res_train, indent=2))


if __name__ == "__main__":
    main()

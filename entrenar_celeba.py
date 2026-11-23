"""Etapa (a): entrenar la CNN de atributos de CelebA.

Guarda el modelo completo y la parte convolucional (sin el clasificador) para
reutilizarla en el clasificador binario de la etapa (b).
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

SEED = 42
SIZE = 64
EPOCHS = 6
BATCH = 128
LR = 1e-3

Path("plots").mkdir(exist_ok=True)
Path("modelos").mkdir(exist_ok=True)
Path("resultados").mkdir(exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def construir_cnn(n_atributos):
    """CNN pequena para predecir los 40 atributos (multietiqueta)."""
    entradas = layers.Input(shape=(SIZE, SIZE, 3), name="entrada")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(entradas)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D(name="pool")(x)

    x = layers.Dense(256, activation="relu", name="densa_atributos")(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(n_atributos, activation="sigmoid",
                          name="atributos")(x)

    modelo = models.Model(entradas, salida, name="celeba_atributos")
    modelo.compile(optimizer=tf.keras.optimizers.Adam(LR),
                   loss="binary_crossentropy",
                   metrics=["binary_accuracy"])
    return modelo


def main():
    datos = np.load("data/celeba.npz")
    x = datos["imagenes"]
    y = datos["attrs"].astype("float32")
    print(f"CelebA: imagenes={x.shape}, atributos={y.shape[1]}")

    gen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.15)
    train = gen.flow(x, y, batch_size=BATCH, subset="training", seed=SEED)
    val = gen.flow(x, y, batch_size=BATCH, subset="validation", seed=SEED)

    modelo = construir_cnn(y.shape[1])
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "modelos/celeba_cnn.keras", monitor="val_loss",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]
    historia = modelo.fit(train, validation_data=val, epochs=EPOCHS,
                          callbacks=callbacks, verbose=2)

    df = pd.DataFrame(historia.history)
    df.index = df.index + 1
    df.index.name = "epoca"
    df.to_csv("resultados/celeba_historial.csv")

    # Parte convolucional (hasta el pooling global) sin el clasificador.
    base = models.Model(modelo.input, modelo.get_layer("pool").output,
                        name="conv_base")
    base.save("modelos/conv_base.keras")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df.index, df["loss"], label="train")
    axes[0].plot(df.index, df["val_loss"], label="val")
    axes[0].set(title="Perdida (entropia cruzada binaria)", xlabel="Epoca",
                ylabel="loss")
    axes[1].plot(df.index, df["binary_accuracy"], label="train")
    axes[1].plot(df.index, df["val_binary_accuracy"], label="val")
    axes[1].set(title="Exactitud binaria promedio", xlabel="Epoca",
                ylabel="accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("plots/celeba_entrenamiento.png", dpi=150)

    metricas = {
        "epocas_entrenadas": int(len(df)),
        "mejor_val_loss": float(df["val_loss"].min()),
        "mejor_val_binary_accuracy": float(df["val_binary_accuracy"].max()),
        "parametros_cnn": int(modelo.count_params()),
        "parametros_base_convolucional": int(base.count_params()),
    }
    Path("resultados/celeba_metricas.json").write_text(
        json.dumps(metricas, indent=2), encoding="utf-8")
    print(json.dumps(metricas, indent=2))


if __name__ == "__main__":
    main()

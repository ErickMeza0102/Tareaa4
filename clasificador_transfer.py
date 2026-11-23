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
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_curve)
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
SIZE = 64
EPOCHS = 18
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
    x = layers.Dense(128, activation="relu", name="densa_1")(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(1, activation="sigmoid", name="salida")(x)

    modelo = models.Model(entradas, salida, name="clasificador_rostro")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return modelo


def evaluar(modelo, x, y, prefijo):
    """Calcula metricas y guarda la matriz de confusion."""
    prob = modelo.predict(x, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y.astype(int), pred)
    fpr, tpr, _ = roc_curve(y, prob)
    res = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(auc(fpr, tpr)),
        "matriz_confusion": cm.tolist(),
    }
    Path(f"resultados/{prefijo}_metricas.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["otro", "objetivo"],
           yticklabels=["otro", "objetivo"],
           xlabel="Prediccion", ylabel="Real",
           title=f"Matriz de confusion ({prefijo})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"plots/confusion_{prefijo}.png", dpi=150)
    plt.close(fig)
    return res


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
            monitor="val_auc", mode="max", patience=5,
            restore_best_weights=True),
    ]
    historia = modelo.fit(train, validation_data=val, epochs=EPOCHS,
                          callbacks=callbacks, verbose=2)

    df = pd.DataFrame(historia.history)
    df.index = df.index + 1
    df.index.name = "epoca"
    df.to_csv("resultados/transfer_historial.csv")

    res_test = evaluar(modelo, d["x_test"], d["y_test"], "frozen")
    res_train = evaluar(modelo, d["x_train"], d["y_train"], "frozen_train")

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

    print("TEST :", json.dumps(res_test, indent=2))
    print("TRAIN:", json.dumps(res_train, indent=2))


if __name__ == "__main__":
    main()

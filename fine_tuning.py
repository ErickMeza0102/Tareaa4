"""Etapa (c): fine-tuning de las ultimas capas convolucionales.

Se parte del clasificador con la base congelada, se descongela el ultimo
bloque convolucional (manteniendo BatchNorm congelado) y se entrena con una
tasa de aprendizaje pequena.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from clasificador_transfer import evaluar

SEED = 42
BATCH = 32
EPOCHS = 12
LR = 1e-4
CAPAS_DESCONGELADAS = 6

Path("plots").mkdir(exist_ok=True)
Path("modelos").mkdir(exist_ok=True)
Path("resultados").mkdir(exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def graficar_predicciones(modelo, x, y, ruta, n=8):
    prob = modelo.predict(x, verbose=0).ravel()
    idx = np.random.default_rng(0).choice(len(x), size=min(n, len(x)),
                                          replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i])
        real = "objetivo" if y[i] > 0.5 else "otro"
        pred = "objetivo" if prob[i] >= 0.5 else "otro"
        ax.set_title(f"real: {real}\npred: {pred} ({prob[i]:.2f})", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    d = np.load("data/binario.npz")
    modelo = tf.keras.models.load_model("modelos/clasificador_frozen.keras")
    base = modelo.get_layer("conv_base")

    # Descongelar solo el ultimo bloque convolucional; BatchNorm se mantiene
    # congelado para no perder las estadisticas del pre-entrenamiento.
    base.trainable = True
    for capa in base.layers[:-CAPAS_DESCONGELADAS]:
        capa.trainable = False
    for capa in base.layers:
        if isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
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
            "modelos/clasificador_finetuned.keras", monitor="val_auc",
            mode="max", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4,
            restore_best_weights=True),
    ]
    historia = modelo.fit(train, validation_data=val, epochs=EPOCHS,
                          callbacks=callbacks, verbose=2)

    df = pd.DataFrame(historia.history)
    df.index = df.index + 1
    df.index.name = "epoca"
    df.to_csv("resultados/finetuning_historial.csv")

    res_test = evaluar(modelo, d["x_test"], d["y_test"], "finetuned")
    res_train = evaluar(modelo, d["x_train"], d["y_train"], "finetuned_train")

    prob = modelo.predict(d["x_test"], verbose=0).ravel()
    fpr, tpr, _ = roc_curve(d["y_test"], prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title="Curva ROC (prueba)", xlabel="Falsos positivos",
           ylabel="Verdaderos positivos")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/roc_finetuned.png", dpi=150)
    plt.close(fig)

    graficar_predicciones(modelo, d["x_test"], d["y_test"],
                          "plots/ejemplos_predicciones.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df.index, df["loss"], label="train")
    axes[0].plot(df.index, df["val_loss"], label="val")
    axes[0].set(title="Perdida (fine-tuning)", xlabel="Epoca", ylabel="loss")
    axes[1].plot(df.index, df["auc"], label="train")
    axes[1].plot(df.index, df["val_auc"], label="val")
    axes[1].set(title="AUC (fine-tuning)", xlabel="Epoca", ylabel="AUC")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("plots/finetuning_entrenamiento.png", dpi=150)
    plt.close(fig)

    print("TEST :", json.dumps(res_test, indent=2))
    print("TRAIN:", json.dumps(res_train, indent=2))


if __name__ == "__main__":
    main()

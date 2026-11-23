"""Etapa (c): fine-tuning en dos fases.

La clave es construir el modelo llamando a la base convolucional con
`base(entradas)` (sin `training=False`), para que las capas
BatchNormalization usen el modo de entrenamiento y puedan adaptar sus
estadisticas al nuevo dominio (LFW). Si se dejan congeladas en modo
inferencia, la base no se adapta y el clasificador se queda en el azar.

Fase 1 (calentamiento): se descongela el ultimo bloque convolucional, con
        BatchNorm congelado y lr = 1e-3 durante pocas epocas.
Fase 2 (refinamiento): se descongela toda la base (BatchNorm incluido),
        lr = 5e-4 con ReduceLROnPlateau y early stopping.
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
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from metricas import evaluar, mejor_umbral

SEED = 42
SIZE = 64
BATCH = 32
EPOCHS_FASE1 = 3
EPOCHS_FASE2 = 30
LR_FASE1 = 1e-3
LR_FASE2 = 5e-4
CAPAS_FASE1 = 6

Path("plots").mkdir(exist_ok=True)
Path("modelos").mkdir(exist_ok=True)
Path("resultados").mkdir(exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


def configurar_base(base, n_capas, congelar_bn):
    """Descongela las ultimas n_capas (None = todas)."""
    base.trainable = True
    total = len(base.layers)
    for i, capa in enumerate(base.layers):
        capa.trainable = (n_capas is None) or (i >= total - n_capas)
        if congelar_bn and isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False


def compilar(modelo, lr):
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])


def graficar_predicciones(modelo, x, y, ruta, umbral, n=8):
    prob = modelo.predict(x, verbose=0).ravel()
    idx = np.random.default_rng(0).choice(len(x), size=min(n, len(x)),
                                          replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(x[i])
        real = "objetivo" if y[i] > 0.5 else "otro"
        pred = "objetivo" if prob[i] >= umbral else "otro"
        acierto = "ok" if real == pred else "error"
        ax.set_title(f"real: {real}\npred: {pred} ({prob[i]:.2f}) - {acierto}",
                     fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    d = np.load("data/binario.npz")
    base = tf.keras.models.load_model("modelos/conv_base.keras",
                                      compile=False)

    # Modelo nuevo: la base hereda el modo de entrenamiento (BN se adapta).
    entradas = layers.Input(shape=(SIZE, SIZE, 3), name="entrada")
    x = base(entradas)
    x = layers.LayerNormalization(name="normalizacion")(x)
    x = layers.Dense(128, activation="relu", name="densa_1")(x)
    x = layers.Dropout(0.4)(x)
    salida = layers.Dense(1, activation="sigmoid", name="salida")(x)
    modelo = models.Model(entradas, salida, name="clasificador_finetuned")

    gen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.12,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest")
    train = gen.flow(d["x_train"], d["y_train"], batch_size=BATCH, seed=SEED)
    val = ImageDataGenerator(rescale=1 / 255.0).flow(
        d["x_val"], d["y_val"], batch_size=BATCH, shuffle=False)

    historias = []

    # ---------------- Fase 1: ultimo bloque convolucional ----------------
    configurar_base(base, CAPAS_FASE1, congelar_bn=True)
    compilar(modelo, LR_FASE1)
    h1 = modelo.fit(train, validation_data=val, epochs=EPOCHS_FASE1,
                    verbose=2)
    historias.append(pd.DataFrame(h1.history))

    # ---------------- Fase 2: toda la base ----------------
    configurar_base(base, None, congelar_bn=False)
    compilar(modelo, LR_FASE2)
    modelo.summary()
    h2 = modelo.fit(
        train, validation_data=val, epochs=EPOCHS_FASE2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "modelos/clasificador_finetuned.keras", monitor="val_auc",
                mode="max", save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=8,
                restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.5, patience=3,
                min_lr=1e-5),
        ],
        verbose=2)
    historias.append(pd.DataFrame(h2.history))

    df = pd.concat(historias, ignore_index=True)
    df.index = df.index + 1
    df.index.name = "epoca"
    df.to_csv("resultados/finetuning_historial.csv")

    prob_val = modelo.predict(d["x_val"], verbose=0).ravel()
    umbral = mejor_umbral(d["y_val"], prob_val)
    res_test = evaluar(modelo, d["x_test"], d["y_test"], "finetuned", umbral)
    res_train = evaluar(modelo, d["x_train"], d["y_train"],
                        "finetuned_train", umbral)

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
                          "plots/ejemplos_predicciones.png", umbral)

    corte = len(h1.history["loss"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df.index, df["loss"], label="train")
    axes[0].plot(df.index, df["val_loss"], label="val")
    axes[0].axvline(corte + 0.5, color="gray", linestyle=":")
    axes[0].set(title="Perdida (fine-tuning)", xlabel="Epoca", ylabel="loss")
    axes[1].plot(df.index, df["auc"], label="train")
    axes[1].plot(df.index, df["val_auc"], label="val")
    axes[1].axvline(corte + 0.5, color="gray", linestyle=":")
    axes[1].set(title="AUC (fine-tuning)", xlabel="Epoca", ylabel="AUC")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig("plots/finetuning_entrenamiento.png", dpi=150)
    plt.close(fig)

    print("UMBRAL:", umbral)
    print("TEST :", json.dumps(res_test, indent=2))
    print("TRAIN:", json.dumps(res_train, indent=2))


if __name__ == "__main__":
    main()

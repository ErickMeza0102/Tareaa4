"""Metricas y graficas compartidas por las etapas de la tarea 4."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)


def mejor_umbral(y, prob):
    """Umbral que maximiza el indice de Youden (TPR - FPR) en validacion."""
    fpr, tpr, umbrales = roc_curve(y, prob)
    j = tpr - fpr
    umbral = float(umbrales[np.argmax(j)])
    if not np.isfinite(umbral):
        return 0.5
    return float(np.clip(umbral, 0.0, 1.0))


def evaluar(modelo, x, y, prefijo, umbral=0.5):
    """Calcula metricas con un umbral dado y guarda la matriz de confusion."""
    prob = modelo.predict(x, verbose=0).ravel()
    pred = (prob >= umbral).astype(int)
    cm = confusion_matrix(y.astype(int), pred)
    fpr, tpr, _ = roc_curve(y, prob)
    res = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(auc(fpr, tpr)),
        "umbral": float(umbral),
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

"""Prepara los datos de la tarea 4 (reconocimiento facial).

1. Descarga un subconjunto de CelebA (imagenes 64x64 + 40 atributos) para
   pre-entrenar la parte convolucional.
2. Descarga LFW y elige la identidad con mas fotos como "mi rostro"; el resto
   de identidades se usa como clase negativa.
3. Guarda todo en data/ para que los scripts de entrenamiento sean rapidos y
   reproducibles.
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from sklearn.datasets import fetch_lfw_people

N_CELEBA = 10000
SIZE = 64
SEED = 42
MIN_FOTOS_IDENTIDAD = 70

DATA = Path("data")
DATA.mkdir(exist_ok=True)


def preparar_celeba():
    print("Descargando CelebA (subconjunto)...")
    ds = load_dataset("flwrlabs/celeba", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=2000, seed=SEED)

    imagenes, attrs, ids = [], [], []
    atributos = None
    for ex in ds:
        if atributos is None:
            atributos = [k for k in ex if k not in ("image", "celeb_id")]
        img = ex["image"].convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        imagenes.append(np.asarray(img, dtype=np.uint8))
        attrs.append([1 if ex[a] else 0 for a in atributos])
        ids.append(int(ex["celeb_id"]))
        if len(imagenes) % 500 == 0:
            print(f"  CelebA: {len(imagenes)}/{N_CELEBA}", flush=True)
        if len(imagenes) >= N_CELEBA:
            break

    imagenes = np.stack(imagenes)
    attrs = np.asarray(attrs, dtype=np.uint8)
    ids = np.asarray(ids, dtype=np.int32)
    np.savez(DATA / "celeba.npz", imagenes=imagenes, attrs=attrs, ids=ids)
    print(f"CelebA listo: {imagenes.shape}, atributos={len(atributos)}")
    return atributos


def preparar_lfw():
    print("Descargando LFW...")
    lfw = fetch_lfw_people(min_faces_per_person=MIN_FOTOS_IDENTIDAD,
                           color=True, resize=0.5)
    nombres = list(lfw.target_names)
    y = np.asarray(lfw.target)
    conteo = np.bincount(y)
    idx_objetivo = int(np.argmax(conteo))
    print(f"  LFW: {lfw.images.shape}, identidades={len(nombres)}")
    print(f"  identidad objetivo: {nombres[idx_objetivo]} "
          f"({conteo[idx_objetivo]} fotos)")

    x = np.empty((len(lfw.images), SIZE, SIZE, 3), dtype=np.uint8)
    for i, img in enumerate(lfw.images):
        arr = np.asarray(img)
        # sklearn entrega las imagenes como float en [0, 1]; hay que llevar
        # ese rango a 0-255 antes de convertirlas a uint8.
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
        x[i] = np.asarray(pil.resize((SIZE, SIZE), Image.BILINEAR))

    pos = np.where(y == idx_objetivo)[0]
    neg = np.where(y != idx_objetivo)[0]
    rng = np.random.default_rng(SEED)
    rng.shuffle(pos)
    rng.shuffle(neg)
    neg = neg[: len(pos)]

    def tres(idx):
        n = len(idx)
        a, b = int(0.7 * n), int(0.85 * n)
        return idx[:a], idx[a:b], idx[b:]

    p_tr, p_va, p_te = tres(pos)
    n_tr, n_va, n_te = tres(neg)

    def armar(p, n):
        xx = np.concatenate([x[p], x[n]])
        yy = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
        orden = rng.permutation(len(xx))
        return xx[orden], yy[orden].astype("float32")

    x_train, y_train = armar(p_tr, n_tr)
    x_val, y_val = armar(p_va, n_va)
    x_test, y_test = armar(p_te, n_te)

    np.savez(DATA / "binario.npz",
             x_train=x_train, y_train=y_train,
             x_val=x_val, y_val=y_val,
             x_test=x_test, y_test=y_test)

    print(f"  binario: train={len(x_train)} val={len(x_val)} test={len(x_test)}")
    return nombres, nombres[idx_objetivo], int(conteo[idx_objetivo]), idx_objetivo


def main():
    atributos = preparar_celeba()
    nombres, objetivo, n_pos, idx_obj = preparar_lfw()

    meta = {
        "size": SIZE,
        "atributos": atributos,
        "celeba_imagenes": N_CELEBA,
        "lfw_identidad_objetivo": objetivo,
        "lfw_positivos": n_pos,
        "lfw_identidades": nombres,
        "lfw_indice_objetivo": idx_obj,
    }
    (DATA / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Datos guardados en", DATA.resolve())


if __name__ == "__main__":
    main()

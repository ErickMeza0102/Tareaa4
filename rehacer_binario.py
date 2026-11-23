"""Regenera solo el conjunto binario de LFW (sin volver a bajar CelebA)."""

import json

from preparar_datos import DATA, preparar_lfw

nombres, objetivo, n_pos, idx_obj = preparar_lfw()
ruta = DATA / "meta.json"
meta = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
meta.update({
    "lfw_identidad_objetivo": objetivo,
    "lfw_positivos": n_pos,
    "lfw_identidades": nombres,
    "lfw_indice_objetivo": idx_obj,
})
ruta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print("data/binario.npz regenerado")

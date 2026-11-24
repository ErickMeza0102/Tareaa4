# Tarea 4: Reconocimiento facial con transfer learning

Repositorio: https://github.com/ErickMeza0102/Tareaa4

El objetivo es entrenar una red que distinga "mi rostro" del resto, usando
**transfer learning**: primero se pre-entrena una CNN con los 40 atributos de
**CelebA** y después se reutiliza su parte convolucional en un clasificador
binario.

Como no se cuentan con fotos propias, se usa una **identidad pública de LFW**
(la persona con más fotos del conjunto) como "mi rostro"; el resto de
identidades son la clase negativa. El procedimiento es exactamente el mismo
que se seguiría con fotos propias.

## Etapas

| Etapa | Script | Qué hace |
|---|---|---|
| Datos | `preparar_datos.py` | Descarga 10 000 imágenes de CelebA (64×64) y LFW; arma los conjuntos balanceados. |
| (a) Pre-entrenamiento | `entrenar_celeba.py` | CNN con 40 salidas sigmoide para los atributos de CelebA. Guarda `conv_base`. |
| (b) Clasificador congelado | `clasificador_transfer.py` | Congela `conv_base`, agrega una capa densa y una salida; entrena solo el clasificador. |
| (c) Fine-tuning | `fine_tuning.py` | Calienta el último bloque y luego descongela toda la base con `lr=5e-4`. |

## Resultados

Resumen sobre el conjunto de prueba (160 imágenes balanceadas):

| Modelo | AUC | Exactitud | F1 |
|---|---|---|---|
| (a) CNN de atributos de CelebA | — | 0.797 (exactitud binaria media por atributo) | — |
| (b) Base congelada + clasificador | 0.578 | 0.556 | 0.523 |
| (c) Fine-tuning | **0.967** | **0.888** | **0.886** |
| Referencia: misma CNN desde cero | 0.829 | — | — |

La base congelada no alcanza a separar identidades porque las características
pre-entrenadas describen atributos (sonrisa, lentes, pelo), no identidades. El
fine-tuning, en cambio, adapta las capas convolucionales y las estadísticas de
BatchNorm al nuevo dominio y supera incluso al mismo modelo entrenado desde
cero. Los detalles están en `reporte.pdf`.

## Archivos

- `preparar_datos.py`: descarga y prepara CelebA + LFW.
- `entrenar_celeba.py`: etapa (a), atributos de CelebA.
- `clasificador_transfer.py`: etapa (b), base congelada.
- `fine_tuning.py`: etapa (c), refinamiento.
- `metricas.py`: métricas, umbral de validación y matrices de confusión.
- `rehacer_binario.py`: regenera sólo `data/binario.npz` sin volver a bajar CelebA.
- `modelos/`: `celeba_cnn.keras`, `conv_base.keras`, `clasificador_frozen.keras`, `clasificador_finetuned.keras`.
- `plots/`: curvas de entrenamiento, matrices de confusión, ROC y ejemplos.
- `resultados/`: historiales CSV y métricas JSON.
- `reporte.pdf`: reporte de la tarea.
- `reporte_ia.pdf`: reporte de uso de IA.

## Ejecución

```bash
pip install -r requirements.txt
python preparar_datos.py
python entrenar_celeba.py
python clasificador_transfer.py
python fine_tuning.py
```

Los datos se guardan en `data/` (no se suben al repositorio por tamaño). El
resto de artefactos (modelos, gráficas, métricas) sí se versiona.

## Notas

- Todo se entrenó en CPU, con imágenes de 64×64 y un subconjunto de CelebA
  para que el tiempo sea razonable. La arquitectura es la misma que se usaría
  con imágenes de 224×224, solo cambia la resolución de entrada.
- Las métricas finales están en `resultados/` y se resumen en `reporte.pdf`.

# Manifiesto del dataset

## Winoground (fuente principal)

| Campo | Valor |
|---|---|
| Identificador | `facebook/winoground` en HuggingFace |
| Referencia | Thrush et al., *Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality*, CVPR 2022, arXiv:2204.03162 |
| Revisión usada | `b400e173549071916ad1b3d449293bc8d8b4b763` |
| Tamaño | 400 ejemplos = 800 imágenes + 800 captions |
| Formato | Parquet con las imágenes embebidas (367 MB) |
| Acceso | **Gated.** Hay que iniciar sesión en HuggingFace y aceptar el acuerdo de licencia. La aprobación es automática, pero sin ella la descarga devuelve 401. |

### Restricciones de uso

El acuerdo de licencia del dataset restringe su uso a **investigación no comercial**. Las
imágenes proceden de Getty Images y **no se redistribuyen** en este repositorio: solo se
versionan los *embeddings* derivados (`data/embeddings/winoground_real_vitb32.npz`) y las
matrices de similitud, que no permiten reconstruir las imágenes originales.

`.gitignore` excluye `data/winoground_cache/` justamente por esto.

### Estructura relevante

| Columna | Uso en este proyecto |
|---|---|
| `id` | 0–399, clave del ejemplo |
| `image_0`, `image_1` | las dos imágenes del par mínimo |
| `caption_0`, `caption_1` | los dos captions; comparten exactamente el mismo conjunto de palabras |
| `collapsed_tag` | `Object` (141), `Relation` (233), `Both` (26); base del análisis por tag |

Convención del scorer: `caption_0` corresponde a `image_0` y `caption_1` a `image_1`.

## Pesos de modelos

| Modelo | Origen | Ubicación local |
|---|---|---|
| CLIP ViT-B/32, ViT-B/16, ViT-L/14 | OpenCLIP (`open_clip.create_model_and_transforms`) | caché de HuggingFace |
| BLIP ITM base COCO | `Salesforce/blip-itm-base-coco` | `data/models/blip-itm-base-coco/` |

**Por qué BLIP está en `data/models/` y no en la caché de HuggingFace:** en este entorno las
descargas iniciadas desde Python se quedan colgadas indefinidamente — el proceso queda a 0 %
de CPU sin abrir socket — mientras que `curl` funciona con normalidad. Los pesos se
descargaron con:

```bash
curl -L -H "Authorization: Bearer $HF_TOKEN" \
  -o data/models/blip-itm-base-coco/pytorch_model.bin \
  https://huggingface.co/Salesforce/blip-itm-base-coco/resolve/main/pytorch_model.bin
```

`src/blip_utils.py::resolve_model_id` prefiere esa copia local si existe, de modo que la
ejecución es reproducible y completamente offline. El directorio está en `.gitignore`
(895 MB).

## Datos derivados que sí se versionan

| Ruta | Contenido | Regenerable con |
|---|---|---|
| `data/winoground_meta.csv` | ids, captions y tags; sin imágenes | `scripts/10_export_sims.py` |
| `data/embeddings/*.npz` | embeddings de CLIP ViT-B/32 | `scripts/02_run_winoground.py` |
| `outputs/sims/*.npz` | matrices 2×2 por ejemplo y modelo | `scripts/10_export_sims.py` |

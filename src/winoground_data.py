"""Carga de datos para la evaluación composicional.

Dos fuentes:
  - REAL: el benchmark oficial Winoground (facebook/winoground, gated en HF).
    Se descarga el parquet con imágenes embebidas vía huggingface_hub.
  - CURADO (fallback offline): un conjunto pequeño de pares mínimos
    composicionales generados con PIL (binding color-objeto y relaciones
    espaciales), 100% reproducible sin red. Mismo formato y mismo scorer.

Formato común de cada ejemplo (dataclass `Example`):
    id, image_0 (PIL), image_1 (PIL), caption_0, caption_1, tag
La convención del scorer es: caption_0 ↔ image_0, caption_1 ↔ image_1.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image


@dataclass
class Example:
    id: str
    image_0: Image.Image
    image_1: Image.Image
    caption_0: str
    caption_1: str
    tag: str = ""


# --------------------------------------------------------------------------- #
# Fuente REAL: Winoground oficial (gated)                                       #
# --------------------------------------------------------------------------- #
def load_winoground_real(cache_dir: str | Path = "data/winoground_cache") -> List[Example]:
    """Descarga y parsea el Winoground oficial. Lanza excepción si está gated
    y la cuenta no aceptó la licencia."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(
        "facebook/winoground",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    df = pd.read_parquet(path)

    def to_img(cell) -> Image.Image:
        # HF Image: dict {'bytes': ..., 'path': ...}
        if isinstance(cell, dict) and cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"]))
        if isinstance(cell, (bytes, bytearray)):
            return Image.open(io.BytesIO(cell))
        raise TypeError(f"formato de imagen no reconocido: {type(cell)}")

    examples: List[Example] = []
    for _, row in df.iterrows():
        examples.append(
            Example(
                id=str(row["id"]),
                image_0=to_img(row["image_0"]),
                image_1=to_img(row["image_1"]),
                caption_0=str(row["caption_0"]),
                caption_1=str(row["caption_1"]),
                # collapsed_tag (Object/Relation/Both) es el agrupamiento canónico
                # del paper para el análisis de error; cae a 'tag' si no existe.
                tag=str(row.get("collapsed_tag") or row.get("tag", "")),
            )
        )
    return examples


# --------------------------------------------------------------------------- #
# Fuente CURADA: pares mínimos offline                                          #
# --------------------------------------------------------------------------- #
def load_curated(data_dir: str | Path = "data/curated") -> List[Example]:
    data_dir = Path(data_dir)
    manifest = data_dir / "examples.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"No existe {manifest}. Genera el set curado con scripts/01_prepare_data.py."
        )
    examples: List[Example] = []
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            examples.append(
                Example(
                    id=str(rec["id"]),
                    image_0=Image.open(data_dir / rec["image_0"]).convert("RGB"),
                    image_1=Image.open(data_dir / rec["image_1"]).convert("RGB"),
                    caption_0=rec["caption_0"],
                    caption_1=rec["caption_1"],
                    tag=rec.get("tag", ""),
                )
            )
    return examples


def load_dataset(prefer_real: bool = True, curated_dir: str | Path = "data/curated",
                 cache_dir: str | Path = "data/winoground_cache") -> Tuple[List[Example], str]:
    """Devuelve (ejemplos, fuente). fuente ∈ {'winoground_real', 'curated'}.

    Intenta el real; si no hay acceso/red, cae al curado (etiquetado como tal)."""
    if prefer_real:
        try:
            ex = load_winoground_real(cache_dir)
            return ex, "winoground_real"
        except Exception as exc:  # noqa: BLE001
            print(f"[winoground_data] No se pudo cargar el Winoground real ({type(exc).__name__}). "
                  f"Usando set curado offline.")
    return load_curated(curated_dir), "curated"

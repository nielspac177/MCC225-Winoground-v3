"""Exporta las matrices de similitud 2x2 por ejemplo a una caché reutilizable.

## Por qué existe este script

El pipeline principal (`02_run_winoground.py`) acopla tres cosas: cargar 800
imágenes, ejecutar modelos y analizar resultados. Eso obliga a pagar el coste
completo (importar torch, decodificar el parquet, mover tensores) cada vez que
se quiere recalcular una métrica, aunque la métrica sea una comparación entre
cuatro números.

Este script rompe ese acoplamiento. Todo lo que cualquier métrica de Winoground
necesita es, por ejemplo, una matriz 2x2. Una vez exportadas, el análisis
completo — empates, GroupMatch, bootstrap, desglose por tag — corre con numpy
puro en menos de un segundo y sin importar torch.

Beneficio adicional no trivial: evita el interbloqueo entre el OpenMP de FAISS
y el de PyTorch en macOS, que cuelga `02_run_winoground.py` en su bloque final
(dos runtimes `libomp` en el mismo proceso). El análisis ya no comparte proceso
con ninguno de los dos.

Salidas:
  data/winoground_meta.csv   metadatos por ejemplo (id, captions, tags)
  outputs/sims/<clave>.npz   sims (N,2,2) float64 + etiqueta del modelo

Uso:
    python scripts/10_export_sims.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EMB = ROOT / "data" / "winoground_cache" / "embeddings"
SIMS = ROOT / "outputs" / "sims"
META = ROOT / "data" / "winoground_meta.csv"

# Etiquetas legibles para las claves de caché de embeddings.
ETIQUETAS = {
    "winoground_real__ViT-B-32__laion2b_s34b_b79k": "CLIP ViT-B/32 (laion2b)",
    "winoground_real__ViT-B-16__datacomp_xl_s13b_b90k": "CLIP ViT-B/16 (datacomp_xl)",
    "winoground_real__ViT-L-14-quickgelu__openai": "CLIP ViT-L/14 (openai)",
}


def exportar_metadatos() -> None:
    """Vuelca los metadatos del parquet oficial SIN decodificar las imágenes.

    Leer solo las columnas de texto evita los ~46 s de decodificación de las
    800 imágenes, que este análisis no necesita.
    """
    if META.exists():
        print(f"[meta] ya existe: {META.relative_to(ROOT)}")
        return
    import pandas as pd
    from huggingface_hub import hf_hub_download

    ruta = hf_hub_download(
        "facebook/winoground",
        "data/test-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=str(ROOT / "data" / "winoground_cache"),
    )
    df = pd.read_parquet(
        ruta,
        columns=["id", "caption_0", "caption_1", "tag", "secondary_tag",
                 "collapsed_tag", "num_main_preds"],
    )
    META.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(META, index=False)
    print(f"[meta] {len(df)} ejemplos -> {META.relative_to(ROOT)}")


def sims_desde_embeddings(cap: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Convierte embeddings intercalados en matrices 2x2 por ejemplo.

    La convención de `02_run_winoground.py` es intercalada: las filas 2k y 2k+1
    de `cap` e `img` pertenecen al ejemplo k. La matriz resultante cumple
    sim[c, i] = similitud(caption_c, imagen_i), igual que el scorer oficial.
    """
    if cap.shape != img.shape:
        raise ValueError(f"formas distintas: cap {cap.shape} vs img {img.shape}")
    if cap.shape[0] % 2:
        raise ValueError(f"se esperaba un número par de filas, hay {cap.shape[0]}")
    n = cap.shape[0] // 2
    sims = np.empty((n, 2, 2), dtype=np.float64)
    for k in range(n):
        sims[k] = cap[2 * k : 2 * k + 2] @ img[2 * k : 2 * k + 2].T
    return sims


def main() -> None:
    exportar_metadatos()
    SIMS.mkdir(parents=True, exist_ok=True)

    fuentes = sorted(EMB.glob("winoground_real__*.npz"))
    if not fuentes:
        raise SystemExit(
            f"No hay embeddings en {EMB}. Ejecuta antes `make run` "
            "para poblar la caché."
        )

    for ruta in fuentes:
        clave = ruta.stem
        z = np.load(ruta)
        sims = sims_desde_embeddings(z["cap"].astype(np.float64),
                                     z["img"].astype(np.float64))
        destino = SIMS / f"{clave}.npz"
        np.savez_compressed(
            destino,
            sims=sims,
            etiqueta=ETIQUETAS.get(clave, clave),
            n_params=z["n_params"] if "n_params" in z.files else np.int64(0),
        )
        print(f"[sims] {ETIQUETAS.get(clave, clave):32s} {sims.shape} -> "
              f"{destino.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

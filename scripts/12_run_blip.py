"""Evalúa BLIP sobre Winoground por las dos rutas: ITC (dual) e ITM (cross-attention).

Ablación controlada de §11.2.1 y §11.2.3: mismos pesos, mismo preprocesado,
misma data; la única variable que cambia es si existe atención cruzada entre
los tokens de texto y los parches de imagen.

Escribe en la misma caché que `10_export_sims.py`, de modo que el análisis
posterior trata a BLIP igual que a cualquier checkpoint de CLIP.

Salidas:
  outputs/sims/winoground_real__BLIP-base__itc.npz
  outputs/sims/winoground_real__BLIP-base__itm.npz

Uso:
    python scripts/12_run_blip.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import blip_utils as bu           # noqa: E402
from src.winoground_data import load_dataset  # noqa: E402
from src.winoground_eval import aggregate     # noqa: E402

SIMS = ROOT / "outputs" / "sims"

RUTAS = [
    (False, "itc", "BLIP-base ITC (dual-encoder)"),
    (True, "itm", "BLIP-base ITM (cross-attention)"),
]


def main() -> None:
    SIMS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("[blip] cargando Winoground...", flush=True)
    ejemplos, fuente = load_dataset(prefer_real=True)
    print(f"[blip] fuente={fuente}  N={len(ejemplos)}  ({time.time() - t0:.0f}s)", flush=True)
    if fuente != "winoground_real":
        raise SystemExit(
            "Se requiere el benchmark oficial. El set curado no sirve para esta "
            "comparación porque sus pares no son los del paper."
        )

    for use_itm, sufijo, etiqueta in RUTAS:
        t = time.time()
        print(f"[blip] {etiqueta} (use_itm_head={use_itm})", flush=True)
        sims, n_params = bu.winoground_sims(ejemplos, use_itm_head=use_itm)
        s = aggregate(sims)
        seg = time.time() - t
        destino = SIMS / f"winoground_real__BLIP-base__{sufijo}.npz"
        np.savez_compressed(
            destino, sims=sims, etiqueta=etiqueta,
            n_params=np.int64(n_params), segundos=np.float64(seg),
        )
        print(
            f"  -> text={s.text:.4f} image={s.image:.4f} group={s.group:.4f} "
            f"group_match={s.group_match:.4f}  ({seg:.0f}s)",
            flush=True,
        )
        print(f"  -> {destino.relative_to(ROOT)}", flush=True)

    print(f"[blip] listo en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

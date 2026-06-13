#!/usr/bin/env python3
"""Valida el scorer contra los scores OFICIALES de CLIP del dataset Winoground.

Descarga `statistics/model_scores/clip.jsonl` (1600 = 400×4 scores crudos de CLIP
ViT-B/32 del paper), reconstruye la matriz 2x2 por ejemplo y aplica NUESTRO scorer.
Debe reproducir text=0.3075, image=0.1050, group=0.0800.

Si el dataset está gated sin acceso, sale con aviso (no falla en CI).
Uso:  python scripts/validate_against_official.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.winoground_eval import aggregate  # noqa: E402

EXPECTED = {"text": 0.3075, "image": 0.1050, "group": 0.0800}


def main():
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("facebook/winoground", "statistics/model_scores/clip.jsonl",
                            repo_type="dataset", cache_dir=str(ROOT / "data" / "winoground_cache"))
    except Exception as exc:  # noqa: BLE001
        print(f"[validate] sin acceso al dataset oficial ({type(exc).__name__}); "
              f"acepta la licencia para validar. Omitido.")
        return 0

    scores = {}
    for line in open(p):
        r = json.loads(line)
        m = re.match(r"(\d+)_c(\d)_i(\d)", r["label"])
        eid, c, i = int(m.group(1)), int(m.group(2)), int(m.group(3))
        scores.setdefault(eid, np.zeros((2, 2)))[c, i] = r["score"]
    sims = [scores[k] for k in sorted(scores)]
    agg = aggregate(sims)
    got = {"text": round(agg.text, 4), "image": round(agg.image, 4), "group": round(agg.group, 4)}
    print(f"[validate] n={agg.n}  obtenido={got}  esperado={EXPECTED}")

    ok = all(abs(got[k] - EXPECTED[k]) < 1e-3 for k in EXPECTED)
    if ok:
        print("[validate] OK ✅  el scorer reproduce los scores oficiales de CLIP.")
        return 0
    print("[validate] FALLO ❌  el scorer NO coincide con los scores oficiales.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

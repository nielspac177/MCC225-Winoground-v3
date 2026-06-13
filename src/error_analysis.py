"""Análisis de error de Winoground por tag y casos cualitativos.

(Responde la pregunta 8.6.3: ¿qué tipo de error aparece en Winoground?)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .winoground_eval import image_correct, per_example_scores, text_correct


def scores_by_tag(sims: List[np.ndarray], tags: List[str]) -> pd.DataFrame:
    """Promedia text/image/group por tag."""
    rows = per_example_scores(sims)
    df = pd.DataFrame(rows)
    df["tag"] = [t if t else "(sin_tag)" for t in tags]
    grouped = df.groupby("tag")[["text", "image", "group"]].mean()
    counts = df.groupby("tag").size().rename("n")
    out = grouped.join(counts).reset_index()
    return out.sort_values("group")


def failure_cases(sims: List[np.ndarray], examples, max_cases: int = 8) -> List[Dict]:
    """Selecciona ejemplos con group=0 y anota qué condición falló (text/image)."""
    cases: List[Dict] = []
    for k, (sim, ex) in enumerate(zip(sims, examples)):
        t = text_correct(sim)
        i = image_correct(sim)
        if not (t and i):
            failed = []
            if not t:
                failed.append("text")
            if not i:
                failed.append("image")
            cases.append({
                "index": k,
                "id": getattr(ex, "id", str(k)),
                "tag": getattr(ex, "tag", ""),
                "caption_0": getattr(ex, "caption_0", ""),
                "caption_1": getattr(ex, "caption_1", ""),
                "text_ok": int(t),
                "image_ok": int(i),
                "failed": "+".join(failed),
                "sim": np.asarray(sim).round(3).tolist(),
            })
    return cases[:max_cases]

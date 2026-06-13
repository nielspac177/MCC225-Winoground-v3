#!/usr/bin/env python3
"""Pipeline principal: evalúa OpenCLIP en Winoground (razonamiento composicional).

Genera en outputs/metrics/:
  - scores.json                 scores del checkpoint primario + fuente + entorno
  - per_example.csv             text/image/group por ejemplo
  - bootstrap_ci.csv            IC bootstrap de los 3 scores
  - checkpoint_comparison.csv   los 3 scores por checkpoint (+ #params, tiempo)
  - by_tag.csv                  scores por tag (object/relation/...)
  - blindness.json              prueba de ceguera (real vs imágenes permutadas)
  - recall_vs_group.json        Recall@K (alto) vs group score (≈azar): la tesis
  - faiss_demo.csv              búsqueda semántica con FAISS
  - failure_cases.json          casos cualitativos con group=0

Uso:  python scripts/02_run_winoground.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import openclip_utils as oc          # noqa: E402
from src.blindness_probe import run_blindness_probe  # noqa: E402
from src.env_logging import environment_snapshot     # noqa: E402
from src.error_analysis import failure_cases, scores_by_tag  # noqa: E402
from src.metrics import bootstrap_ci, recall_at_k    # noqa: E402
from src.winoground_data import load_dataset         # noqa: E402
from src.winoground_eval import aggregate, per_example_scores  # noqa: E402

OUT = ROOT / "outputs" / "metrics"
CACHE = ROOT / "data" / "winoground_cache" / "embeddings"


def load_cfg():
    import os
    cfg = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    ckpts = yaml.safe_load((ROOT / "configs" / "checkpoints.yaml").read_text())["checkpoints"]
    # Overrides por entorno (útil en CI):
    if os.environ.get("PREFER_REAL", "").lower() in ("false", "0", "no"):
        cfg["prefer_real"] = False
    limit = os.environ.get("WINO_CKPT_LIMIT")
    if limit:
        ckpts = ckpts[: int(limit)]
    return cfg, ckpts


def embed_all(examples, model_name, pretrained, source, cache=True):
    """Codifica las 2N imágenes y 2N captions. Devuelve (img_feats, cap_feats) (2N,d)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{source}__{model_name}__{pretrained}".replace("/", "_")
    cache_path = CACHE / f"{key}.npz"
    if cache and cache_path.exists():
        z = np.load(cache_path)
        return z["img"], z["cap"]

    images, captions = [], []
    for ex in examples:
        images.extend([ex.image_0, ex.image_1])
        captions.extend([ex.caption_0, ex.caption_1])

    model, preprocess, tokenizer, device = oc.create_model(model_name, pretrained)
    n_params = oc.count_parameters(model)
    img = oc.encode_images(model, preprocess, images, device)
    cap = oc.encode_texts(model, tokenizer, captions, device)
    if cache:
        np.savez_compressed(cache_path, img=img, cap=cap, n_params=n_params)
    del model
    return img, cap


def per_example_sims(img_feats, cap_feats, n):
    """Construye las n matrices 2x2 sim[caption, imagen] por ejemplo."""
    sims = []
    for k in range(n):
        cf = cap_feats[2 * k:2 * k + 2]   # (2, d) captions del ejemplo k
        imf = img_feats[2 * k:2 * k + 2]  # (2, d) imágenes del ejemplo k
        sims.append(cf @ imf.T)           # (2,2): sim[c,i]
    return sims


def main():
    cfg, ckpts = load_cfg()
    seed = cfg["seed"]
    np.random.seed(seed)
    OUT.mkdir(parents=True, exist_ok=True)

    examples, source = load_dataset(prefer_real=cfg["prefer_real"])
    n = len(examples)
    tags = [ex.tag for ex in examples]
    print(f"[run] fuente={source}  N={n} ejemplos")

    # ---------- comparación de checkpoints ----------
    comparison_rows = []
    primary = None
    for ck in ckpts:
        t0 = time.time()
        img, cap = embed_all(examples, ck["name"], ck["pretrained"], source)
        sims = per_example_sims(img, cap, n)
        agg = aggregate(sims)
        elapsed = time.time() - t0
        key = f"{source}__{ck['name']}__{ck['pretrained']}".replace("/", "_")
        npz = np.load(CACHE / f"{key}.npz")
        comparison_rows.append({
            "label": ck["label"], "name": ck["name"], "pretrained": ck["pretrained"],
            "n_params_M": round(int(npz["n_params"]) / 1e6, 1),
            "text_score": round(agg.text, 4), "image_score": round(agg.image, 4),
            "group_score": round(agg.group, 4), "seconds": round(elapsed, 1),
        })
        print(f"  [{ck['label']}] text={agg.text:.3f} image={agg.image:.3f} "
              f"group={agg.group:.3f} ({elapsed:.1f}s)")
        if ck["name"] == cfg["primary_checkpoint"] and primary is None:
            primary = {"ck": ck, "img": img, "cap": cap, "sims": sims, "agg": agg}

    pd.DataFrame(comparison_rows).to_csv(OUT / "checkpoint_comparison.csv", index=False)

    if primary is None:  # fallback: usa el primero
        ck = ckpts[0]
        img, cap = embed_all(examples, ck["name"], ck["pretrained"], source)
        sims = per_example_sims(img, cap, n)
        primary = {"ck": ck, "img": img, "cap": cap, "sims": sims, "agg": aggregate(sims)}

    sims = primary["sims"]
    img, cap = primary["img"], primary["cap"]
    agg = primary["agg"]

    # ---------- per-example + bootstrap CI ----------
    rows = per_example_scores(sims)
    for r, ex in zip(rows, examples):
        r["id"] = ex.id
        r["tag"] = ex.tag
    pd.DataFrame(rows).to_csv(OUT / "per_example.csv", index=False)

    ci_rows = []
    for metric in ("text", "image", "group"):
        vals = [r[metric] for r in rows]
        ci = bootstrap_ci(vals, rounds=cfg["bootstrap_rounds"], seed=seed)
        chance = 0.25 if metric in ("text", "image") else 1.0 / 6.0
        ci_rows.append({"metric": metric, **{k: round(v, 4) for k, v in ci.items()},
                        "chance": round(chance, 4)})
    pd.DataFrame(ci_rows).to_csv(OUT / "bootstrap_ci.csv", index=False)

    # ---------- by tag ----------
    by_tag = scores_by_tag(sims, tags)
    by_tag.to_csv(OUT / "by_tag.csv", index=False)

    # ---------- blindness probe ----------
    cap_list = [cap[2 * k:2 * k + 2] for k in range(n)]
    img_list = [img[2 * k:2 * k + 2] for k in range(n)]
    blindness = run_blindness_probe(cap_list, img_list, seed=seed)
    (OUT / "blindness.json").write_text(json.dumps(blindness, indent=2, ensure_ascii=False))

    # ---------- Recall@K (alto) vs group score (≈azar): LA TESIS ----------
    # Recuperación con TODA la galería: 2N captions × 2N imágenes; el positivo de
    # la caption j es la imagen j (caption_0↔image_0, caption_1↔image_1).
    full_sim = cap @ img.T                      # (2N, 2N) text->image
    pos = list(range(2 * n))
    t2i = recall_at_k(full_sim, pos, ks=tuple(cfg["recall_ks"]))
    i2t = recall_at_k(full_sim.T, pos, ks=tuple(cfg["recall_ks"]))
    recall_vs_group = {
        "note": ("Recall@K usa toda la galería (fácil); el group score exige ganar "
                 "el contraste de pares mínimos (difícil). Alto R@K + bajo group = "
                 "el retrieval no implica razonamiento composicional."),
        "text_to_image_recall": t2i,
        "image_to_text_recall": i2t,
        "winoground_text_score": round(agg.text, 4),
        "winoground_image_score": round(agg.image, 4),
        "winoground_group_score": round(agg.group, 4),
        "chance_group": round(1.0 / 6.0, 4),
    }
    (OUT / "recall_vs_group.json").write_text(json.dumps(recall_vs_group, indent=2, ensure_ascii=False))

    # ---------- FAISS demo (extensión C10) ----------
    try:
        import faiss
        d = img.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(img.astype(np.float32))
        model, _, tokenizer, device = oc.create_model(primary["ck"]["name"], primary["ck"]["pretrained"])
        q = oc.encode_texts(model, tokenizer, [cfg["faiss_query"]], device)
        scores_f, idxs = index.search(q.astype(np.float32), min(5, 2 * n))
        faiss_rows = []
        for rank, (j, sc) in enumerate(zip(idxs[0], scores_f[0]), 1):
            ex = examples[j // 2]
            which = "img0" if j % 2 == 0 else "img1"
            cap_text = ex.caption_0 if j % 2 == 0 else ex.caption_1
            faiss_rows.append({"rank": rank, "score": round(float(sc), 4),
                               "example_id": ex.id, "image": which, "matching_caption": cap_text})
        pd.DataFrame(faiss_rows).to_csv(OUT / "faiss_demo.csv", index=False)
        del model
    except Exception as exc:  # noqa: BLE001
        (OUT / "faiss_demo.csv").write_text(f"faiss_error,{type(exc).__name__}\n")

    # ---------- failure cases ----------
    cases = failure_cases(sims, examples, max_cases=8)
    (OUT / "failure_cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False))

    # ---------- resumen ----------
    scores = {
        "source": source,
        "n_examples": n,
        "primary_checkpoint": primary["ck"]["label"],
        "scores": agg.as_dict(),
        "human_reference": {"text": 0.895, "image": 0.885, "group": 0.855},
        "environment": environment_snapshot(ROOT),
    }
    (OUT / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False))

    print(f"\n[run] LISTO. group_score={agg.group:.3f} (azar={1/6:.3f}) | "
          f"R@1 text->image={t2i['R@1']:.3f}")
    print(f"[run] salidas en {OUT}")


if __name__ == "__main__":
    main()

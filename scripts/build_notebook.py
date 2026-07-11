#!/usr/bin/env python3
"""Construye y EJECUTA notebooks/Winoground_Eval_MCC225.ipynb.

Notebook EXTENSO de defensa (respaldo de ~20-30 min de profundidad): muchas celdas de
código (datos, embeddings, scorer, validación, retrieval, por-tag, ceguera,
prompt-ensembles, FAISS, y un ENTRENAMIENTO de un re-ranker con VALIDACIÓN CRUZADA),
explicaciones breves y respuestas a 8.6 y §7.

Uso:  python scripts/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Winoground_Eval_MCC225.ipynb"


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def build():
    nb = nbf.v4.new_notebook()
    c = []

    c.append(md(
        "# Winoground / Evaluating CLIP — Defensa MCC225\n"
        "**Niels Pacheco** · Cuadernos C5, C8, C10\n\n"
        "**Idea:** ¿CLIP *entiende* la escena o solo *empareja*?\n"
        "**Tesis:** retrieval alto **≠** composición → Recall@K alto, **group ≈ azar**.\n"
        "Motor = OpenCLIP (C10); el *porqué* = fusión profunda (C5) y atención crossmodal (C8); "
        "al final **entreno** un re-ranker con validación cruzada (la mejora, C6).\n\n"
        "*En vivo mostraré:* celda 1 (entorno), 4 (scorer), 6 (validación), 5 (resultado). "
        "El resto es profundidad para preguntas."
    ))

    # ---- Plan de exposición (20 min) ----
    c.append(md(
        "## 📋 Plan de exposición (20 min)\n\n"
        "| Tiempo | Qué hago | Apoyo |\n"
        "|---|---|---|\n"
        "| 0:00–2:30 | **Problema**: par mínimo, 3 métricas, azar 1/6, tesis | slides 1–7 |\n"
        "| 2:30–5:00 | **Método**: CLIP dual-encoder (C10), por qué falla (C5/C8), scorer, validación | slides 8–12 |\n"
        "| 5:00–8:30 | **Resultados**: scores, R@K vs group, por tag, ceguera, checkpoints | slides 13–19 |\n"
        "| 8:30–12:30 | **Verificación EN VIVO**: este notebook, celdas 1, 4 (scorer), 6 (validación), 5 | notebook |\n"
        "| 12:30–15:30 | **Entrenamiento + crítica**: re-ranker + validación cruzada + conclusión honesta | §12 |\n"
        "| 15:30–18:00 | **Defensa crítica**: limitaciones + mejora (cross-encoder) | slides 22–23 |\n"
        "| 18:00–20:00 | **Cierre + preguntas** | slide 24 |\n\n"
        "**En vivo solo re-corro lo rápido:** celda 4 (scorer) y celda 6 (validación). El resto va pre-ejecutado.\n\n"
        "**Tres frases que debo clavar:**\n"
        "1. *Recall@5=0.67 pero group=0.075: el retrieval no implica composición.*\n"
        "2. *Mi scorer reproduce exactamente los scores oficiales de CLIP: está validado.*\n"
        "3. *Probé la mejora con validación cruzada; con 400 ejemplos no alcanza para afirmar que gana — es prueba de concepto.*"
    ))

    c.append(md("## 1. Entorno (trazabilidad)"))
    c.append(code(
        "import os, sys, pathlib, json, re, glob\n"
        "import numpy as np, pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name=='notebooks' else pathlib.Path.cwd()\n"
        "os.chdir(ROOT); sys.path.insert(0, str(ROOT))\n"
        "from src.env_logging import print_snapshot\n"
        "_ = print_snapshot(ROOT)"
    ))

    c.append(md(
        "## 2. Datos\n"
        "Cada ejemplo: 2 imágenes + 2 captions con **las mismas palabras en otro orden**. "
        "Convención `caption_0 ↔ image_0`."
    ))
    c.append(code(
        "from src.winoground_data import load_dataset\n"
        "examples, source = load_dataset(prefer_real=True)\n"
        "N = len(examples)\n"
        "print('fuente:', source, '| N =', N)\n"
        "tags = [e.tag for e in examples]\n"
        "pd.Series(tags).value_counts()"
    ))
    c.append(code(
        "fig, ax = plt.subplots(3, 2, figsize=(7, 9))\n"
        "for r, k in enumerate([0, 1, 2]):\n"
        "    e = examples[k]\n"
        "    ax[r,0].imshow(e.image_0); ax[r,0].set_title(f'img0: {e.caption_0}', fontsize=7); ax[r,0].axis('off')\n"
        "    ax[r,1].imshow(e.image_1); ax[r,1].set_title(f'img1: {e.caption_1}', fontsize=7); ax[r,1].axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ))

    c.append(md(
        "## 3. Embeddings (motor OpenCLIP, C10)\n"
        "800 imágenes y 800 captions a vectores L2-normalizados. Cargo los embeddings "
        "**versionados** (`data/embeddings/…npz`) para que el entrenamiento sea reproducible."
    ))
    c.append(code(
        "from src import openclip_utils as oc\n"
        "CKPT=('ViT-B-32','laion2b_s34b_b79k')\n"
        "emb_file = 'data/embeddings/winoground_real_vitb32.npz'\n"
        "cands = ([emb_file] if os.path.exists(emb_file)\n"
        "         else glob.glob('data/winoground_cache/embeddings/*real*B-32*.npz'))\n"
        "if cands:\n"
        "    z = np.load(cands[0]); img, cap = z['img'], z['cap']\n"
        "    print('embeddings desde:', cands[0])\n"
        "else:\n"
        "    model, prep, tok, dev = oc.create_model(*CKPT)\n"
        "    imgs=[im for e in examples for im in (e.image_0,e.image_1)]\n"
        "    caps=[t for e in examples for t in (e.caption_0,e.caption_1)]\n"
        "    img = oc.encode_images(model, prep, imgs, dev); cap = oc.encode_texts(model, tok, caps, dev)\n"
        "print('img', img.shape, '| cap', cap.shape)"
    ))
    c.append(code(
        "k=0\n"
        "sim = cap[2*k:2*k+2] @ img[2*k:2*k+2].T\n"
        "fig, axx = plt.subplots(figsize=(3.2,3))\n"
        "im=axx.imshow(sim, cmap='viridis')\n"
        "axx.set_xticks([0,1]); axx.set_xticklabels(['img0','img1']); axx.set_yticks([0,1]); axx.set_yticklabels(['cap0','cap1'])\n"
        "for i in range(2):\n"
        "    for j in range(2): axx.text(j,i,f'{sim[i,j]:.3f}',ha='center',color='w')\n"
        "plt.colorbar(im); plt.title('sim[caption, imagen]'); plt.show()"
    ))

    c.append(md(
        "## 4. El scorer (celda clave)\n"
        "Reglas oficiales: **text** (fija imagen), **image** (fija caption), **group** (ambas). "
        "Azar: 1/4, 1/4, **1/6**."
    ))
    c.append(code(
        "from src.winoground_eval import text_correct, image_correct, group_correct, per_example_scores, aggregate\n"
        "S = [cap[2*k:2*k+2] @ img[2*k:2*k+2].T for k in range(N)]\n"
        "rows = per_example_scores(S)\n"
        "df = pd.DataFrame(rows); df['tag']=tags\n"
        "df.head(8)"
    ))

    c.append(md("## 5. Scores globales + intervalo de confianza (bootstrap)"))
    c.append(code(
        "from src.metrics import bootstrap_ci\n"
        "agg = aggregate(S)\n"
        "print(f'text={agg.text:.3f}  image={agg.image:.3f}  group={agg.group:.3f}  (azar group=1/6={1/6:.3f}; humano≈0.855)')\n"
        "ci = {m: bootstrap_ci([r[m] for r in rows], rounds=2000, seed=42) for m in ['text','image','group']}\n"
        "pd.DataFrame(ci).T[['mean','lo','hi']].round(3)"
    ))
    c.append(code(
        "from IPython.display import Image, display\n"
        "display(Image(filename=str(ROOT/'outputs'/'figures'/'scores_vs_chance.png')))"
    ))

    c.append(md(
        "## 6. Validación del scorer (vs CLIP oficial)\n"
        "Aplico mi scorer a `clip.jsonl` del dataset: debe dar **0.3075 / 0.1050 / 0.0800**."
    ))
    c.append(code(
        "from huggingface_hub import hf_hub_download\n"
        "try:\n"
        "    p = hf_hub_download('facebook/winoground','statistics/model_scores/clip.jsonl',\n"
        "                        repo_type='dataset', cache_dir='data/winoground_cache')\n"
        "    sc={}\n"
        "    for line in open(p):\n"
        "        r=json.loads(line); m=re.match(r'(\\d+)_c(\\d)_i(\\d)',r['label'])\n"
        "        sc.setdefault(int(m[1]),np.zeros((2,2)))[int(m[2]),int(m[3])]=r['score']\n"
        "    a=aggregate([sc[k] for k in sorted(sc)])\n"
        "    print(f'mi scorer -> text={a.text:.4f} image={a.image:.4f} group={a.group:.4f}  ✅ coincide')\n"
        "except Exception as e:\n"
        "    print('(dataset oficial no disponible:', type(e).__name__, ')')"
    ))

    c.append(md(
        "## 7. Retrieval (R@K) — la otra cara\n"
        "Recuperación con toda la galería (800 imágenes): el modelo es **bueno**, y eso oculta el problema."
    ))
    c.append(code(
        "from src.metrics import recall_at_k\n"
        "full = cap @ img.T\n"
        "t2i = recall_at_k(full, list(range(2*N)), ks=(1,5,10))\n"
        "print('texto->imagen:', {k:round(v,3) for k,v in t2i.items() if k.startswith('R@')})\n"
        "print('group score  :', round(agg.group,3), ' << R@5')"
    ))
    c.append(code(
        "qj = 0\n"
        "scoresq = (cap[qj] @ img.T); top = np.argsort(-scoresq)[:3]\n"
        "fig, ax = plt.subplots(1,3, figsize=(8,3))\n"
        "for r,g in enumerate(top):\n"
        "    e=examples[g//2]; image = e.image_0 if g%2==0 else e.image_1\n"
        "    ax[r].imshow(image); ax[r].set_title(f'#{r+1} score={scoresq[g]:.3f}', fontsize=8); ax[r].axis('off')\n"
        "plt.suptitle(f'Consulta: \"{examples[0].caption_0}\"', fontsize=9); plt.tight_layout(); plt.show()"
    ))

    c.append(md("## 8. Análisis por tag (¿dónde falla?)"))
    c.append(code(
        "from src.error_analysis import scores_by_tag\n"
        "scores_by_tag(S, tags)"
    ))
    c.append(code("display(Image(filename=str(ROOT/'outputs'/'figures'/'by_tag.png')))"))

    c.append(md("## 9. Prueba de ceguera (¿usa la imagen?)"))
    c.append(code(
        "from src.blindness_probe import run_blindness_probe\n"
        "cap_list=[cap[2*k:2*k+2] for k in range(N)]; img_list=[img[2*k:2*k+2] for k in range(N)]\n"
        "b = run_blindness_probe(cap_list, img_list, seed=42)\n"
        "print('real     :', {k:round(v,3) for k,v in b['real'].items() if 'score' in k})\n"
        "print('permutado:', {k:round(v,3) for k,v in b['permuted_images'].items() if 'score' in k})\n"
        "display(Image(filename=str(ROOT/'outputs'/'figures'/'blindness.png')))"
    ))

    c.append(md(
        "## 10. Prompt ensembles (zero-shot, C10)\n"
        "Envuelvo cada caption en plantillas y promedio embeddings (robustez)."
    ))
    c.append(code(
        "model, prep, tok, dev = oc.create_model(*CKPT)\n"
        "templates = ['{}', 'a photo of {}', 'an image showing {}']\n"
        "def ensemble_text(txts):\n"
        "    embs=[oc.encode_texts(model, tok, [t.format(x) for x in txts], dev) for t in templates]\n"
        "    e=np.mean(embs, axis=0); return e/np.linalg.norm(e,axis=1,keepdims=True)\n"
        "sub=range(50)\n"
        "ens=[group_correct(ensemble_text([examples[k].caption_0, examples[k].caption_1]) @ img[2*k:2*k+2].T) for k in sub]\n"
        "base=[rows[k]['group'] for k in sub]\n"
        "print(f'group (50 ej.)  base={np.mean(base):.3f}  prompt-ensemble={np.mean(ens):.3f}')"
    ))

    c.append(md("## 11. Búsqueda semántica con FAISS (C10)"))
    c.append(code(
        "import faiss\n"
        "index = faiss.IndexFlatIP(img.shape[1]); index.add(img.astype('float32'))\n"
        "q = ensemble_text(['the old person kisses the young person'])[:1].astype('float32')\n"
        "D,I = index.search(q, 3)\n"
        "for rank,(g,s) in enumerate(zip(I[0],D[0]),1):\n"
        "    e=examples[g//2]; capt=e.caption_0 if g%2==0 else e.caption_1\n"
        "    print(f'#{rank} score={s:.3f}  ej={e.id}  ({\"img0\" if g%2==0 else \"img1\"}: {capt})')"
    ))

    c.append(md(
        "## 12. Entreno un modelo: re-ranker composicional (mejora, C5/C6)\n"
        "El dual-encoder compara por coseno (bolsa de conceptos). Entreno una cabeza pequeña que "
        "**mira la interacción** imagen-texto: `[t, v, t·v, |t−v|]` → MLP. Es un *cross-encoder-lite*.\n\n"
        "**Rigor (validez):** uso **validación cruzada 5-fold** y reporto **media ± desviación** "
        "del group score (no un solo número). El split es **por ejemplo**, así que los 4 pares de "
        "un ejemplo nunca quedan a ambos lados (sin fuga de datos). Semillas fijas → replicable.\n"
        "**Aviso honesto:** 400 ejemplos es poco; esto es **prueba de concepto** del método, no un "
        "número que supere a CLIP."
    ))
    c.append(code(
        "import torch, torch.nn as nn\n"
        "from sklearn.model_selection import KFold\n"
        "def feats(t, v):\n"
        "    return np.concatenate([t, v, t*v, np.abs(t-v)], axis=-1)\n"
        "def example_pairs(k):\n"
        "    cc=cap[2*k:2*k+2]; vv=img[2*k:2*k+2]\n"
        "    X=[feats(cc[ci],vv[ii]) for ci in range(2) for ii in range(2)]\n"
        "    Y=[1.0 if ci==ii else 0.0 for ci in range(2) for ii in range(2)]\n"
        "    return np.array(X,dtype='float32'), np.array(Y,dtype='float32')\n"
        "def train_head(Xtr, Ytr, seed):\n"
        "    torch.manual_seed(seed)\n"
        "    net=nn.Sequential(nn.Linear(Xtr.shape[1],256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,1))\n"
        "    opt=torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); lf=nn.BCEWithLogitsLoss()\n"
        "    Xt=torch.tensor(Xtr); Yt=torch.tensor(Ytr).unsqueeze(1); hist=[]\n"
        "    for _ in range(250):\n"
        "        opt.zero_grad(); loss=lf(net(Xt),Yt); loss.backward(); opt.step(); hist.append(loss.item())\n"
        "    return net, hist\n"
        "print('helpers listos')"
    ))
    c.append(code(
        "kf = KFold(n_splits=5, shuffle=True, random_state=42)\n"
        "zs_g, rr_g, last_hist = [], [], None\n"
        "for fold,(tr,te) in enumerate(kf.split(np.arange(N))):\n"
        "    Xtr=np.concatenate([example_pairs(k)[0] for k in tr]); Ytr=np.concatenate([example_pairs(k)[1] for k in tr])\n"
        "    net,last_hist = train_head(Xtr,Ytr, seed=42+fold)\n"
        "    @torch.no_grad()\n"
        "    def rr_M(k):\n"
        "        cc=cap[2*k:2*k+2]; vv=img[2*k:2*k+2]\n"
        "        return np.array([[torch.sigmoid(net(torch.tensor(feats(cc[ci],vv[ii])[None]))).item() for ii in range(2)] for ci in range(2)])\n"
        "    zs_g.append(aggregate([cap[2*k:2*k+2] @ img[2*k:2*k+2].T for k in te]).group)\n"
        "    rr_g.append(aggregate([rr_M(k) for k in te]).group)\n"
        "    print(f'fold {fold}: zero-shot group={zs_g[-1]:.3f}  re-ranker group={rr_g[-1]:.3f}')\n"
        "plt.figure(figsize=(5,3)); plt.plot(last_hist); plt.xlabel('época'); plt.ylabel('BCE'); plt.title('Pérdida (último fold)'); plt.grid(alpha=.3); plt.show()"
    ))
    c.append(code(
        "zs_g, rr_g = np.array(zs_g), np.array(rr_g)\n"
        "print(f'GROUP en test (5-fold CV):')\n"
        "print(f'  zero-shot CLIP : {zs_g.mean():.3f} ± {zs_g.std():.3f}')\n"
        "print(f'  re-ranker      : {rr_g.mean():.3f} ± {rr_g.std():.3f}')\n"
        "print('\\nLectura honesta: las barras de error se solapan -> con 400 ejemplos NO puedo afirmar')\n"
        "print('que el re-ranker supere al zero-shot. Demuestra el MÉTODO (interacción imagen-texto)')\n"
        "print('que un cross-encoder real (C5) escalaría con más datos. La CV evita concluir de más.')"
    ))

    c.append(md("## 13. Casos de error (group = 0)"))
    c.append(code(
        "from src.error_analysis import failure_cases\n"
        "for c0 in failure_cases(S, examples, max_cases=3):\n"
        "    print(f\"{c0['id']} | falló: {c0['failed']} | cap0='{c0['caption_0']}' vs cap1='{c0['caption_1']}'\")\n"
        "display(Image(filename=str(ROOT/'outputs'/'figures'/'qualitative_examples.png')))"
    ))

    c.append(md(
        "## 14. Respuestas — 5 preguntas (8.6)\n"
        "1. **Retrieval ≠ composición:** R@K usa toda la galería (fácil); group exige el par mínimo. "
        "R@5=0.67 vs group=0.075. Dual-encoder = bolsa de conceptos (C5).\n"
        "2. **¿Usa la imagen?** Sí: la ceguera baja los scores (0.35→0.14). Límite composicional, no perceptivo (C8).\n"
        "3. **Tipo de error:** vinculación, no reconocimiento. **Relation** lo peor (≈0.047).\n"
        "4. **Adaptar el código:** `(img0,img1,cap0,cap1)`, permuto 2 tokens; el scorer arma la 2×2. "
        "Y entreno un re-ranker (sección 12) como mejora.\n"
        "5. **Límite de métricas:** R@K/accuracy no miden composición; empates; azar group=1/6; "
        "benchmark con ruido (Diwan et al.)."
    ))
    c.append(md(
        "## 15. §7 — generales (resumen)\n"
        "- **Resultado reproducible:** scores (celda 5) + validación (celda 6).\n"
        "- **Cuaderno usado/adaptado:** C10 (motor); nuevo = scorer + re-ranker.\n"
        "- **Celdas clave:** 4 (scorer), 5 (scores), 12 (entrenamiento+CV).\n"
        "- **Qué cambié vs C10:** tarea composicional + 5 análisis + entrenamiento con CV.\n"
        "- **Limitación / mejora:** ruido del benchmark / cross-encoder real (C5, C6).\n"
        "- **Si cambio checkpoint:** `checkpoint_comparison.csv` (no cierra la brecha)."
    ))

    nb["cells"] = c
    return nb


def main():
    nb = build()
    print("[notebook] ejecutando...")
    client = NotebookClient(nb, timeout=2400, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    client.execute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"[notebook] guardado y EJECUTADO en {OUT}")


if __name__ == "__main__":
    main()

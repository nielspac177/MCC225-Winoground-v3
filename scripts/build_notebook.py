#!/usr/bin/env python3
"""Construye y EJECUTA notebooks/Winoground_Eval_MCC225.ipynb.

El notebook deriva del flujo del Cuaderno 10: imprime la hoja de trazabilidad,
carga datos, demuestra el scorer en un par mínimo, y muestra las métricas y figuras
ya generadas por el pipeline (evidencia reproducible para la defensa).

Uso:  python scripts/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Winoground_Eval_MCC225.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(md(
        "# Winoground / Evaluating CLIP — Defensa MCC225\n"
        "**Niels Victor Pacheco Barrios** · Cuadernos **C5, C8, C10**.\n\n"
        "**Tesis:** retrieval alto $\\neq$ razonamiento composicional. Reutilizo el motor "
        "OpenCLIP del **Cuaderno 10** y evalúo Winoground con los tres scores oficiales."
    ))

    cells.append(md("## 1. Hoja de trazabilidad (entorno de ejecución)"))
    cells.append(code(
        "import os, sys, pathlib\n"
        "ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name=='notebooks' else pathlib.Path.cwd()\n"
        "os.chdir(ROOT)            # rutas relativas (data/, outputs/) respecto a la raíz del repo\n"
        "sys.path.insert(0, str(ROOT))\n"
        "from src.env_logging import print_snapshot\n"
        "_ = print_snapshot(ROOT)"
    ))

    cells.append(md(
        "## 2. Datos: Winoground real (gated) o set curado offline\n"
        "`load_dataset` intenta el benchmark oficial y, si no hay acceso, cae al set curado "
        "de pares mínimos composicionales (mismo formato, mismo scorer)."
    ))
    cells.append(code(
        "from src.winoground_data import load_dataset\n"
        "examples, source = load_dataset(prefer_real=True)\n"
        "print('fuente:', source, '| N =', len(examples))\n"
        "ex = examples[0]\n"
        "print('caption_0:', ex.caption_0)\n"
        "print('caption_1:', ex.caption_1)"
    ))
    cells.append(code(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(1,2, figsize=(6,3))\n"
        "ax[0].imshow(ex.image_0); ax[0].set_title('image_0', fontsize=9); ax[0].axis('off')\n"
        "ax[1].imshow(ex.image_1); ax[1].set_title('image_1', fontsize=9); ax[1].axis('off')\n"
        "plt.suptitle('Par mínimo: mismas palabras, distinto orden', fontsize=10); plt.tight_layout(); plt.show()"
    ))

    cells.append(md(
        "## 3. Celda clave: el scorer de Winoground sobre un par\n"
        "Codifico las 2 imágenes y 2 captions con OpenCLIP (C10), construyo la matriz "
        "$2\\times2$ `sim[caption, imagen]` y aplico las definiciones del paper."
    ))
    cells.append(code(
        "from src import openclip_utils as oc\n"
        "from src.winoground_eval import text_correct, image_correct, group_correct\n"
        "model, preprocess, tokenizer, device = oc.create_model('ViT-B-32','laion2b_s34b_b79k')\n"
        "imgf = oc.encode_images(model, preprocess, [ex.image_0, ex.image_1], device)\n"
        "capf = oc.encode_texts(model, tokenizer, [ex.caption_0, ex.caption_1], device)\n"
        "sim = capf @ imgf.T   # sim[caption, imagen]\n"
        "print('matriz 2x2 sim[caption,imagen]:\\n', sim.round(3))\n"
        "print('text=', text_correct(sim), ' image=', image_correct(sim), ' group=', group_correct(sim))"
    ))

    cells.append(md(
        "## 4. Resultado principal (todas las métricas)\n"
        "Generadas por `scripts/02_run_winoground.py`. Si no existen, ejecútalo primero."
    ))
    cells.append(code(
        "import json, pandas as pd\n"
        "MET = ROOT/'outputs'/'metrics'\n"
        "scores = json.loads((MET/'scores.json').read_text())\n"
        "print('fuente:', scores['source'], '| checkpoint:', scores['primary_checkpoint'])\n"
        "print(json.dumps(scores['scores'], indent=2))\n"
        "pd.read_csv(MET/'bootstrap_ci.csv')"
    ))
    cells.append(code(
        "pd.read_csv(MET/'checkpoint_comparison.csv')"
    ))
    cells.append(code(
        "rg = json.loads((MET/'recall_vs_group.json').read_text())\n"
        "print('Recall texto->imagen:', rg['text_to_image_recall'])\n"
        "print('group score:', rg['winoground_group_score'], ' (azar', rg['chance_group'], ')')\n"
        "print('\\n=>', rg['note'])"
    ))

    cells.append(md("## 5. Análisis por tag y prueba de ceguera"))
    cells.append(code(
        "display(pd.read_csv(MET/'by_tag.csv'))\n"
        "print(json.dumps(json.loads((MET/'blindness.json').read_text()), indent=2, ensure_ascii=False))"
    ))

    cells.append(md("## 6. Figuras"))
    cells.append(code(
        "from IPython.display import Image, display\n"
        "FIG = ROOT/'outputs'/'figures'\n"
        "for name in ['scores_vs_chance.png','recall_vs_group.png','blindness.png','by_tag.png','checkpoint_comparison.png']:\n"
        "    p = FIG/name\n"
        "    if p.exists(): display(Image(filename=str(p)))"
    ))

    cells.append(md(
        "## 7. Relación con los cuadernos y limitaciones\n"
        "- **C10:** motor OpenCLIP (embeddings, coseno, checkpoints, FAISS).\n"
        "- **C5:** dual-encoder vs fusión profunda — explica *por qué* falla la composición.\n"
        "- **C8:** atención crossmodal / interpretabilidad — versión interpretable de '¿usa la imagen?'.\n\n"
        "**Limitación:** el group score mezcla fallo composicional con ítems ambiguos; el set curado es un "
        "proxy. **Mejora:** evaluar un cross-encoder real (C5) y re-ranking con interacción cruzada (C6)."
    ))

    nb["cells"] = cells
    return nb


def main():
    nb = build()
    print("[notebook] ejecutando celdas...")
    client = NotebookClient(nb, timeout=1200, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    client.execute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"[notebook] guardado y EJECUTADO en {OUT}")


if __name__ == "__main__":
    main()

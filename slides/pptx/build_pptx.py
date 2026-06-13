#!/usr/bin/env python3
"""Genera el deck PPTX de la defensa (espejo del Beamer), usando outputs/figures.

Uso:  python slides/pptx/build_pptx.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "slides" / "pptx" / "defensa_winoground.pptx"

AZUL = RGBColor(0x00, 0x72, 0xB2)
GRIS = RGBColor(0x33, 0x33, 0x33)
W, H = Inches(13.333), Inches(7.5)


def _title_only(prs, title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.0))
    tf = box.text_frame
    tf.text = title
    p = tf.paragraphs[0]
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = AZUL
    return slide


def _bullets(slide, items, left=Inches(0.6), top=Inches(1.4), width=Inches(7.0), size=18):
    box = slide.shapes.add_textbox(left, top, width, Inches(5.5))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "•  " + it
        p.font.size = Pt(size)
        p.font.color.rgb = GRIS
        p.space_after = Pt(8)


def _img(slide, name, left, top, height=None, width=None):
    path = FIG / name
    if not path.exists():
        return
    kw = {}
    if height:
        kw["height"] = height
    if width:
        kw["width"] = width
    slide.shapes.add_picture(str(path), left, top, **kw)


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # Portada
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(0.8), Inches(2.4), Inches(11.7), Inches(2.5)).text_frame
    t.word_wrap = True
    t.text = "Winoground / Evaluating CLIP"
    t.paragraphs[0].font.size = Pt(44)
    t.paragraphs[0].font.bold = True
    t.paragraphs[0].font.color.rgb = AZUL
    for line in ["Retrieval alto ≠ razonamiento composicional",
                 "Niels Victor Pacheco Barrios — MCC225 · Examen Parcial 2026-1",
                 "Cuadernos C5 · C8 · C10"]:
        p = t.add_paragraph(); p.text = line; p.font.size = Pt(20); p.font.color.rgb = GRIS

    # 2 min
    s = _title_only(prs, "Recordatorio: paper y resultado (2 min)")
    _bullets(s, [
        "Winoground: pares mínimos (2 imágenes, 2 captions, mismas palabras, distinto orden).",
        "Mide composición, no reconocimiento. Azar: text=image=1/4, group=1/6.",
        "Evaluating CLIP: mayor accuracy no implica mejor modelo (sesgos, diseño de clases).",
        "Resultado: Recall@K alto, pero group score ≈ azar.",
    ], width=Inches(6.0))
    _img(s, "recall_vs_group.png", Inches(7.0), Inches(1.5), height=Inches(4.6))

    # 3 min — cómo
    s = _title_only(prs, "Cómo lo obtuve ejecutando C10 (3 min)")
    _bullets(s, [
        "Motor OpenCLIP del Cuaderno 10: encode imagen/texto, similitud coseno.",
        "winoground_data.py: real (HF gated) + fallback curado.",
        "winoground_eval.py: matriz 2×2 sim[caption,imagen] y los 3 scores.",
        "02_run_winoground.py: scores + IC bootstrap + by-tag + ceguera + R@K.",
        "text=1 ⇔ s(c0,i0)>s(c0,i1) y s(c1,i1)>s(c1,i0).",
        "group = text AND image.",
    ], width=Inches(11.8))

    # 3 min — verificación
    s = _title_only(prs, "Verificación en vivo: los tres scores (3 min)")
    _img(s, "scores_vs_chance.png", Inches(2.8), Inches(1.4), height=Inches(5.6))

    # 3 min — mejora
    s = _title_only(prs, "Mejora en vivo: ¿usa la imagen? y por tag (3 min)")
    _img(s, "blindness.png", Inches(0.5), Inches(1.5), height=Inches(4.6))
    _img(s, "by_tag.png", Inches(6.9), Inches(1.5), height=Inches(4.6))

    # 2 min — crítica
    s = _title_only(prs, "Defensa crítica: limitaciones y mejora (2 min)")
    _bullets(s, [
        "El group score mezcla fallo composicional con ítems ambiguos (Diwan et al.).",
        "Set curado = proxy controlado, no el benchmark oficial.",
        "Azar del group = 1/6 (no 1/16): text e image no son independientes.",
        "Sensibilidad a empates y a la normalización de embeddings.",
        "Mejora: evaluar cross-encoder / fusión profunda (C5) y re-ranking (C6); versionar embeddings (DVC).",
    ], width=Inches(11.8))

    # Backups
    s = _title_only(prs, "Backup: comparación de checkpoints")
    _img(s, "checkpoint_comparison.png", Inches(2.5), Inches(1.4), height=Inches(5.6))
    s = _title_only(prs, "Backup: casos de error")
    _img(s, "qualitative_examples.png", Inches(3.8), Inches(1.3), height=Inches(5.8))

    prs.save(OUT)
    print(f"[pptx] guardado en {OUT}")


if __name__ == "__main__":
    main()

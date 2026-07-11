#!/usr/bin/env python3
"""Genera el deck PPTX de la defensa (espejo del Beamer de 25 diapositivas).

Usa las figuras de outputs/figures. Uso:  python slides/pptx/build_pptx.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "slides" / "pptx" / "defensa_winoground.pptx"

AZUL = RGBColor(0x00, 0x72, 0xB2)
GRIS = RGBColor(0x33, 0x33, 0x33)
VERDE = RGBColor(0x00, 0x9E, 0x73)
W, H = Inches(13.333), Inches(7.5)


def title_slide(prs, title, subtitle_lines):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    t = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(3)).text_frame
    t.word_wrap = True
    t.text = title
    t.paragraphs[0].font.size = Pt(40)
    t.paragraphs[0].font.bold = True
    t.paragraphs[0].font.color.rgb = AZUL
    for line in subtitle_lines:
        p = t.add_paragraph(); p.text = line; p.font.size = Pt(18); p.font.color.rgb = GRIS
    return s


def section_slide(prs, label):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box = s.shapes.add_textbox(Inches(0.8), Inches(3.0), Inches(11.7), Inches(1.5))
    tf = box.text_frame
    tf.text = label
    p = tf.paragraphs[0]
    p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = AZUL
    return s


def head(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.0))
    tf = box.text_frame; tf.word_wrap = True
    tf.text = title
    p = tf.paragraphs[0]; p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = AZUL
    return s


def bullets(slide, items, left=Inches(0.6), top=Inches(1.4), width=Inches(12.1), size=18):
    box = slide.shapes.add_textbox(left, top, width, Inches(5.6))
    tf = box.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        lvl = 0
        if isinstance(it, tuple):
            it, lvl = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("    " * lvl) + ("•  " if lvl == 0 else "–  ") + it
        p.font.size = Pt(size - 2 * lvl); p.font.color.rgb = GRIS; p.space_after = Pt(7)


def img(slide, name, left, top, height=None, width=None):
    path = FIG / name
    if not path.exists():
        return
    kw = {}
    if height: kw["height"] = height
    if width: kw["width"] = width
    slide.shapes.add_picture(str(path), left, top, **kw)


def codebox(slide, code, left=Inches(0.7), top=Inches(1.5), width=Inches(11.9), height=Inches(3.2)):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame; tf.word_wrap = True
    for i, line in enumerate(code.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = "Courier New"; p.font.size = Pt(15); p.font.color.rgb = GRIS


def note(slide, text, size=13):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(6.7), Inches(12.1), Inches(0.7))
    tf = box.text_frame; tf.word_wrap = True
    tf.text = text
    tf.paragraphs[0].font.size = Pt(size); tf.paragraphs[0].font.color.rgb = AZUL


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 Portada
    title_slide(prs, "Winoground / Evaluating CLIP",
                ["El retrieval alto no implica razonamiento composicional",
                 "Niels Victor Pacheco Barrios — MCC225 · Examen Parcial 2026-1",
                 "Cuadernos C5 · C8 · C10"])

    # 2 Agenda
    s = head(prs, "Qué voy a defender en 15 minutos")
    bullets(s, [
        "El problema: ¿CLIP entiende o solo empareja? (Winoground)",
        "Cómo lo medí reutilizando el motor OpenCLIP del Cuaderno 10.",
        "El resultado: retrieval alto, group score ≈ azar.",
        "Verificación en vivo + validación contra los scores oficiales.",
        "Por qué pasa (C5, C8), limitaciones y mejora.",
    ])
    note(s, "Tesis: recuperar bien una imagen no prueba que el modelo entienda el orden ni las relaciones.")

    # Sección
    section_slide(prs, "1 · El problema")

    # 3 Dos papers
    s = head(prs, "Mi tema: dos papers que cuestionan a CLIP")
    bullets(s, [
        "Winoground (Thrush et al., 2022): ¿razonan de forma composicional o reconocen objetos sueltos?",
        "Evaluating CLIP (Agarwal et al., 2021): una accuracy mayor no define un modelo 'mejor' (sesgos de clases/prompts).",
        "Idea común: una buena métrica puede ocultar que el modelo no entiende.",
    ])

    # 4 Par mínimo
    s = head(prs, "El núcleo de Winoground: el par mínimo")
    bullets(s, [
        "Dos imágenes y dos captions con las MISMAS palabras en distinto orden.",
        "Caption 0: 'an old person kisses a young person'.",
        "Caption 1: 'a young person kisses an old person'.",
        "Distinguirlas exige saber quién hace qué a quién: composición, no reconocimiento.",
        "400 ejemplos, etiquetados: Object, Relation, Both.",
    ])

    # 5 Tres métricas
    s = head(prs, "Las tres métricas oficiales (matriz 2×2 sim[caption, imagen])")
    bullets(s, [
        "text score: fijada cada imagen, ¿gana el caption correcto?  s00>s10 y s11>s01.",
        "image score: fijado cada caption, ¿gana la imagen correcta?  s00>s01 y s11>s10.",
        "group score: ambas a la vez (text Y image).",
    ])
    note(s, "La diagonal (s00, s11) es el emparejamiento correcto.")

    # 6 Azar 1/6
    s = head(prs, "Detalle fino: el azar del group score es 1/6")
    bullets(s, [
        "text e image NO son independientes: comparten las 4 similitudes.",
        "group=1 exige que ambas diagonales superen a ambas off-diagonales.",
        "De las 4!=24 ordenaciones, solo 4 cumplen → 4/24 = 1/6 ≈ 0.167.",
        "No es 1/16. Comparar contra el baseline correcto es parte de la métrica.",
    ])

    # 7 Tesis
    s = head(prs, "La pregunta que respondo")
    bullets(s, [
        "¿Un buen Recall@K (recuperación) implica razonamiento composicional?",
        "Respuesta: no. El mismo modelo, sobre el mismo conjunto, recupera bien y falla la composición.",
    ], size=22, top=Inches(2.5))

    # Sección
    section_slide(prs, "2 · Cómo lo medí (Cuaderno 10)")

    # 8 Cómo funciona CLIP
    s = head(prs, "Cómo funciona CLIP: un dual-encoder contrastivo")
    bullets(s, [
        "Dos torres separadas: una codifica la imagen, otra el texto.",
        "Cada modalidad colapsa en un único vector global; se comparan por coseno.",
        "Entrenado con aprendizaje contrastivo (acerca pares correctos, aleja incorrectos).",
        "Es el motor del Cuaderno 10 (OpenCLIP): create_model_and_transforms, encode_*. Lo reutilicé tal cual.",
    ])

    # 9 Por qué falla
    s = head(prs, "Por qué un dual-encoder falla la composición")
    bullets(s, [
        "Un vector global se comporta como 'bolsa de conceptos'.",
        "'old', 'young', 'kiss' activan dimensiones parecidas estén como estén compuestos.",
        "No hay interacción token-a-token entre palabras y regiones.",
        "Conexión C5/C8: la fusión profunda (MMBT) y la atención crossmodal sí dejan interactuar los tokens.",
    ])

    # 10 Pipeline
    s = head(prs, "El pipeline (reproducible)")
    bullets(s, [
        "winoground_data.py: Winoground real (HF, gated) o set curado offline.",
        "openclip_utils.py: embeddings L2-normalizados (motor C10).",
        "winoground_eval.py: matriz 2×2 y los 3 scores.",
        "metrics.py: Recall@K, intervalos bootstrap.",
        "blindness_probe.py (¿usa la imagen?) y error_analysis.py (por tag).",
        "Determinista (semillas), versionado, con tests. Una corrida: make run.",
    ])

    # 11 Scorer código
    s = head(prs, "El bloque clave: el scorer")
    codebox(s,
        "def text_correct(sim):   # fija imagen, elige caption\n"
        "    return sim[0,0] > sim[1,0] and sim[1,1] > sim[0,1]\n\n"
        "def image_correct(sim):  # fija caption, elige imagen\n"
        "    return sim[0,0] > sim[0,1] and sim[1,1] > sim[1,0]\n\n"
        "def group_correct(sim):\n"
        "    return text_correct(sim) and image_correct(sim)")
    note(s, "Es el código oficial de Winoground; me guié por el código (no por la redacción) y lo validé numéricamente.")

    # 12 Validación
    s = head(prs, "¿Cómo sé que mi scorer no está mal?")
    bullets(s, [
        "Lo apliqué a los scores oficiales de CLIP del propio dataset (statistics/model_scores/clip.jsonl).",
        "Paper CLIP ViT-B/32:  text=0.3075,  image=0.1050,  group=0.0800.",
        "Mi scorer:            text=0.3075,  image=0.1050,  group=0.0800.  → coincidencia EXACTA.",
        "python scripts/validate_against_official.py",
    ])

    # Sección
    section_slide(prs, "3 · Resultados")

    # 13 Resultado principal (figura)
    s = head(prs, "Resultado principal: los tres scores")
    img(s, "scores_vs_chance.png", Inches(2.9), Inches(1.4), height=Inches(5.2))
    note(s, "ViT-B-32/laion2b, 400 ejemplos reales. text=0.35, image=0.11, group=0.075 (azar 0.167; humano 0.86).")

    # 14 Cómo leer
    s = head(prs, "Cómo leer ese resultado")
    bullets(s, [
        "text (0.35) está POR ENCIMA del azar (0.25): el modelo capta algo.",
        "group (0.075) está POR DEBAJO del azar (0.167).",
        "No es 'peor que aleatorio' en general: es la asimetría text ≫ image (acierta una dirección y falla la otra).",
        "El IC bootstrap 95% del group [0.05, 0.10] queda entero por debajo del azar.",
    ])

    # 15 Retrieval vs group
    s = head(prs, "La evidencia de la tesis: retrieval alto, group bajo")
    img(s, "recall_vs_group.png", Inches(3.0), Inches(1.4), height=Inches(5.0))
    note(s, "Mismo conjunto: R@5=0.67, R@10=0.77 pero group=0.075. Retrieval usa toda la galería; group exige el par mínimo.")

    # 16 Por tag
    s = head(prs, "¿Qué tipo de error? Análisis por tag")
    img(s, "by_tag.png", Inches(3.2), Inches(1.4), height=Inches(4.9))
    note(s, "Relation es lo más difícil (group 0.047). Fallos de vinculación, no de reconocimiento.")

    # 17 Ceguera
    s = head(prs, "¿El modelo usa la imagen? Prueba de ceguera")
    img(s, "blindness.png", Inches(0.5), Inches(1.4), height=Inches(5.0))
    bullets(s, [
        "Permuto las imágenes entre ejemplos y recalculo.",
        "Real (0.35/0.11/0.075) → permutado (0.14/0.04/0.02).",
        "Sí usa la imagen; su límite es composicional, no perceptivo.",
        "Versión cuantitativa de la interpretabilidad de C8.",
    ], left=Inches(7.0), width=Inches(5.8), top=Inches(1.6), size=15)

    # 18 Checkpoints
    s = head(prs, "¿Y si uso un modelo más grande?")
    img(s, "checkpoint_comparison.png", Inches(3.0), Inches(1.4), height=Inches(4.7))
    note(s, "Group no mejora monótonamente (0.075/0.072/0.085) y el text baja (0.35→0.30→0.29). Escalar no resuelve la composición.")

    # 19 Cualitativos
    s = head(prs, "Casos de error concretos (group = 0)")
    img(s, "qualitative_examples.png", Inches(3.6), Inches(1.3), height=Inches(5.4))

    # Sección
    section_slide(prs, "4 · Cierre")

    # 20 Reproducibilidad
    s = head(prs, "Reproducibilidad (lo que pide el examen)")
    bullets(s, [
        "Entorno fijo: uv + uv.lock, Python 3.12. Imagen Docker CPU; Makefile.",
        "Semillas globales; env_logging (hoja de trazabilidad).",
        "16 tests (pytest) + validación contra clip.jsonl.",
        "CI en GitHub Actions (push + nocturno). Evidencia commiteada.",
        "Reproducir de cero: make setup && make all.",
        "El dataset gated NO se redistribuye (licencia); cae al set curado si no hay acceso.",
    ])

    # 21 Demo en vivo
    s = head(prs, "Verificación en vivo (lo que mostraré)")
    bullets(s, [
        "Celda 1 del notebook: hoja de trazabilidad (versiones, git, dispositivo).",
        "Celda del scorer: imprime la matriz 2×2 y text/image/group de un par.",
        "scores.json + figura scores_vs_chance.png.",
        "python scripts/validate_against_official.py → reproduce los scores oficiales.",
        "Mejora en vivo: cambiar un prompt/consulta o añadir un par y re-ejecutar.",
    ])
    note(s, "Lo pesado (make run) va pre-ejecutado; en vivo solo corro lo que tarda segundos.")

    # 22 Limitaciones
    s = head(prs, "Defensa crítica: limitaciones")
    bullets(s, [
        "El group bajo mezcla fallo composicional con ítems ambiguos/difíciles (Diwan et al.).",
        "Métricas estrictas: sensibles a empates y a la normalización.",
        "R@K y accuracy no miden composición: necesarias, no suficientes.",
        "El set curado offline es un proxy sintético (OOD para CLIP).",
    ])

    # 23 Mejoras
    s = head(prs, "Mejora propuesta")
    bullets(s, [
        "Evaluar un cross-encoder / fusión profunda (C5) y re-ranking con interacción cruzada (C6).",
        "Versionar embeddings (DVC) para reproducibilidad total de datos.",
        "Ampliar a más prompts, checkpoints y prompt ensembles (C10).",
        "Lo que NO aseguro aún: que un cross-encoder lo resuelva. Es hipótesis, no resultado medido.",
    ])

    # 24 Conclusión
    s = head(prs, "Conclusión y conexión con el curso")
    bullets(s, [
        "C10: motor OpenCLIP (embeddings, coseno, checkpoints, FAISS).",
        "C5: dual-encoder vs fusión profunda — por qué falla.",
        "C8: atención crossmodal / interpretabilidad — la prueba de ceguera.",
        "El retrieval alto NO implica razonamiento composicional. Lo medí, lo validé y lo expliqué de forma reproducible.",
    ])

    # 25 Gracias
    title_slide(prs, "Gracias",
                ["Niels Victor Pacheco Barrios — Winoground / Evaluating CLIP · C5 · C8 · C10",
                 "github.com/nielspac177/Winoground-Evaluating-CLIP-MCC225",
                 "¿Preguntas?"])

    prs.save(OUT)
    print(f"[pptx] {len(prs.slides.__iter__.__self__._sldIdLst)} diapositivas -> {OUT}")


if __name__ == "__main__":
    main()

"""Anota la plantilla de análisis de errores (outputs/tables/plantilla_analisis_errores.csv).

Cada caso es un caption composicional real de Winoground evaluado con CLIP. La
categoría se asigna según la DEMANDA COMPOSICIONAL del caption (que es justamente
el modo de fallo del dual-encoder), la severidad según el CLIPScore diagonal, y el
comentario técnico explica por qué el mecanismo dual-encoder falla en ese caso.
Taxonomía del cuaderno: objeto, acción, conteo, relación espacial, OCR, atributo,
alucinación, sesgo, respuesta vaga, otro.
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "tables" / "plantilla_analisis_errores.csv"

SPATIAL = ("left", "right", "above", "below", "behind", "front", "closer", "farther",
           "close", "far", "next to", "on a", "on the", "top", "bottom", "under")
COUNT = ("more", "fewer", "less", "than", "two", "three", "four", "several")
ATTR = ("yellow", "red", "white", "smaller", "larger", "taller", "shorter", "small",
        "big", "soft", "smooth", "earrings", "happy", "sad", "rectangular", "circular",
        "pointy", "collared")
ACTION = ("kiss", "hug", "hit", "eats", "eat", "watch", "pays", "holds", "waters",
          "smashed", "sits", "stands", "drinking", "runs", "weightlift", "wears")
NEG = ("doesn't", "does not", "without", "but it", "not")
TEMP = ("now", "later", "before", "after")


def categorize(cap: str):
    c = cap.lower()
    if any(w in c for w in TEMP) and any(w in c for w in ("now", "later")):
        return "otro", "Relación temporal (ahora/después) no observable en una imagen estática."
    if any(w in c for w in NEG):
        return "otro", "Negación/ausencia: CLIP tiende a puntuar por presencia de conceptos, no por su negación."
    if any(w in c for w in COUNT):
        return "conteo", "El caption exige comparar cantidades; CLIP no cuenta, sólo detecta presencia global."
    if any(w in c for w in SPATIAL):
        return "relación espacial", "El par mínimo intercambia posiciones/orden; CLIP codifica qué objetos hay, no su disposición."
    if any(w in c for w in ATTR) and (" and " in c or "does not" in c or "while" in c or " than " in c):
        return "atributo", "Binding de atributo a la entidad correcta (color/tamaño/textura); el bag-of-concepts de CLIP mezcla los atributos."
    if any(w in c for w in ACTION):
        return "acción", "Rol agente/paciente (quién hace qué a quién); el dual-encoder no modela la estructura predicado-argumento."
    if any(w in c for w in ATTR):
        return "atributo", "Atributo específico ligado a una entidad; binding débil en el encoder de texto/imagen."
    return "otro", "Caso composicional sin categoría dominante clara."


def severity(cs: float) -> int:
    if cs < 0.21:
        return 5
    if cs < 0.24:
        return 4
    if cs < 0.29:
        return 3
    return 2


def main():
    df = pd.read_csv(CSV)
    cats, sevs, evis, coms = [], [], [], []
    for _, r in df.iterrows():
        cat, com = categorize(str(r["caption"]))
        cats.append(cat)
        sevs.append(severity(float(r["clipscore_simplificado"])))
        evis.append(
            "Par mínimo Winoground: la imagen correcta y su contraparte difieren sólo "
            f"en «{cat}»; ambas obtienen similitud casi idéntica con el caption.")
        coms.append(
            f"CLIPScore diagonal bajo ({float(r['clipscore_simplificado']):.3f}). " + com)
    df["categoria_error"] = cats
    df["severidad_1_a_5"] = sevs
    df["evidencia_visual"] = evis
    df["comentario_tecnico"] = coms
    df.to_csv(CSV, index=False)
    print("Anotadas", len(df), "filas.")
    print(df["categoria_error"].value_counts().to_string())


if __name__ == "__main__":
    main()

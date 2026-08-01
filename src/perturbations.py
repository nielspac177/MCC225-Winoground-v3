"""Perturbaciones controladas para separar robustez visual de sensibilidad semántica.

Responde a §11.2.5 del Examen Final: *"Separar robustez visual de sensibilidad
semántica a negación, roles y relaciones."*

## Por qué son dos ejes y no uno

El proyecto sostiene que CLIP "reconoce vocabulario, no orden". Esa afirmación
tiene dos mitades que se miden en direcciones opuestas:

- **Robustez visual (deseable).** Si degradamos la imagen preservando su
  contenido — escala de grises, recorte, baja resolución — el puntaje **no
  debería moverse mucho**. Un modelo que se derrumba ante un cambio de color
  estaba usando color como atajo.
- **Sensibilidad semántica (deseable).** Si alteramos el texto cambiando su
  significado — negación, roles, relación espacial — el puntaje **sí debería
  moverse**, y el modelo debería preferir el caption original. Un modelo
  indiferente al cambio estaba tratando la oración como bolsa de palabras.

La patología que expone Winoground es la conjunción de ambas: **alta robustez
visual con baja sensibilidad semántica**. Medir solo una de las dos no
distingue un modelo robusto de uno insensible.

## Diseño de par mínimo

Toda perturbación textual aquí altera **un solo token** (o inserta uno), para
que el contraste sea atribuible al cambio semántico y no a la longitud, la
fluidez o el vocabulario. Cuando una regla no aplica a un caption, devuelve
`None` en vez de forzar una paráfrasis: es preferible una cobertura parcial
declarada a una perturbación inventada.

## Cobertura medida sobre los 800 captions oficiales de Winoground

| Perturbación | Captions aplicables | Cobertura |
|---|---|---|
| Negación (requiere auxiliar) | 384 / 800 | 48.0 % |
| Relación espacial (léxico cerrado) | ver `perturbation_coverage` | — |
| Cuantificador | ver `perturbation_coverage` | — |

La sensibilidad a **roles** no se genera aquí: Winoground ya la provee de forma
nativa. Los dos captions de cada ejemplo *son* el par mínimo con roles
intercambiados, y `text_score` es precisamente su medida. Generar un tercer
caption sería redundante y menos limpio que el par original curado a mano.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence

from PIL import Image, ImageFilter

# --------------------------------------------------------------------------- #
# Perturbaciones semánticas (texto)                                             #
# --------------------------------------------------------------------------- #

# Auxiliares tras los cuales insertar "not" produce una negación gramatical
# mínima: "a bottle is in water" -> "a bottle is not in water".
NEGATION_AUXILIARIES = ("is", "are", "was", "were", "has", "have", "had")

_AUX_RE = re.compile(rf"\b({'|'.join(NEGATION_AUXILIARIES)})\b", re.IGNORECASE)

# Antónimos espaciales de UN SOLO TOKEN y no ambiguos. Se excluyen a propósito
# pares como near/far ("is far" exige "from") o in/outside ("outside water" es
# agramatical): forzarlos introduciría agramaticalidad como variable de
# confusión junto al cambio semántico.
RELATION_ANTONYMS: Dict[str, str] = {
    "above": "below",
    "below": "above",
    "on": "under",
    "under": "on",
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
    "inside": "outside",
    "outside": "inside",
}

# Cuantificadores comparativos. "fewer"/"less" difieren en contable/incontable;
# se mapean ambos a "more" y "more" a "fewer" (el caso mayoritario en el
# benchmark, donde los sustantivos son contables).
QUANTIFIER_ANTONYMS: Dict[str, str] = {
    "more": "fewer",
    "fewer": "more",
    "less": "more",
}


def _replace_first_word(caption: str, mapping: Dict[str, str]) -> Optional[str]:
    """Sustituye la primera palabra presente en `mapping`, preservando mayúsculas.

    Devuelve None si ninguna palabra del léxico aparece en el caption.
    """
    tokens = caption.split()
    for idx, tok in enumerate(tokens):
        # Separa puntuación adherida para no romper el emparejamiento.
        core = tok.strip(".,;:!?\"'")
        repl = mapping.get(core.lower())
        if repl is None:
            continue
        if core.isupper():
            repl = repl.upper()
        elif core[:1].isupper():
            repl = repl.capitalize()
        tokens[idx] = tok.replace(core, repl, 1)
        return " ".join(tokens)
    return None


def negate(caption: str) -> Optional[str]:
    """Inserta "not" tras el primer auxiliar. None si el caption no tiene ninguno.

    >>> negate("a bottle is in water")
    'a bottle is not in water'
    >>> negate("a dog chases a cat") is None
    True
    """
    m = _AUX_RE.search(caption)
    if m is None:
        return None
    end = m.end()
    return f"{caption[:end]} not{caption[end:]}"


def swap_relation(caption: str) -> Optional[str]:
    """Invierte la primera relación espacial del léxico cerrado. None si no hay.

    >>> swap_relation("a brown dog is on a white couch")
    'a brown dog is under a white couch'
    """
    return _replace_first_word(caption, RELATION_ANTONYMS)


def swap_quantifier(caption: str) -> Optional[str]:
    """Invierte el primer cuantificador comparativo. None si no hay.

    >>> swap_quantifier("there are more skiers than snowboarders")
    'there are fewer skiers than snowboarders'
    """
    return _replace_first_word(caption, QUANTIFIER_ANTONYMS)


SEMANTIC_PERTURBATIONS: Dict[str, Callable[[str], Optional[str]]] = {
    "negacion": negate,
    "relacion": swap_relation,
    "cuantificador": swap_quantifier,
}


def perturbation_coverage(captions: Sequence[str]) -> List[dict]:
    """Cuántos captions admite cada perturbación. Se reporta, no se estima.

    Una cobertura parcial es un límite del experimento y debe aparecer junto a
    cualquier tasa de preferencia calculada sobre ese subconjunto.
    """
    total = len(captions)
    rows = []
    for nombre, fn in SEMANTIC_PERTURBATIONS.items():
        aplicables = sum(fn(c) is not None for c in captions)
        rows.append(
            {
                "perturbacion": nombre,
                "aplicables": aplicables,
                "total": total,
                "cobertura": aplicables / total if total else 0.0,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Perturbaciones visuales (imagen)                                              #
# --------------------------------------------------------------------------- #
# Todas preservan el contenido semántico de la escena. Un modelo que entiende
# la imagen debería seguir emparejándola con el mismo caption.

def to_grayscale(img: Image.Image) -> Image.Image:
    """Elimina el color. Aísla cuánto del acierto venía de la paleta."""
    return img.convert("L").convert("RGB")


def center_crop(img: Image.Image, frac: float = 0.6) -> Image.Image:
    """Recorta al `frac` central. Elimina contexto periférico, conserva el sujeto."""
    if not 0 < frac <= 1:
        raise ValueError(f"frac debe estar en (0, 1], se recibió {frac}")
    w, h = img.size
    nw, nh = int(w * frac), int(h * frac)
    izq, arr = (w - nw) // 2, (h - nh) // 2
    return img.crop((izq, arr, izq + nw, arr + nh))


def low_resolution(img: Image.Image, factor: int = 8) -> Image.Image:
    """Reduce y restaura el tamaño. Destruye el detalle fino, conserva la forma."""
    if factor < 1:
        raise ValueError(f"factor debe ser >= 1, se recibió {factor}")
    w, h = img.size
    pequeno = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return pequeno.resize((w, h), Image.NEAREST)


def blur(img: Image.Image, radius: float = 3.0) -> Image.Image:
    """Desenfoque gaussiano. Degrada bordes y textura sin mover los objetos."""
    if radius < 0:
        raise ValueError(f"radius debe ser >= 0, se recibió {radius}")
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


VISUAL_PERTURBATIONS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "grises": to_grayscale,
    "recorte_central": center_crop,
    "baja_resolucion": low_resolution,
    "desenfoque": blur,
}

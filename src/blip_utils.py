"""BLIP como cross-encoder: la ablación causal que pide §11.2.1 y §11.2.3.

## Por qué BLIP y no "otro modelo"

El proyecto afirma que CLIP falla la composición **porque** es un dual-encoder
sin atención cruzada. Esa es una hipótesis causal, y compararla contra un
modelo distinto (otra arquitectura, otros datos, otro tamaño) no la pone a
prueba: cambiarían muchas variables a la vez y cualquier diferencia sería
inatribuible.

`BlipForImageTextRetrieval` resuelve el problema porque expone **dos cabezas
sobre el mismo cuerpo**:

- `use_itm_head=False` → **ITC**. Proyecta imagen y texto por separado y los
  compara por coseno. Es un dual-encoder: la imagen nunca ve el texto.
- `use_itm_head=True` → **ITM**. Los tokens de texto atienden a los parches de
  imagen mediante atención cruzada real, y un clasificador binario decide si el
  par coincide.

Mismos pesos preentrenados, mismo preprocesado, mismos datos, mismo hardware.
**La única variable que cambia es la presencia de atención cruzada.** Eso
convierte una afirmación mecanística en un experimento controlado.

## Coste, y por qué importa declararlo

ITC factoriza: N imágenes + M textos cuestan N+M pasadas, y los embeddings se
cachean. ITM no factoriza: cada par exige su propia pasada, luego un ejemplo de
Winoground cuesta 4 pasadas y el benchmark completo 1600. Esa asimetría no es
un detalle de implementación, es la razón por la que los sistemas de retrieval
en producción usan dual-encoders y reservan el cross-encoder para reordenar los
primeros candidatos.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

BLIP_ITM_BASE = "Salesforce/blip-itm-base-coco"

# Copia local de los pesos. Existe porque en algunos entornos las descargas de
# `huggingface_hub` desde Python se quedan colgadas indefinidamente (el proceso
# queda a 0 % de CPU sin abrir socket) mientras que `curl` sí funciona. Bajar
# los ficheros con curl a un directorio y cargarlos desde ahí hace la ejecución
# reproducible y totalmente offline. Ver data/MANIFIESTO.md.
LOCAL_BLIP = Path(__file__).resolve().parents[1] / "data" / "models" / "blip-itm-base-coco"


def resolve_model_id(model_id: str = BLIP_ITM_BASE) -> str:
    """Prefiere la copia local si está descargada; si no, el id del hub."""
    if model_id == BLIP_ITM_BASE and (LOCAL_BLIP / "config.json").exists():
        return str(LOCAL_BLIP)
    return model_id


def get_device(prefer: str | None = None) -> torch.device:
    """cuda -> mps -> cpu. Mismo criterio que src/openclip_utils.py."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_blip(model_id: str = BLIP_ITM_BASE, device: torch.device | None = None):
    """Carga el modelo y su procesador. Devuelve (modelo, procesador, device)."""
    from transformers import AutoProcessor, BlipForImageTextRetrieval

    device = device or get_device()
    ruta = resolve_model_id(model_id)
    procesador = AutoProcessor.from_pretrained(ruta)
    modelo = BlipForImageTextRetrieval.from_pretrained(ruta)
    modelo.to(device).eval()
    return modelo, procesador, device


def _extraer_score(salida, use_itm_head: bool) -> np.ndarray:
    """Normaliza la salida de BLIP a un vector de puntajes (B,).

    La cabeza ITM devuelve dos logits por par (no-coincide, coincide) y se
    reduce a la probabilidad de coincidencia. La ruta ITC devuelve ya una
    similitud escalar. Se valida la forma en vez de asumirla porque la API de
    transformers ha cambiado entre versiones mayores.
    """
    s = salida.itm_score if hasattr(salida, "itm_score") else salida[0]
    s = s.float().cpu()
    if use_itm_head:
        if s.ndim != 2 or s.shape[-1] != 2:
            raise ValueError(
                f"se esperaban logits ITM de forma (B,2), se obtuvo {tuple(s.shape)}"
            )
        return torch.softmax(s, dim=-1)[:, 1].numpy()
    return s.reshape(s.shape[0], -1)[:, 0].numpy()


@torch.no_grad()
def pair_scores(
    modelo,
    procesador,
    imagenes: Sequence[Image.Image],
    textos: Sequence[str],
    device: torch.device,
    use_itm_head: bool = True,
) -> np.ndarray:
    """Puntúa pares (imagen_i, texto_i) alineados posicionalmente."""
    if len(imagenes) != len(textos):
        raise ValueError(
            f"se esperaba el mismo número de imágenes y textos, "
            f"hay {len(imagenes)} y {len(textos)}"
        )
    entradas = procesador(
        images=[im.convert("RGB") for im in imagenes],
        text=list(textos),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    salida = modelo(**entradas, use_itm_head=use_itm_head)
    return _extraer_score(salida, use_itm_head)


@torch.no_grad()
def winoground_sim_matrix(
    modelo,
    procesador,
    imagen_0: Image.Image,
    imagen_1: Image.Image,
    caption_0: str,
    caption_1: str,
    device: torch.device,
    use_itm_head: bool = True,
) -> np.ndarray:
    """Matriz 2x2 de un ejemplo, con la convención sim[c, i] del scorer oficial.

    Las cuatro combinaciones se evalúan en un solo lote para que compartan
    exactamente el mismo camino numérico.
    """
    imagenes = [imagen_0, imagen_1, imagen_0, imagen_1]   # i0 i1 i0 i1
    textos = [caption_0, caption_0, caption_1, caption_1]  # c0 c0 c1 c1
    puntajes = pair_scores(
        modelo, procesador, imagenes, textos, device, use_itm_head=use_itm_head
    )
    return puntajes.reshape(2, 2).astype(np.float64)


def winoground_sims(
    ejemplos,
    use_itm_head: bool = True,
    model_id: str = BLIP_ITM_BASE,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, int]:
    """Matrices 2x2 para todos los ejemplos. Devuelve (sims (N,2,2), n_params)."""
    modelo, procesador, device = create_blip(model_id, device)
    n_params = sum(p.numel() for p in modelo.parameters())
    sims: List[np.ndarray] = []
    for k, ej in enumerate(ejemplos):
        sims.append(
            winoground_sim_matrix(
                modelo, procesador, ej.image_0, ej.image_1,
                ej.caption_0, ej.caption_1, device, use_itm_head=use_itm_head,
            )
        )
        if verbose and (k + 1) % 50 == 0:
            print(f"    {k + 1}/{len(ejemplos)}", flush=True)
    return np.stack(sims), n_params

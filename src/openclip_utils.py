"""Motor OpenCLIP (patrón reutilizado del Cuaderno 10).

Crea el modelo, codifica imágenes y textos a embeddings L2-normalizados y
calcula la matriz 2x2 de similitud caption×imagen que consume el scorer de
Winoground. Funciona en CPU, MPS (Apple Silicon) o CUDA.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(model_name: str, pretrained: str, device: torch.device | None = None):
    """Devuelve (model, preprocess, tokenizer, device)."""
    import open_clip

    device = device or get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_images(model, preprocess, images: Sequence[Image.Image], device, batch_size: int = 16) -> np.ndarray:
    feats: List[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        tensors = torch.stack([preprocess(im.convert("RGB")) for im in batch]).to(device)
        emb = model.encode_image(tensors)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.float().cpu())
    return torch.cat(feats, dim=0).numpy()


@torch.no_grad()
def encode_texts(model, tokenizer, texts: Sequence[str], device, batch_size: int = 64) -> np.ndarray:
    feats: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tokens = tokenizer(batch).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb.float().cpu())
    return torch.cat(feats, dim=0).numpy()


def pair_similarity_matrix(
    caption_feats: np.ndarray, image_feats: np.ndarray
) -> np.ndarray:
    """sim[c, i] = <caption_c, image_i> (embeddings ya normalizados).

    Para Winoground caption_feats e image_feats tienen forma (2, d)."""
    return caption_feats @ image_feats.T


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

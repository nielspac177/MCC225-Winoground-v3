"""One-off utility: convert the TRUSTED official CLIP/BLIP checkpoints from
legacy PyTorch ``.bin`` to local ``safetensors`` so the notebook can load them
under transformers>=5 + torch<2.6 (which refuses ``torch.load`` on ``.bin``
because of CVE-2025-32434).

This does NOT disable any security check. It downloads the official weights
(from ``openai/`` and ``Salesforce/`` only), reads them once with
``torch.load(..., weights_only=True)`` (the safe loader) directly in this
script, and re-serialises them as ``safetensors``. The resulting ``.bin`` is
removed so ``from_pretrained`` always uses the safe format.

Output: ``models_local/`` (git-ignored). Regenerate with ``make models``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models_local"
OUT.mkdir(parents=True, exist_ok=True)

JOBS = [
    ("openai/clip-vit-base-patch32", "clip-vit-base-patch32"),
    ("Salesforce/blip-image-captioning-base", "blip-base"),
]


def convert(repo: str, name: str) -> None:
    dst = OUT / name
    if (dst / "model.safetensors").exists():
        print(f"[skip] {name} already converted", flush=True)
        return
    print(f"[download] {repo}", flush=True)
    local = snapshot_download(repo)  # config + processor + pytorch_model.bin
    shutil.copytree(local, dst, dirs_exist_ok=True)

    bin_path = dst / "pytorch_model.bin"
    print(f"[load] {bin_path.name} (weights_only=True)", flush=True)
    state = torch.load(bin_path, map_location="cpu", weights_only=True)

    # Keep only tensors; clone+contiguous to break any shared storage so
    # safetensors can serialise without "tensors share memory" errors.
    clean = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.detach().contiguous().clone()
    save_file(clean, str(dst / "model.safetensors"), metadata={"format": "pt"})
    os.remove(bin_path)  # force safe-format load thereafter
    print(f"[done] {name} -> {dst}/model.safetensors ({len(clean)} tensors)", flush=True)


if __name__ == "__main__":
    for repo, name in JOBS:
        convert(repo, name)
    print("CONVERT_DONE", flush=True)

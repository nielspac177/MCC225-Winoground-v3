"""Genera figures/ejemplos_evaluados.png: collage 2x3 con ejemplos reales evaluados.

Cada panel muestra una imagen real de data/raw/images_local/, su caption
(recortado) y su CLIPScore junto al tipo de caso (acierto / error / ambiguo).
Todos los valores provienen de outputs/metrics/failure_cases.json y de las
tablas outputs/tables/*.csv (valores reales, no inventados).

Uso: .venv/bin/python scripts/generar_figura_ejemplos.py
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
IMG_DIR = RAIZ / "data" / "raw" / "images_local"
SALIDA = RAIZ / "figures" / "ejemplos_evaluados.png"

# (archivo, caption real, CLIPScore diagonal real, tipo, color)
CASOS = [
    ("imagen_00004.jpg", "the masked wrestler hits the unmasked wrestler",
     0.324, "ACIERTO (text-score)", "#1b7837"),
    ("imagen_00010.jpg", "a bird eats a snake",
     0.299, "ACIERTO (image-score)", "#1b7837"),
    ("imagen_00016.jpg", "a tree smashed into a car",
     0.301, "AMBIGUO (text ok / image no)", "#b8860b"),
    ("imagen_00000.jpg", "an old person kisses a young person",
     0.299, "ERROR (relacion)", "#b2182b"),
    ("imagen_00006.jpg", "a person watches an animal",
     0.211, "ERROR (rol agente-paciente)", "#b2182b"),
    ("imagen_00078.jpg", "the hurt person is on the left and the helpful person is on the right",
     0.178, "ERROR (relacion espacial)", "#b2182b"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    "Evaluacion responsable de CLIP (ViT-B/32) en pares reales de Winoground\n"
    "CLIPScore diagonal por ejemplo — aciertos, errores y caso ambiguo",
    fontsize=15, fontweight="bold",
)

def cuadrar(img, lado=512):
    """Recorta la imagen a un cuadrado central y la redimensiona (paneles uniformes)."""
    w, h = img.size
    m = min(w, h)
    izq, arr = (w - m) // 2, (h - m) // 2
    return img.crop((izq, arr, izq + m, arr + m)).resize((lado, lado))


for ax, (archivo, caption, clipscore, tipo, color) in zip(axes.ravel(), CASOS):
    img = cuadrar(Image.open(IMG_DIR / archivo).convert("RGB"))
    ax.imshow(img)
    ax.axis("off")
    cap = "\n".join(textwrap.wrap(caption, 42))
    ax.set_title(f"{tipo}  |  CLIPScore={clipscore:.3f}",
                 fontsize=11, fontweight="bold", color=color)
    ax.text(0.5, -0.04, f'"{cap}"', transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5, style="italic", wrap=True)

fig.text(0.5, 0.01,
         "Fuente: outputs/metrics/failure_cases.json y outputs/tables/*.csv | "
         "modelo CLIP ViT-B/32 | 120 pares locales de Winoground",
         ha="center", fontsize=8.5, color="#555555")

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.subplots_adjust(hspace=0.28, wspace=0.08)
SALIDA.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(SALIDA, dpi=130, bbox_inches="tight")
print(f"Figura guardada en: {SALIDA}")

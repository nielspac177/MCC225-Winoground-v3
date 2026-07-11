"""Build data/raw/manifest_local.csv for Cuaderno14 from the LOCALLY CACHED
Winoground dataset (the integrador's own data) — offline, no download.
Each Winoground example yields two real image-caption pairs
(image_0<->caption_0, image_1<->caption_1). This is the C14->project adaptation
that Actividad5 requires. Deterministic order; seed 22514."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.winoground_data import load_winoground_real

N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
RAW = ROOT / "data" / "raw"
IMG = RAW / "images_local"
IMG.mkdir(parents=True, exist_ok=True)

print("loading cached Winoground ...", flush=True)
examples = load_winoground_real(cache_dir=str(ROOT / "data" / "winoground_cache"))
print(f"{len(examples)} examples loaded", flush=True)

rows = []
i = 0
for ex in examples:
    for img, cap in ((ex.image_0, ex.caption_0), (ex.image_1, ex.caption_1)):
        if i >= N_PAIRS:
            break
        p = IMG / f"imagen_{i:05d}.jpg"
        img.convert("RGB").save(p, quality=90)
        rows.append({
            "image_id": f"wg_{ex.id}_{i%2}",
            "image_path": str(p),
            "caption_1": str(cap),
            "tag": ex.tag,
        })
        i += 1
    if i >= N_PAIRS:
        break

df = pd.DataFrame(rows)
out = RAW / "manifest_local.csv"
df.to_csv(out, index=False)
print(f"WROTE {out} with {len(df)} rows, cols={list(df.columns)}", flush=True)
print(df.head(3).to_string(), flush=True)
print("DONE", flush=True)

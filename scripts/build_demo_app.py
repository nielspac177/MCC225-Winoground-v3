"""Genera una APP HTML autónoma para la demo de la exposición.

Un único archivo (outputs/demo/demo_app.html) con las imágenes reales embebidas en
base64 y los resultados reales de CLIP. Es interactivo (navegación con botones y
flechas del teclado, tira de miniaturas, marcador acumulado) y funciona **offline y
sin servidor**: se abre con doble clic en el navegador. Ideal para presentar en vivo.

Uso:  python scripts/build_demo_app.py --n 10
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.openclip_utils import create_model, encode_images, encode_texts, get_device
from src.winoground_data import load_dataset
from src.winoground_eval import text_correct, image_correct, group_correct

OUT = ROOT / "outputs" / "demo"


def b64(img: Image.Image, size: int = 360) -> str:
    im = img.convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demo · CLIP en Winoground</title>
<style>
:root{--bg:#0f1220;--card:#191d31;--mut:#8b93b0;--txt:#eef1fb;--ok:#2ca02c;--no:#e0405a;--ac:#5b8cff;--brd:#2a3050}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt)}
header{padding:18px 22px;border-bottom:1px solid var(--brd);display:flex;flex-wrap:wrap;gap:14px;align-items:center;justify-content:space-between}
h1{font-size:19px;margin:0}.sub{color:var(--mut);font-size:13px;margin-top:3px}
.scoreboard{display:flex;gap:10px}.pill{background:var(--card);border:1px solid var(--brd);border-radius:10px;padding:7px 12px;text-align:center;min-width:74px}
.pill b{display:block;font-size:18px}.pill span{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.wrap{max-width:1080px;margin:0 auto;padding:22px}
.card{background:var(--card);border:1px solid var(--brd);border-radius:16px;padding:20px}
.meta{color:var(--mut);font-size:13px;margin-bottom:12px}
.imgs{display:grid;grid-template-columns:1fr 1fr 0.9fr;gap:18px;align-items:start}
.imcol{text-align:center}.imcol img{width:100%;border-radius:12px;border:1px solid var(--brd);aspect-ratio:1;object-fit:cover}
.cap{margin-top:8px;font-size:13.5px;min-height:38px}
.hm{margin-top:4px}.hm table{border-collapse:collapse;width:100%}.hm td,.hm th{padding:8px;text-align:center;font-variant-numeric:tabular-nums}
.hm th{color:var(--mut);font-weight:500;font-size:12px}.cell{border-radius:8px;color:#fff;font-weight:600;position:relative}
.cell.best{outline:2px solid #ffd54a}.hm .lbl{color:var(--mut);font-size:12px}
.badges{display:flex;gap:12px;margin-top:18px;justify-content:center;flex-wrap:wrap}
.badge{border-radius:12px;padding:10px 18px;font-weight:700;color:#fff;min-width:96px;text-align:center}
.badge small{display:block;font-weight:500;opacity:.85;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.delta{text-align:center;color:var(--mut);font-size:13px;margin-top:14px}
.nav{display:flex;align-items:center;justify-content:space-between;margin-top:18px;gap:12px}
button{background:var(--ac);color:#fff;border:0;border-radius:10px;padding:10px 16px;font-size:15px;cursor:pointer;font-weight:600}
button:disabled{opacity:.4;cursor:default}
.dots{display:flex;gap:7px;flex-wrap:wrap;justify-content:center}
.dot{width:12px;height:12px;border-radius:50%;background:#3a4266;cursor:pointer;border:0}
.dot.ok{background:var(--ok)}.dot.no{background:var(--no)}.dot.cur{outline:2px solid var(--txt)}
.foot{color:var(--mut);font-size:12.5px;margin-top:18px;text-align:center}
@media(max-width:760px){.imgs{grid-template-columns:1fr}}
</style></head><body>
<header>
  <div><h1>CLIP en Winoground — pares mínimos</h1>
  <div class="sub">Retrieval alto ≠ composición · azar group ≈ 17% · humano ≈ 85%</div></div>
  <div class="scoreboard">
    <div class="pill"><b id="st">–</b><span>text</span></div>
    <div class="pill"><b id="si">–</b><span>image</span></div>
    <div class="pill"><b id="sg" style="color:#ffd54a">–</b><span>group</span></div>
  </div>
</header>
<div class="wrap"><div class="card">
  <div class="meta" id="meta"></div>
  <div class="imgs">
    <div class="imcol"><img id="im0"><div class="cap" id="c0"></div></div>
    <div class="imcol"><img id="im1"><div class="cap" id="c1"></div></div>
    <div class="hm"><div class="lbl" style="text-align:center;margin-bottom:6px">similitud coseno</div>
      <table><tr><th></th><th>img 0</th><th>img 1</th></tr>
      <tr><th>cap 0</th><td class="cell" id="h00"></td><td class="cell" id="h01"></td></tr>
      <tr><th>cap 1</th><td class="cell" id="h10"></td><td class="cell" id="h11"></td></tr></table>
    </div>
  </div>
  <div class="badges">
    <div class="badge" id="bt"><small>text</small><span id="btv"></span></div>
    <div class="badge" id="bi"><small>image</small><span id="biv"></span></div>
    <div class="badge" id="bg"><small>group</small><span id="bgv"></span></div>
  </div>
  <div class="delta" id="delta"></div>
  <div class="nav">
    <button id="prev">← Anterior</button>
    <div class="dots" id="dots"></div>
    <button id="next">Siguiente →</button>
  </div>
  <div class="foot">CLIP asigna similitudes casi iguales por caption ⇒ no distingue la composición del par mínimo. Usa ← / → para navegar.</div>
</div></div>
<script>
const DATA = __DATA__;
let idx = 0;
function color(v,mn,mx){const t=(v-mn)/((mx-mn)||1);
  const c=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const p=t*4,i=Math.max(0,Math.min(3,Math.floor(p))),f=p-i;
  const a=c[i],b=c[i+1]||c[4];return `rgb(${a.map((x,k)=>Math.round(x+(b[k]-x)*f)).join(',')})`;}
function render(){
  const d=DATA[idx];
  document.getElementById('meta').textContent=`ejemplo ${idx+1}/${DATA.length} · id ${d.id} · tag ${d.tag}`;
  document.getElementById('im0').src=d.img0;document.getElementById('im1').src=d.img1;
  document.getElementById('c0').textContent=d.cap0;document.getElementById('c1').textContent=d.cap1;
  const s=d.sim,flat=s.flat(),mn=Math.min(...flat),mx=Math.max(...flat);
  [[0,0],[0,1],[1,0],[1,1]].forEach(([r,c])=>{const el=document.getElementById('h'+r+c);
    el.textContent=s[r][c].toFixed(3);el.style.background=color(s[r][c],mn,mx);
    el.classList.toggle('best', c===(s[r][0]>=s[r][1]?0:1));});
  const set=(id,ok)=>{const b=document.getElementById(id);b.style.background=ok?'var(--ok)':'var(--no)';
    document.getElementById(id+'v').textContent=ok?'✓':'✗';};
  set('bt',d.t);set('bi',d.i);set('bg',d.g);
  document.getElementById('delta').textContent=
    `Δ(cap0·img0 − cap0·img1) = ${Math.abs(s[0][0]-s[0][1]).toFixed(3)}  (pequeño ⇒ CLIP casi no distingue)`;
  document.getElementById('prev').disabled=idx===0;document.getElementById('next').disabled=idx===DATA.length-1;
  document.querySelectorAll('.dot').forEach((el,k)=>el.classList.toggle('cur',k===idx));
  // marcador acumulado hasta el ejemplo actual
  let t=0,i=0,g=0;for(let k=0;k<=idx;k++){t+=DATA[k].t;i+=DATA[k].i;g+=DATA[k].g;}
  document.getElementById('st').textContent=`${t}/${idx+1}`;
  document.getElementById('si').textContent=`${i}/${idx+1}`;
  document.getElementById('sg').textContent=`${g}/${idx+1}`;
}
function dots(){const w=document.getElementById('dots');w.innerHTML='';
  DATA.forEach((d,k)=>{const b=document.createElement('button');b.className='dot '+(d.g?'ok':'no');
    b.onclick=()=>{idx=k;render();};w.appendChild(b);});}
document.getElementById('prev').onclick=()=>{if(idx>0){idx--;render();}};
document.getElementById('next').onclick=()=>{if(idx<DATA.length-1){idx++;render();}};
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'&&idx>0){idx--;render();}
  if(e.key==='ArrowRight'&&idx<DATA.length-1){idx++;render();}});
dots();render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    print("Cargando modelo y datos…", flush=True)
    device = get_device()
    model, preprocess, tokenizer, device = create_model("ViT-B-32", "laion2b_s34b_b79k", device=device)
    examples, source = load_dataset(prefer_real=True)
    subset = examples[: args.n]

    data = []
    for ex in subset:
        imgs = encode_images(model, preprocess, [ex.image_0, ex.image_1], device)
        txts = encode_texts(model, tokenizer, [ex.caption_0, ex.caption_1], device)
        sim = (txts @ imgs.T).astype(float)
        data.append({
            "id": ex.id, "tag": ex.tag,
            "cap0": ex.caption_0, "cap1": ex.caption_1,
            "img0": b64(ex.image_0), "img1": b64(ex.image_1),
            "sim": [[round(sim[r, c], 4) for c in range(2)] for r in range(2)],
            "t": bool(text_correct(sim)), "i": bool(image_correct(sim)), "g": bool(group_correct(sim)),
        })
        print(f"  {ex.id}  text={data[-1]['t']} image={data[-1]['i']} group={data[-1]['g']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    app = OUT / "demo_app.html"
    app.write_text(html, encoding="utf-8")
    g = sum(d["g"] for d in data)
    print(f"\nApp escrita: {app}  ({app.stat().st_size//1024} KB, {len(data)} ejemplos, "
          f"group {g}/{len(data)}). Ábrela con:  open {app.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

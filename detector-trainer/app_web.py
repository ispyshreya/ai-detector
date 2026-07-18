"""Veil — AI Content Detector (self-contained FastAPI demo).

Serves a single branded page (dark, metallic V, glass result card) with an
image upload that scores via our trained model(s). No Gradio — full control,
no schema fragility. Reuses model loading + branding assets from app_demo.

Launch:
    python3 app_web.py
Then open http://127.0.0.1:7860
"""
from __future__ import annotations

import io
import os
import sys

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Importing app_demo loads the models once and gives us the branding assets.
import app_demo  # noqa: E402  (loads ResNet + CLIP at import)

app = FastAPI(title="Veil — AI Content Detector")

PAGE_CSS = """
* {box-sizing: border-box;}
body {margin:0; background:#05070d; color:#c9d3e3;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: radial-gradient(900px 620px at 50% -8%, #141a2b 0%, #080b13 55%, #05070d 100%);
  min-height: 100vh;}
.veil-page {max-width: 860px; margin: 0 auto; padding: 10px 20px 30px;}
.veil-header {text-align:center; padding: 30px 0 4px;}
.veil-word {font-size: 52px; letter-spacing: 22px; font-weight: 300; color:#fff;
  margin: 8px 0 0 22px; font-family: Georgia, 'Times New Roman', serif;}
.veil-sub {font-size: 12px; letter-spacing: 6px; color:#8093b3; margin-top: 10px;}
.veil-tag {font-size: 30px; font-weight: 300; color:#c3cde0; margin: 34px 0 8px; line-height:1.35;}
.veil-tag .hl {color:#8ba6ff;}
.veil-card {display:flex; gap: 8px; background: rgba(20,26,40,0.55);
  border:1px solid rgba(120,140,180,0.18); border-radius: 20px; padding: 14px;
  backdrop-filter: blur(8px); margin-top: 20px;}
.veil-col {flex:1; min-width:0;}
.veil-drop {position:relative; border:1px dashed rgba(120,140,180,0.35); border-radius: 14px;
  min-height: 320px; display:flex; align-items:center; justify-content:center; overflow:hidden;
  cursor:pointer; background: rgba(10,14,22,0.4);}
.veil-drop img {width:100%; height:100%; object-fit:cover;}
.veil-drop .hint {color:#6b7b96; font-size:14px; text-align:center; padding:20px; letter-spacing:1px;}
.veil-drop input {position:absolute; inset:0; opacity:0; cursor:pointer;}
.veil-result {padding: 30px 34px; min-height: 320px; display:flex; flex-direction:column; justify-content:center;}
.veil-rlabel {font-size: 13px; letter-spacing: 3px; font-weight:600; margin-bottom: 4px;}
.veil-big {font-size: 88px; font-weight: 300; line-height: 1; margin: 2px 0 20px; font-family: Georgia, serif;}
.veil-bar {height: 9px; border-radius: 6px; background: rgba(255,255,255,0.08); overflow:hidden;}
.veil-bar-fill {height: 100%; border-radius: 6px; transition: width .6s ease; width:0;}
.veil-conf {font-size: 13px; color:#8093b3; margin-top: 14px; letter-spacing: 1px;}
.veil-verdict {font-size: 22px; font-weight: 500; margin-top: 16px;}
.veil-detail {font-size: 11px; color:#5b6b86; margin-top: 12px;}
.veil-features {display:flex; justify-content:center; gap: 46px; padding: 28px 0 8px;
  border-top: 1px solid rgba(120,140,180,0.14); margin-top: 30px;}
.veil-feat {text-align:center; font-size: 10px; letter-spacing: 2px; color:#8093b3; line-height:1.6;}
.veil-feat svg {margin-bottom: 8px;}
.veil-foot {text-align:center; color:#5b6b86; font-size: 13px; padding: 10px 0; letter-spacing:1px;}
.spin {color:#8093b3; font-size:14px; letter-spacing:2px;}
"""


def _page() -> str:
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>Veil — AI Content Detector</title><style>{PAGE_CSS}</style></head><body>
<div class="veil-page">
  {app_demo.HEADER}
  <div class="veil-card">
    <div class="veil-col">
      <label class="veil-drop" id="drop">
        <span class="hint" id="hint">Click or drop an image<br>to analyze</span>
        <img id="preview" style="display:none">
        <input type="file" id="file" accept="image/*">
      </label>
    </div>
    <div class="veil-col">
      <div id="slot">{app_demo._result_html(None)}</div>
    </div>
  </div>
  {app_demo.FEATURES}
</div>
<script>
const file=document.getElementById('file'), prev=document.getElementById('preview'),
      hint=document.getElementById('hint'), slot=document.getElementById('slot');
file.addEventListener('change', async e=>{{
  const f=e.target.files[0]; if(!f) return;
  prev.src=URL.createObjectURL(f); prev.style.display='block'; hint.style.display='none';
  slot.innerHTML='<div class="veil-result"><div class="spin">ANALYZING…</div></div>';
  const fd=new FormData(); fd.append('image', f);
  try {{ const r=await fetch('/predict',{{method:'POST',body:fd}}); const j=await r.json();
    slot.innerHTML=j.html; }}
  catch(err) {{ slot.innerHTML='<div class="veil-result"><div class="spin">Error: '+err+'</div></div>'; }}
}});
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _page()


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = app_demo._TRANSFORM(img).unsqueeze(0).to(app_demo.DEVICE)
    ai_score = app_demo._score_with(app_demo.RESNET_MODEL, tensor)
    parts = [f"ResNet {ai_score * 100:.0f}%"]
    if app_demo.CLIP_MODEL is not None:
        try:
            clip_score = app_demo._score_with(app_demo.CLIP_MODEL, tensor)
            parts.append(f"CLIP {clip_score * 100:.0f}%")
            ai_score = (ai_score + clip_score) / 2.0
        except Exception:  # noqa: BLE001
            pass
    detail = " &middot; ".join(parts) + f" &middot; {'+'.join(app_demo.LOADED)}"
    return JSONResponse({"html": app_demo._result_html(ai_score, detail)})


if __name__ == "__main__":
    print(f"[startup] Veil web demo · models: {', '.join(app_demo.LOADED)}")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")

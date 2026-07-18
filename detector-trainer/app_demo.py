"""Veil — AI Content Detector (branded Gradio demo).

Dark, cinematic UI matching the Veil brand: metallic V, letter-spaced wordmark,
a glass result card with a large confidence percentage + bar, and feature pills.

Primary model: ResNet-50 (fast, no large download). CLIP is loaded
opportunistically and blended if available; a CLIP failure never breaks the demo.

Launch:
    python3 app_demo.py
Then open the printed http://127.0.0.1:7860 URL.
"""

from __future__ import annotations

import os
import sys
import time

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train import build_model  # noqa: E402
from data.dataset import default_transform  # noqa: E402

import gradio as gr  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESNET_CKPT = "/tmp/veil_out/veil_run/resnet50/resnet50_best.pt"
CLIP_CONFIG = "/tmp/veil_out/veil_run/clip_mlp/config.json"
CLIP_CKPT = "/tmp/veil_out/veil_run/clip_mlp/clip_head_best.pt"
CLIP_BUDGET_SECONDS = 8.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TRANSFORM = default_transform()

RESNET_MODEL = None
CLIP_MODEL = None
LOADED = []


def _load_resnet():
    global RESNET_MODEL
    model = build_model("resnet50", pretrained=False)
    state = torch.load(RESNET_CKPT, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith("backbone") for k in state
    ):
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    RESNET_MODEL = model
    LOADED.append("ResNet-50")
    print("[startup] ResNet-50 loaded from", RESNET_CKPT)


def _try_load_clip():
    global CLIP_MODEL
    start = time.time()
    try:
        import importlib
        if importlib.util.find_spec("open_clip") is None:
            print("[startup] open_clip not available -> ResNet-only.")
            return
        from models.clip_head import build_clip_detector
        model = build_clip_detector(CLIP_CONFIG, CLIP_CKPT, device=DEVICE)
        model.eval()
        CLIP_MODEL = model
        LOADED.append("CLIP")
        print(f"[startup] CLIP loaded in {time.time() - start:.1f}s.")
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] CLIP unavailable ({type(exc).__name__}: {exc}). "
              "Falling back to ResNet-only.")


_load_resnet()
_try_load_clip()
if not LOADED:
    raise RuntimeError("No model could be loaded; cannot start demo.")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def _score_with(model, tensor) -> float:
    logit = model(tensor).reshape(-1)[0]
    return float(torch.sigmoid(logit).item())


def _confidence_label(ai_score: float) -> str:
    dist = abs(ai_score - 0.5)
    return "Low" if dist < 0.15 else ("Medium" if dist < 0.30 else "High")


# Brand palette
COL_AI = "#f2704d"     # coral — likely AI
COL_REAL = "#3fbf86"   # green — likely real
COL_MID = "#e0a13c"    # amber — uncertain
COL_MUTE = "#5b6b86"


def _result_html(ai_score: float | None, detail: str = "") -> str:
    if ai_score is None:
        return f"""
        <div class="veil-result">
          <div class="veil-rlabel" style="color:{COL_MUTE}">AWAITING IMAGE</div>
          <div class="veil-big" style="color:#3a4456">&mdash;</div>
          <div class="veil-bar"><div class="veil-bar-fill" style="width:0%;background:{COL_MUTE}"></div></div>
          <div class="veil-conf">Confidence Score</div>
        </div>"""
    pct = ai_score * 100.0
    if ai_score > 0.65:
        label, color, big, verdict = "LIKELY AI", COL_AI, pct, "High AI / scam risk"
    elif ai_score < 0.35:
        label, color, big, verdict = "LIKELY REAL", COL_REAL, 100 - pct, "Looks authentic"
    else:
        label, color, big, verdict = "UNCERTAIN", COL_MID, max(pct, 100 - pct), "Needs review"
    conf = _confidence_label(ai_score)
    return f"""
    <div class="veil-result">
      <div class="veil-rlabel" style="color:{color}">{label}</div>
      <div class="veil-big" style="color:{color}">{big:.0f}%</div>
      <div class="veil-bar"><div class="veil-bar-fill" style="width:{big:.0f}%;background:{color}"></div></div>
      <div class="veil-conf">Confidence Score &middot; {conf}</div>
      <div class="veil-verdict" style="color:{color}">{verdict}</div>
      <div class="veil-detail">{detail}</div>
    </div>"""


def predict(image: Image.Image):
    if image is None:
        return _result_html(None)
    img = image.convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0).to(DEVICE)
    ai_score = _score_with(RESNET_MODEL, tensor)
    parts = [f"ResNet {ai_score * 100:.0f}%"]
    if CLIP_MODEL is not None:
        try:
            clip_score = _score_with(CLIP_MODEL, tensor)
            parts.append(f"CLIP {clip_score * 100:.0f}%")
            ai_score = (ai_score + clip_score) / 2.0
        except Exception:  # noqa: BLE001
            pass
    detail = " &middot; ".join(parts) + f" &middot; {'+'.join(LOADED)}"
    return _result_html(ai_score, detail)


# ---------------------------------------------------------------------------
# Branding assets
# ---------------------------------------------------------------------------
V_LOGO = """
<svg width="64" height="64" viewBox="0 0 100 100" fill="none">
  <defs>
    <linearGradient id="vg" x1="10" y1="10" x2="90" y2="90" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#f4f7fc"/><stop offset="0.45" stop-color="#aab6cc"/>
      <stop offset="1" stop-color="#4a566e"/>
    </linearGradient>
    <linearGradient id="vg2" x1="50" y1="10" x2="50" y2="90" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#dfe6f2"/><stop offset="1" stop-color="#7c8aa6"/>
    </linearGradient>
  </defs>
  <path d="M14 16 L50 88 L60 88 L30 16 Z" fill="url(#vg)"/>
  <path d="M86 16 L50 88 L60 88 L86 34 Z" fill="url(#vg2)" opacity="0.92"/>
  <path d="M50 88 L44 76 L56 76 Z" fill="#eef2f8" opacity="0.5"/>
</svg>"""

HEADER = f"""
<div class="veil-header">
  {V_LOGO}
  <div class="veil-word">VEIL</div>
  <div class="veil-sub">AI&nbsp;&nbsp;CONTENT&nbsp;&nbsp;DETECTOR</div>
  <div class="veil-tag">See the <span class="hl">truth.</span> Behind the <span class="hl">image.</span></div>
</div>"""

_FEAT = [
    ("M12 3l7 3v5c0 4-3 7-7 8-4-1-7-4-7-8V6z", "ACCURATE<br>DETECTION"),
    ("M13 2L4 14h6l-1 8 9-12h-6z", "INSTANT<br>RESULTS"),
    ("M6 10V8a6 6 0 1112 0v2h1v11H5V10zm3 0h6V8a3 3 0 10-6 0z", "PRIVATE &amp;<br>SECURE"),
    ("M4 20V10h3v10zm6 0V4h3v16zm6 0v-7h3v7z", "BUILT FOR<br>EVERYONE"),
]
FEATURES = '<div class="veil-features">' + "".join(
    f'<div class="veil-feat"><svg width="22" height="22" viewBox="0 0 24 24" '
    f'fill="none" stroke="#8fa3c4" stroke-width="1.5"><path d="{d}"/></svg>'
    f'<div>{lbl}</div></div>' for d, lbl in _FEAT
) + '</div><div class="veil-foot">veil.app</div>'

CSS = """
.gradio-container {max-width: 860px !important; margin: auto !important;
  background: radial-gradient(900px 600px at 50% -8%, #141a2b 0%, #080b13 55%, #05070d 100%) !important;}
body, .gradio-container {color: #c9d3e3;}
footer {display:none !important;}
.veil-header {text-align:center; padding: 26px 0 4px;}
.veil-word {font-size: 50px; letter-spacing: 20px; font-weight: 300; color:#fff;
  margin: 6px 0 0 20px; font-family: Georgia, 'Times New Roman', serif;}
.veil-sub {font-size: 12px; letter-spacing: 6px; color:#8093b3; margin-top: 8px;}
.veil-tag {font-size: 28px; font-weight: 300; color:#c3cde0; margin: 30px 0 6px;}
.veil-tag .hl {color:#8ba6ff;}
.veil-card {background: rgba(20,26,40,0.55); border:1px solid rgba(120,140,180,0.18);
  border-radius: 18px; padding: 10px; backdrop-filter: blur(8px);}
.veil-result {padding: 26px 30px; min-height: 300px; display:flex; flex-direction:column;
  justify-content:center;}
.veil-rlabel {font-size: 13px; letter-spacing: 3px; font-weight:600; margin-bottom: 4px;}
.veil-big {font-size: 84px; font-weight: 300; line-height: 1; margin: 2px 0 18px;}
.veil-bar {height: 8px; border-radius: 6px; background: rgba(255,255,255,0.08); overflow:hidden;}
.veil-bar-fill {height: 100%; border-radius: 6px; transition: width .5s ease;}
.veil-conf {font-size: 13px; color:#8093b3; margin-top: 12px; letter-spacing: 1px;}
.veil-verdict {font-size: 20px; font-weight: 500; margin-top: 14px;}
.veil-detail {font-size: 11px; color:#5b6b86; margin-top: 10px;}
.veil-features {display:flex; justify-content:center; gap: 40px; padding: 26px 0 6px;
  border-top: 1px solid rgba(120,140,180,0.14); margin-top: 26px;}
.veil-feat {text-align:center; font-size: 10px; letter-spacing: 2px; color:#8093b3; line-height:1.5;}
.veil-feat svg {margin-bottom: 8px;}
.veil-foot {text-align:center; color:#5b6b86; font-size: 13px; padding: 8px 0 16px; letter-spacing:1px;}
#veil_go {background: linear-gradient(90deg,#f2704d,#e85c3a) !important; border:none !important;
  color:#fff !important; font-weight:600 !important; letter-spacing:2px !important;}
"""


def build_ui():
    with gr.Blocks(title="Veil — AI Content Detector",
                   theme=gr.themes.Base(), css=CSS) as demo:
        gr.HTML(HEADER)
        with gr.Row(elem_classes="veil-card", equal_height=True):
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label=None, height=300,
                                    show_label=False)
                analyze_btn = gr.Button("ANALYZE", elem_id="veil_go", size="lg")
            with gr.Column(scale=1):
                result = gr.HTML(_result_html(None))
        gr.HTML(FEATURES)
        analyze_btn.click(fn=predict, inputs=image_in, outputs=result)
        image_in.upload(fn=predict, inputs=image_in, outputs=result)
    return demo


if __name__ == "__main__":
    print(f"[startup] Device: {DEVICE} · Models: {', '.join(LOADED)}")
    demo = build_ui()
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860)
    except ValueError as exc:
        print(f"[startup] Localhost launch failed ({exc}); retrying with share.")
        demo.launch(share=True, server_port=7860)

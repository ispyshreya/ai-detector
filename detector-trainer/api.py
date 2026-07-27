import argparse
import re
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from train import build_model


IMAGE_SIZE = 224
THRESHOLD = 0.5
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_REVISION = None

state = {
    "model": None,
    "device": None,
    "checkpoint": None,
    "vlm_model": None,
    "vlm_processor": None,
    "vlm_device": None,
}


def build_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


eval_transform = build_eval_transform(IMAGE_SIZE)


def load_detector(checkpoint: Path, model_name: str):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = Path(app.state.checkpoint)
    model, device = load_detector(checkpoint, app.state.model_name)
    state["model"] = model
    state["device"] = device
    state["checkpoint"] = str(checkpoint)
    yield


app = FastAPI(title="Veil Local AI Detector", lifespan=lifespan)
app.state.checkpoint = "detector-trainer/output/best_model.pt"
app.state.model_name = "resnet50"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_vlm():
    if state["vlm_model"] is not None and state["vlm_processor"] is not None:
        return state["vlm_model"], state["vlm_processor"], state["vlm_device"]

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = {"": "cuda"} if device.type == "cuda" else {"": "cpu"}
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    model_kwargs = {
        "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
        "device_map": device_map,
    }
    if VLM_REVISION:
        model_kwargs["revision"] = VLM_REVISION
    model = AutoModelForImageTextToText.from_pretrained(VLM_MODEL_ID, **model_kwargs)
    model.eval()

    state["vlm_model"] = model
    state["vlm_processor"] = processor
    state["vlm_device"] = device
    return model, processor, device


def build_explanation_prompt(veil_score: float | None):
    score_text = f"Veil's authenticity risk score is {veil_score:.1%}." if veil_score is not None else ""
    if veil_score is None:
        verdict_instruction = (
            "Explain whether the image looks authentic or suspicious based only on visible evidence. "
            "If the image looks normal, say that no obvious visible AI artifacts were found."
        )
    elif veil_score >= 0.7:
        verdict_instruction = (
            "The system rated this image as high risk. Explain why it may be fake only if you can point to specific visible artifacts. "
            "If you cannot see clear artifacts, say the score is high but no specific visible AI artifacts were found."
        )
    elif veil_score >= 0.4:
        verdict_instruction = (
            "The system rated this image as uncertain. Explain the mixed result by naming visible details that look normal and any details that deserve review. "
            "Do not claim the image is fake unless there is a specific visible artifact."
        )
    else:
        verdict_instruction = (
            "The system rated this image as low risk. Explain why it appears likely authentic based on visible evidence. "
            "Mention normal lighting, coherent objects, readable text, natural anatomy, or consistent reflections only if visible."
        )

    return (
        f"{score_text} You are reviewing the uploaded image itself for visible authenticity warnings. "
        f"{verdict_instruction} "
        "The word Veil is the product name, not an object in the image. Never describe a veil unless one is visibly present. "
        "Ignore app chrome, website text, product logos, or interface labels; they are not evidence that the image is real or fake. "
        "Write for a non-technical person worried about scams. "
        "Only list concrete details that are visibly present in this exact image. "
        "Do not infer problems from missing objects, missing people, absent hands, absent faces, or unclear anatomy if those things are not visible. "
        "Do not use generic claims such as blurry background, indistinct area, unusual clarity, or image may be altered unless you name the specific object or region. "
        "Check visible faces, hands/fingers, text/logos, lighting/shadows, reflections, repeated patterns, and object edges. "
        "Return 1 to 3 short bullets. Do not use markdown headings like '**Visible Text**'. Each bullet must name a specific visible object or region and explain why it supports the verdict. "
        "If the image appears authentic, explain why it looks normal; do not invent fake evidence. "
        "If you cannot confidently explain the verdict from visible details, return exactly: '- No clear visual artifacts were found; verify the source before trusting it.'"
    )


def normalize_generated_text(text: str):
    cleaned = text.strip()
    for marker in ["Assistant:", "assistant\n", "<end_of_utterance>"]:
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def strip_bullet_markup(line: str):
    cleaned = line.strip()
    cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"^\*\*([^*]+)\*\*\s*:?\s*", "", cleaned)
    cleaned = re.sub(r"^([^:]{2,28}):\s+", "", cleaned)
    return cleaned.strip()


def fallback_visual_explanation(veil_score: float | None):
    if veil_score is not None and veil_score >= 0.7:
        return (
            "- No specific visible AI artifacts were found by the visual explainer.\n"
            "- The authenticity score is still high, so verify the image source before trusting it.\n"
            "- For scam checks, ask for independent proof instead of relying on this image alone."
        )
    if veil_score is not None and veil_score >= 0.4:
        return (
            "- The result is uncertain, and the visual explainer did not find one clear defect.\n"
            "- Review the image source, context, sender, and any request attached to it.\n"
            "- Be cautious if the image is being used for money, identity, dating, news, or urgency."
        )
    return (
        "- No obvious visible AI artifacts were found by the visual explainer.\n"
        "- The image appears lower risk, but source and context still matter.\n"
        "- For high-stakes decisions, verify through another trusted channel."
    )


def is_bad_explanation(text: str):
    cleaned = text.strip().lower().strip(".!?:;")
    if cleaned in {"", "no", "none", "nothing", "n/a", "not sure", "i don't know"}:
        return True
    lines = [strip_bullet_markup(line) for line in text.splitlines() if strip_bullet_markup(line)]
    return len(cleaned) < 24 or len(lines) == 0


def is_generic_or_hallucinated_line(line: str):
    lower = line.lower()
    blocked_phrases = [
        "word \"veil\"",
        "word veil",
        "the word \"veil\"",
        "the word veil",
        "text \"veil\"",
        "text veil",
        "the veil",
        "edges of the veil",
        "surface of the object",
        "the object",
        "of the object",
        "shape and structure",
        "look natural and well-formed",
        "without any signs of distortion",
        "normal lighting conditions",
        "no clear facial features",
        "no facial features",
        "no clear hands",
        "no hands",
        "no fingers",
        "align with human anatomy",
        "background appears to be a blurred",
        "blurred, indistinct area",
        "indistinct area",
        "unusual for a real photograph",
        "image might have been digitally altered",
        "potential ai generation",
    ]
    if any(phrase in lower for phrase in blocked_phrases):
        return True

    absence_phrases = [
        "there are no",
        "there is no",
        "not visible",
        "cannot see",
        "no clear",
        "lack of",
        "missing",
    ]
    if any(phrase in lower for phrase in absence_phrases):
        return True

    return False


def keep_warning_lines(text: str):
    reassuring_phrases = ["professional camera", "high-quality photography", "could indicate it was taken"]
    lines = [strip_bullet_markup(line) for line in text.splitlines() if strip_bullet_markup(line)]

    kept = []
    for line in lines:
        lower = line.lower()
        if any(phrase in lower for phrase in reassuring_phrases):
            continue
        if is_generic_or_hallucinated_line(line):
            continue
        kept.append(line)

    return "\n".join(kept).strip()


@app.get("/health")
def health():
    return {
        "status": "ok" if state["model"] is not None else "loading",
        "device": str(state["device"]),
        "checkpoint": state["checkpoint"],
        "positive_class": "FAKE",
        "score_meaning": "probability_fake_or_ai_generated",
        "visual_explanation_model": VLM_MODEL_ID,
        "visual_explanation_revision": VLM_REVISION,
    }


@app.post("/predict")
async def predict(media: UploadFile = File(...)):
    if media.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPG, PNG, or WEBP.",
        )

    contents = await media.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    model = state["model"]
    device = state["device"]
    if model is None or device is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    tensor = eval_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        fake_probability = torch.sigmoid(model(tensor)).item()

    label = "FAKE" if fake_probability >= THRESHOLD else "REAL"
    confidence = fake_probability if label == "FAKE" else 1.0 - fake_probability

    return {
        "genai": fake_probability,
        "deepfake": None,
        "label": label,
        "confidence": confidence,
        "threshold": THRESHOLD,
        "model": "resnet50",
        "checkpoint": state["checkpoint"],
        "score_meaning": "probability_fake_or_ai_generated",
    }


@app.post("/explain")
async def explain(media: UploadFile = File(...), veil_score: float | None = Form(None)):
    if media.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPG, PNG, or WEBP.",
        )

    contents = await media.read()
    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        model, processor, device = load_vlm()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Visual explanation model failed to load: {exc}") from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_explanation_prompt(veil_score)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
        )
    generated_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    explanation = keep_warning_lines(normalize_generated_text(generated_text))
    used_fallback = is_bad_explanation(explanation)

    return {
        "explanation": fallback_visual_explanation(veil_score) if used_fallback else explanation,
        "model": VLM_MODEL_ID,
        "revision": VLM_REVISION,
        "used_fallback": used_fallback,
        "note": (
            "The local visual model did not produce a useful specific explanation, so Veil showed a cautious inspection checklist."
            if used_fallback
            else "Visual explanations are AI-generated and should be treated as possible warning signs, not proof."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the local AI detector API.")
    parser.add_argument("--checkpoint", default="detector-trainer/output/best_model.pt")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app.state.checkpoint = args.checkpoint
    app.state.model_name = args.model

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

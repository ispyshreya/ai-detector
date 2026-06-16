from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.services.pipeline import ForensicsPipeline


settings = get_settings()
settings.artifact_dir.mkdir(parents=True, exist_ok=True)
pipeline = ForensicsPipeline(settings)

app = FastAPI(
    title="Veil Forensics Backend",
    version="0.1.0",
    description="Scaffold for EXIF, ELA, reverse-search, and C2PA modules.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts", StaticFiles(directory=settings.artifact_dir), name="artifacts")


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "veil-forensics-backend",
        "artifact_dir": str(settings.artifact_dir),
        "serpapi_configured": bool(settings.serpapi_key),
    }


@app.post("/analyze")
async def analyze(media: UploadFile = File(...)) -> dict[str, object]:
    if not media.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")

    media_bytes = await media.read()
    if not media_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    response = await pipeline.analyze(
        filename=Path(media.filename).name,
        content_type=media.content_type,
        media_bytes=media_bytes,
    )
    return response.model_dump(mode="json")

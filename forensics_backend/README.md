# Veil Forensics Backend

This folder scaffolds Marko's proposal-owned backend scope:

- EXIF and metadata anomaly extraction
- Error Level Analysis heatmap generation
- Reverse image search integration boundary
- C2PA verification integration boundary

The service is intentionally isolated from the frontend so the backend and orchestration work can evolve without coupling to `app.jsx`.

## Endpoints

- `GET /health`
- `POST /analyze`
- `GET /artifacts/{analysis_id}/{filename}`

## Response Contract

`POST /analyze` returns a single JSON envelope with one inspectable object per module:

```json
{
  "analysis_id": "c0ffee1234",
  "filename": "example.jpg",
  "sha256": "abc123...",
  "forensics": {
    "exif": {
      "status": "ok",
      "anomaly_flags": ["missing_camera_make"]
    },
    "ela": {
      "status": "ok",
      "heatmap_url": "/artifacts/c0ffee1234/ela_heatmap.png"
    },
    "reverse_search": {
      "status": "skipped",
      "summary": "SerpAPI key not configured."
    },
    "c2pa": {
      "status": "skipped",
      "summary": "C2PA verification tool not configured."
    }
  }
}
```

This is the contract the triangulation layer can consume later.

## Local Run

Create a virtual environment, install requirements, and run:

```bash
cd forensics_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

## Notes

- `EXIF` and `ELA` are implemented locally now.
- `Reverse search` and `C2PA` are scaffolded with explicit placeholders so the team can wire real provider/tooling choices later.
- `ELA` artifacts are written under `runtime/artifacts/<analysis_id>/`.

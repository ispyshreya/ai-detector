# Veil - AI Image Authenticity Checker

Veil helps users check whether an image may be AI-generated, edited, or risky to trust. It presents one plain-language Veil score, a confidence rating, scam-aware next steps, and optional visual warning-sign explanations.

## Features

- Image upload for JPG, JPEG, PNG, and WEBP files.
- Single user-facing Veil authenticity score.
- Confidence score based on detector agreement and signal strength.
- Plain-language verdicts: `Looks authentic`, `Needs review`, and `High scam risk`.
- Scam-focused guidance for images used in messages, profiles, posts, or urgent requests.
- History page for recent scans.
- Settings page for local endpoint configuration.
- Optional local visual explanation endpoint for image-specific warning signs.

## Architecture

The React frontend calls two local API routes:

- `POST /predict` returns the local detector score.
- `POST /explain` returns optional visual explanation text.

The app can also call Sightengine from the browser when credentials are configured. Veil combines the detector outputs into one score for the user.

## Environment Variables

Copy `.env.example` to `.env`:

```powershell
copy .env.example .env
```

Configure:

```env
VITE_SIGHTENGINE_API_USER="your_sightengine_user"
VITE_SIGHTENGINE_API_SECRET="your_sightengine_secret"
VITE_CUSTOM_API_URL="http://127.0.0.1:8000/predict"
VITE_EXPLANATION_API_URL="http://127.0.0.1:8000/explain"
VITE_CUSTOM_API_KEY=""
```

## Run the Frontend

Install dependencies and start the Vite development server:

```powershell
npm install
npm run dev
```

The app is served at `http://127.0.0.1:5173/` by default.

## Local Detector API

This repo expects a local model API running on port `8000`. The trainer/API code and model files can be kept local and are not required for the frontend repository to build.

Expected endpoints:

```text
GET  /health
POST /predict
POST /explain
```

`/predict` should return a JSON response with a `genai` score from `0` to `1`.

`/explain` should return:

```json
{
  "explanation": "Short visual warning-sign explanation",
  "note": "Visual explanations are AI-generated and should be treated as possible warning signs, not proof."
}
```

## Detector Trainer

The companion `detector-trainer/` workspace is used locally to:

- split image data into train/validation folders
- train the fake-vs-real ResNet detector
- evaluate the saved checkpoint on a held-out test set
- run the FastAPI model server that exposes `/predict` and `/explain`

The frontend can still build without this folder as long as compatible API endpoints are running.

## Build

```powershell
npm run build
```

Preview the production build:

```powershell
npm run preview
```

## Security Notes

- Do not commit `.env`.
- Rotate any credentials that were accidentally committed.
- Do not expose the local API directly to the public internet without authentication or a private tunnel.
- Visual explanations are assistive, not proof. Veil should support human review rather than replace it.

## Tech Stack

- React 18
- Vite 5
- Plain CSS
- Local REST API integration

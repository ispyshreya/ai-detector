# Veil

Veil is an image-authenticity dashboard for evaluating whether an uploaded image may be AI-generated, manipulated, or risky to trust. It combines detector outputs into a single user-facing score and presents the result in plain language for authenticity review, scam prevention, and media verification workflows.

## Overview

Modern image scams often rely on synthetic profile photos, edited screenshots, fabricated product images, or AI-generated evidence. Veil is designed to make those risks easier to assess by turning model signals into a simple review experience:

- a single Veil authenticity score
- a confidence rating
- a plain-language risk verdict
- visual warning-sign explanations
- practical next steps for verification

The application is intended to support human review. It should not be treated as definitive proof that an image is real or fake.

## Key Features

- **Image authenticity scoring**: Upload an image and receive a normalized Veil score.
- **Plain-language verdicts**: Results are presented as `Looks authentic`, `Needs review`, or `High scam risk`.
- **Confidence rating**: Veil estimates how strongly to trust the result based on signal strength and detector agreement.
- **Visual warning signs**: Optional local vision-language analysis explains why an image may have been flagged.
- **Scam-aware guidance**: Results include verification steps for images tied to money, identity, dating, news, or urgent requests.
- **Scan history**: Recent scans are recorded in the session for quick comparison.
- **Local-first architecture**: The frontend can connect to a local detector API for private model inference.

## Architecture

Veil is a Vite + React frontend that communicates with detector services through HTTP endpoints.

```text
User upload
   |
   v
React dashboard
   |
   |-- Local detector API: /predict
   |-- Local visual explanation API: /explain
   |-- Optional external detector: Sightengine
   |
   v
Veil score + confidence + explanation
```

The frontend expects compatible API endpoints but does not require model files to be committed to the repository.

## Repository Structure

```text
.
├── app.jsx              # Main React application
├── main.jsx             # React entry point
├── index.html           # Vite HTML shell
├── styles.css           # Application styles
├── vite.config.js       # Vite configuration
├── package.json         # Frontend scripts and dependencies
├── .env.example         # Environment variable template
└── detector-trainer/    # Local training/API workspace, kept local if desired
```

## Configuration

Create a local `.env` file from the example:

```powershell
copy .env.example .env
```

Configure the detector endpoints and optional Sightengine credentials:

```env
VITE_CUSTOM_API_URL="http://127.0.0.1:8000/predict"
VITE_EXPLANATION_API_URL="http://127.0.0.1:8000/explain"
VITE_CUSTOM_API_KEY=""

VITE_SIGHTENGINE_API_USER="your_sightengine_user"
VITE_SIGHTENGINE_API_SECRET="your_sightengine_secret"
```

## Development

Install dependencies:

```powershell
npm install
```

Start the development server:

```powershell
npm run dev
```

By default, Vite serves the app at:

```text
http://127.0.0.1:5173/
```

Create a production build:

```powershell
npm run build
```

Preview the production build:

```powershell
npm run preview
```

## Local API Contract

Veil expects a local API server on port `8000` by default.

### `GET /health`

Returns API status and model metadata.

### `POST /predict`

Accepts a multipart image upload with the field name `media`.

Expected response:

```json
{
  "genai": 0.87,
  "label": "FAKE",
  "confidence": 0.87,
  "score_meaning": "probability_fake_or_ai_generated"
}
```

### `POST /explain`

Accepts a multipart image upload with the field name `media` and an optional `veil_score` field.

Expected response:

```json
{
  "explanation": "- Possible warning sign...",
  "note": "Visual explanations are AI-generated and should be treated as possible warning signs, not proof."
}
```

## Detector Trainer

The optional `detector-trainer/` workspace supports the local model pipeline:

- splitting training data into train/validation folders
- fine-tuning a fake-vs-real image classifier
- evaluating the saved checkpoint on a held-out test set
- serving the trained detector through FastAPI
- generating visual explanations with a local vision-language model

The frontend can still be developed and built without committing this folder, as long as compatible API endpoints are available.

## Security

- Do not commit `.env` or real API credentials.
- Rotate credentials if they were ever committed or shared.
- Do not expose the local detector API directly to the public internet without authentication.
- Use a private tunnel or VPN-style tool for remote demos.
- Treat Veil output as decision support, not definitive proof.

## Technology

- React 18
- Vite 5
- CSS Grid/Flexbox
- Local REST API integration
- Optional Sightengine integration
- Optional local vision-language explanation model

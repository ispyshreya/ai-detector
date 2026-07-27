# Detector Trainer

Standalone project for training an AI-generated image detector.

## Structure

- `train.py` - train a binary image classifier using transfer learning.
- `split_data.py` - split an existing `data/train` folder into `data/train` + `data/val`.
- `requirements.txt` - Python dependencies.

## Recommended dataset layout

If you already have `train` and `test` folders, keep `test` untouched and split part of `train` into validation.

Example directory structure:

```
data/train/real/
data/train/ai_generated/
data/val/real/
data/val/ai_generated/
data/test/real/
data/test/ai_generated/
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Split train into validation

```bash
python split_data.py --source data/train --target data --val-fraction 0.1
```

## Train the detector

```bash
python train.py --data-dir data --output-dir output --epochs 10 --batch-size 32
```

## Test the detector

```bash
python test.py --test-dir data/test --checkpoint output/best_model.pt --output-json output/test_metrics.json
```

## Run the local detector API

Start the API:

```bash
python api.py --checkpoint output/best_model.pt --host 127.0.0.1 --port 8000
```

The same API also exposes `/explain`, which uses `HuggingFaceTB/SmolVLM-500M-Instruct`
locally to generate cautious visual warning signs for the uploaded image.

Then point the React app to:

```env
VITE_CUSTOM_API_URL="http://127.0.0.1:8000/predict"
VITE_EXPLANATION_API_URL="http://127.0.0.1:8000/explain"
VITE_CUSTOM_API_KEY=""
```

## Notes

- `train.py` uses a pretrained `resnet50` model and binary cross-entropy loss.
- If `data/val` is absent, `train.py` will split `data/train` automatically.
- `data/test` should remain unchanged for final evaluation.

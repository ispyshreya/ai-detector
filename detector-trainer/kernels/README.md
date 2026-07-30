# Veil Detector — Kaggle Training Loop

The agent-driven cycle from the design spec (§7): push the code + a kernel to
Kaggle's free GPU, run all pipeline stages, pull the results back.

```
edit code  →  push code (dataset)  →  push kernel  →  pull outputs  →  read metrics
```

## Files here

| file | purpose |
|---|---|
| `kernel-metadata.json` | metadata for `kaggle kernels push` (GPU + internet + the two training datasets) |
| `veil-detector-train.ipynb` | the notebook: installs deps, gets the code, runs `run_pipeline.py --stages all`, writes to `/kaggle/working` |
| `README.md` | this file |

## One-time setup

1. **Kaggle credentials** — `~/.kaggle/kaggle.json` (already configured per spec §7).
2. **Fill in the kernel id.** Edit `kernel-metadata.json` and replace
   `<KAGGLE_USERNAME>` in `"id"` with your Kaggle username, e.g.
   `"id": "sarthakhans/veil-detector-train"`.
3. **Choose how the code reaches Kaggle** (see below) and edit the config cell
   at the top of the notebook accordingly.

## Getting the code onto Kaggle

The kernel needs the `detector-trainer/` tree importable. Two options:

### Option A — git clone (default in the notebook)
`enable_internet` is already `true`. In the notebook's config cell set:
```python
USE_GIT  = True
REPO_URL = 'https://github.com/<YOUR_GH_USER>/ai-detector-repo.git'
BRANCH   = 'detector-model'
```
Nothing else to push — the kernel clones at run time. (Private repos need a token
in the URL; prefer Option B for private code.)

### Option B — attach the code as a Kaggle utility dataset
Push `detector-trainer/` as a Dataset once, then version it on each change:
```bash
# from the repo root, first time:
kaggle datasets init -p detector-trainer
#   -> edit detector-trainer/dataset-metadata.json: set "id" to
#      "<KAGGLE_USERNAME>/veil-detector-code" and a title
kaggle datasets create -p detector-trainer --dir-mode zip

# on every subsequent code change:
kaggle datasets version -p detector-trainer -m "update pipeline" --dir-mode zip
```
Then in `kernel-metadata.json` add the code dataset to `dataset_sources`:
```json
"dataset_sources": [
  "cartografia/unbiased-tiny-genimage",
  "rhythmghai/ai-vs-real-images-dataset",
  "<KAGGLE_USERNAME>/veil-detector-code"
]
```
and in the notebook config cell set:
```python
USE_GIT  = False
CODE_DIR = '/kaggle/input/veil-detector-code/detector-trainer'
```

### Diverse real-world photos dataset (fixes real-photo false positives)
The training reals + held-out phone-capture proxy live in a **separate** dataset
(`veil-detector-reals`, ~939 MB) so they are not re-uploaded on every code change.
Acquire and push them:
```bash
cd detector-trainer
python3 acquire_reals.py --out real_world \
    --coco 2000 --ffhq 1000 --unsplash 1000 --div2k 100
# real_world/dataset-metadata.json already sets id -> <USER>/veil-detector-reals
kaggle datasets create -p real_world --dir-mode zip        # first time
kaggle datasets version -p real_world -m "refresh reals" --dir-mode zip  # updates
```
`kernel-metadata.json` already lists `sarthakhans01/veil-detector-reals` in
`dataset_sources`, and the notebook passes each `real_*` folder as its own
`--data-root` (so `source` == folder name) with
`--wild-real-sources real_div2k_heldout` to hold the camera-native proxy out of
training. **Do not** push `real_world/` inside the code dataset — keep it separate.

### Modern-generator fakes dataset (flux/dalle3)
The self-generated Flux + DALL·E images (`wild_data/{flux,dalle3}`) push as a
separate `veil-detector-wild-fakes` dataset:
```bash
cd detector-trainer
# wild_data/dataset-metadata.json sets id -> <USER>/veil-detector-wild-fakes
kaggle datasets create -p wild_data --dir-mode zip
```
By default `WILD_GENERATORS` holds midjourney/dalle3/flux out of training. To
*train* on flux/dalle3 (closing the modern-generator gap) while keeping
Midjourney as the honest unseen test, the notebook passes
`--wild-generators midjourney`. The kernel already lists the dataset in
`dataset_sources` and the notebook adds it to `--data-root`.

## Push the kernel and run

```bash
# push the notebook + metadata (must be run from the kernels/ dir, or pass -p)
kaggle kernels push -p detector-trainer/kernels

# check status until it finishes (queued -> running -> complete/error)
kaggle kernels status <KAGGLE_USERNAME>/veil-detector-train
```

## Pull the outputs back

```bash
kaggle kernels output <KAGGLE_USERNAME>/veil-detector-train -p ./kaggle_out
```
This retrieves everything the pipeline wrote under `/kaggle/working/veil_run`:

- `report/report.md` + `report/plots/*.png` (ROC curves, per-generator AUROC)
- `report/predictions_<model>.csv`
- `resnet50/resnet50_best.pt` + `resnet50/results.json`
- `clip_linear/` and `clip_mlp/` (`clip_head_best.pt` + `config.json`)
- `clip_features/features_*.npy` (cached once — reuse across head experiments)
- `pipeline_summary.json`

Read `report.md`, decide the next iteration, and drop the winning checkpoint into
`backend/` per spec §6.

## Local smoke test (no Kaggle, no GPU, no downloads)

Before pushing, verify the orchestration end-to-end on synthetic data:
```bash
cd detector-trainer
python3 run_pipeline.py --smoke --out runs/smoke
```
This generates tiny noise images across real + all generators (including the
held-out wild MJ/dalle3/flux), builds a manifest, trains ResNet (1 epoch) and both
CLIP heads (stubbed backbone — no weight download), and writes the full report.

## Running individual stages

`--stages` accepts any subset, in order:
```bash
python3 run_pipeline.py --stages manifest --data-root /kaggle/input/... --out runs/veil
python3 run_pipeline.py --stages train_resnet train_clip --out runs/veil
python3 run_pipeline.py --stages evaluate --out runs/veil
```
Stages after `manifest` read the manifest at `--manifest` and write into `--out`.
```

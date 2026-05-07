# Petalawati — Startup Guide

## Prerequisites

- Python virtual environment at `.venv/` (already created)
- All dependencies installed via `requirements.txt`
- Prefect server accessible at `http://127.0.0.1:4200`

---

## First-Time Setup

Run these **once** before your first launch.

```bash
cd /path/to/petalawati

# 1. Create the Prefect work pool
.venv/bin/prefect work-pool create cnn-pool --type process

# 2. Deploy the training flow to the pool
PYTHONPATH=. .venv/bin/prefect deploy --prefect-file prefect.yaml --all
```

---

## Starting the Services

Open **4 separate terminals**, all from the project root.

### Terminal 1 — Prefect Server

```bash
.venv/bin/prefect server start
```

Wait until you see:

```
Prefect UI available at http://127.0.0.1:4200
```

### Terminal 2 — Prefect Worker

Picks up and executes training jobs submitted by the web app.

```bash
.venv/bin/prefect worker start --pool cnn-pool
```

### Terminal 3 — FastAPI Web Server

```bash
PYTHONPATH=. .venv/bin/uvicorn app.main:app --reload --port 8000
```

### Terminal 4 — (Optional) Monitor Logs

Stream worker output or app logs as needed.

---

## Access

| Service | URL |
|---|---|
| Web App | http://localhost:8000 |
| Prefect UI | http://localhost:4200 |

---

## Startup Order

```
Prefect Server  →  Prefect Worker  →  FastAPI App
   (T1)                (T2)              (T3)
```

Always start the Prefect server first. The worker and app can start in any order after that.

---

## Re-deploy After Pipeline Changes

If you modify any file under `app/pipelines/`, re-run the deploy command before submitting new jobs:

```bash
PYTHONPATH=. .venv/bin/prefect deploy --prefect-file prefect.yaml --all
```

---

## Workflow

```
1. Open http://localhost:8000
2. Click "+ New Job"
3. Upload a ZIP file (class folders inside, e.g. cats/, dogs/)
4. Pick a model architecture and set hyperparameters
5. Click "Start Training"
6. Watch live loss/accuracy charts on the Training page
7. Download model.keras from the Results page when done
```

---

## Available Architectures

| Name | Description |
|---|---|
| **KernArc** | Multi-kernel attention gate — 3 parallel conv branches (2×, 4×, 8×) fused via sigmoid mask |
| **SwiftPan** | 3-stage multiply-add attention chain — ~300 preprocessing params, very fast |
| **SepFuse** | Separable-FPN — bottom-up SepConv2D blocks + top-down pyramid merge |
| **DualFuse** | Dual-stream CNN — full-res and half-res paths fused before classification head |
| **SyncGen** | Strided encoder — mirrors a supervised decoder, uses BN + LeakyReLU blocks |

All models train from scratch (no frozen pretrained weights).

---

## Dataset Format

Upload a `.zip` file containing one folder per class:

```
dataset.zip
├── cats/
│   ├── img001.jpg
│   └── img002.jpg
└── dogs/
    ├── img003.jpg
    └── img004.jpg
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

## Artifacts

Trained models and metrics are saved to:

```
data/artifacts/{job_id}/
├── model.keras       ← downloadable trained model
├── metrics.json      ← per-epoch loss and accuracy
├── confusion.json    ← confusion matrix (post-training)
└── job.json          ← job status and config
```

# Petalawati

A web-based CNN training platform. Upload a labelled image dataset, pick a model architecture, set hyperparameters, and get a trained `.keras` model — all from the browser.

Built with FastAPI, TensorFlow, and Prefect for async job orchestration.

---

## Architecture

```
Browser  →  FastAPI (port 8000)  →  Prefect Worker  →  TensorFlow training
                                         ↑
                               Prefect Server (port 4200)
```

Training jobs are submitted as Prefect flows and executed by a background worker, keeping the web server non-blocking.

---

## Models

All five architectures are original designs trained from scratch — no pretrained weights or frozen layers.

---

### KernArc ⭐ recommended default

Multi-kernel attention gate. Three parallel Conv2D branches with receptive fields of 2×, 4×, and 8× run in parallel, are concatenated and compressed, then a sigmoid gate attends back to the original input before the classifier head.

- Best for: high-resolution detail, general-purpose classification
- Key ops: parallel Conv2D → BN → sigmoid attention → Multiply

---

### SwiftPan

Lightweight 3-stage multiply-add attention chain. Each stage multiplies a small conv output with a pooled reference from the original input, then adds a large-kernel context branch. Only ~300 preprocessing parameters.

- Best for: CPU/edge inference, quick experiments, large datasets
- Key ops: Conv(3×3) × input + Conv(9×9), repeated 3× with MaxPool

---

### SepFuse

Separable feature pyramid network. A bottom-up encoder using `SeparableConv2D` + BN + MaxPool builds four spatial scales (H/2 → H/16), then a top-down pass merges them via lateral 1×1 projections and upsampling.

- Best for: multi-scale patterns, fine-grained texture recognition
- Key ops: SepConv2D bottom-up → UpSampling2D top-down → Add

---

### DualFuse

Dual-stream CNN. Two independent CNN streams process the image at full resolution (7×7, 5×5, 3×3 kernels) and half resolution (5×5, 3×3 kernels) separately, then their `GlobalAveragePooling` outputs are concatenated before the dense head.

- Best for: datasets with significant size or scale variance
- Key ops: two parallel cnn_stream → Concatenate → Dense head

---

### SyncGen

Strided encoder modelled after a supervised decoder. Progressive `Conv2D` with strides (2, 2, 4, 2) and `LeakyReLU` + BN blocks compress the image to a bottleneck, mirroring the transposed-conv structure of a generative decoder in reverse.

- Best for: structured or synthetic data, generated imagery
- Key ops: strided Conv2D + BN + LeakyReLU(0.2) → GAP → Dense

---

## Dataset Format

Upload a `.zip` containing one folder per class:

```
dataset.zip
├── cats/
│   ├── img001.jpg
│   └── img002.jpg
└── dogs/
    ├── img003.jpg
    └── img004.jpg
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

---

## Setup

**Requirements:** Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First-time Prefect setup (run once):

```bash
prefect work-pool create cnn-pool --type process
PYTHONPATH=. prefect deploy --prefect-file prefect.yaml --all
```

---

## Running

Start three services in separate terminals, in order:

```bash
# 1 — Prefect server
prefect server start

# 2 — Prefect worker (picks up training jobs)
prefect worker start --pool cnn-pool

# 3 — Web app
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

---

## Workflow

1. Click **+ New Job**
2. Upload a ZIP dataset
3. Choose a model and set hyperparameters (epochs, learning rate, image size, etc.)
4. Click **Start Training**
5. Watch live loss/accuracy charts on the Training page
6. Download `model.keras` from the Results page when done

---

## Artifacts

Each job saves its outputs to `data/artifacts/{job_id}/`:

```
model.keras       — trained model
metrics.json      — per-epoch loss and accuracy
confusion.json    — confusion matrix
job.json          — job status and config
```

---

## Stack

- [FastAPI](https://fastapi.tiangolo.com/) — web server and API
- [TensorFlow](https://tensorflow.org/) — model training
- [Prefect](https://prefect.io/) — async job orchestration
- [Jinja2](https://jinja.palletsprojects.com/) — HTML templates

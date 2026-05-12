# Petalawati

Petalawati is a FastAPI web app for training image classification models from zipped datasets. It provides a browser workflow for uploading an ImageFolder-style archive, configuring a model, submitting a background training job through Prefect, monitoring metrics live, and downloading the resulting `.keras` model.

## What The App Does

- Accepts a `.zip` dataset with one folder per class
- Validates archive structure and image file types at upload time
- Optionally checks image channel consistency before training
- Trains one of several TensorFlow architectures from scratch
- Streams epoch metrics to the UI from file-backed job artifacts
- Saves the trained model and confusion matrix per job
- Lets you cancel running jobs and delete finished or failed jobs

## Stack

- FastAPI for the web server and JSON APIs
- Jinja2 templates for the UI
- Prefect for asynchronous job orchestration
- TensorFlow / Keras for model training
- Pillow for archive image inspection
- scikit-learn for confusion matrix generation

## Project Layout

```text
app/
  main.py                    FastAPI app entrypoint
  config.py                  paths, Prefect URL, architecture catalogue
  job_store.py               file-backed job metadata and artifacts
  dataset_archive.py         ZIP validation and extraction helpers
  routes/                    HTML pages and API endpoints
  pipelines/
    training_flow.py         Prefect flow
    tasks/
      data_tasks.py          dataset extraction / TF dataset building
      model_tasks.py         model builders
      train_tasks.py         training, metrics logging, evaluation, save
  templates/                 dashboard, upload, configure, monitor, results
data/
  uploads/                   uploaded ZIPs and extracted datasets
  artifacts/                 per-job outputs
start_app.sh                 convenience launcher for local development
prefect.yaml                 Prefect deployment definition
```

## Training Flow

```text
Browser
  -> FastAPI
  -> /api/upload stores ZIP and creates a pending job
  -> /api/jobs submits a Prefect deployment
  -> Prefect worker runs app.pipelines.training_flow
  -> TensorFlow trains and writes metrics/artifacts
  -> UI polls /api/jobs/{job_id}/progress
```

The app keeps job state in `data/artifacts/{job_id}/job.json`. Metrics are appended to `metrics.json` after each epoch, and completed jobs also produce `confusion.json` and `model.keras`.

## Supported Architectures

The UI currently exposes these architecture keys from `app/config.py`:

- Custom: `kernarc`, `swiftpan`, `sepfuse`, `dualfuse`, `syncgen`, `nanonet`
- Canonical / compact implementations: `alexnet`, `vggnet`, `resnet`, `inceptionv3`, `efficientnet`, `vit`, `densenet`, `mobilenet`, `swin`, `convnext`

The default selection in the configuration screen is `resnet`. All models are built in-repo and trained from scratch.

## Dataset Requirements

Upload a `.zip` file containing one directory per class:

```text
dataset.zip
├── cats/
│   ├── img001.jpg
│   └── img002.jpg
└── dogs/
    ├── img003.jpg
    └── img004.jpg
```

Rules enforced by the current code:

- Files must be exactly one level below the class folder, such as `cats/img001.jpg`
- Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Hidden macOS archive entries such as `__MACOSX` and `._*` are ignored
- Empty or invalid ZIP files are rejected
- If enabled, the consistency check rejects unsupported channel modes and mixed grayscale/colour classes

## Setup

Requirements:

- Python 3.10+
- A local virtual environment at `.venv`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

### Recommended: one-command startup

`start_app.sh` will:

- verify `.venv`
- install missing packages from `requirements.txt`
- start or reuse the Prefect server
- create the `cnn-pool` work pool if missing
- deploy `prefect.yaml`
- start a Prefect worker
- start the FastAPI app on port `8000`

Run:

```bash
./start_app.sh
```

### Manual startup

If you want separate terminals:

```bash
# 1. Prefect server
.venv/bin/prefect server start

# 2. Prefect worker
.venv/bin/prefect worker start --pool cnn-pool

# 3. Deploy the flow
PYTHONPATH=. .venv/bin/prefect deploy --prefect-file prefect.yaml --all

# 4. Web app
PYTHONPATH=. .venv/bin/uvicorn app.main:app --reload --port 8000
```

Open:

- Web UI: `http://127.0.0.1:8000`
- Prefect UI: `http://127.0.0.1:4200`

## Environment

Optional environment variables:

- `DATA_DIR`: overrides the default `data/` directory root
- `PREFECT_API_URL`: defaults to `http://127.0.0.1:4200/api`

The app loads `.env` automatically via `python-dotenv`.

## User Workflow

1. Open `/upload`
2. Upload a dataset ZIP
3. Review detected classes and image count
4. Open `/configure/{job_id}`
5. Choose an architecture and hyperparameters
6. Submit the job
7. Monitor training on `/training/{job_id}`
8. View charts and confusion matrix on `/results/{job_id}`
9. Download the generated `.keras` model

## API Surface

Current routes implemented under `app/routes/`:

- `POST /api/upload` upload dataset ZIP and create a pending job
- `POST /api/jobs` submit a training job to Prefect
- `GET /api/jobs` list stored jobs
- `GET /api/jobs/{job_id}/progress` return status, metrics, classes, confusion matrix, error
- `POST /api/jobs/{job_id}/cancel` cancel a running Prefect flow
- `DELETE /api/jobs/{job_id}` delete non-running job files
- `GET /api/jobs/{job_id}/download` download the trained model

HTML pages:

- `/` dashboard
- `/upload`
- `/catalogue`
- `/configure/{job_id}`
- `/training/{job_id}`
- `/results/{job_id}`

## Job Artifacts

Each job writes to `data/artifacts/{job_id}/`:

```text
job.json         job metadata and status
metrics.json     per-epoch loss, accuracy, timing, ETA
confusion.json   confusion matrix after evaluation
model.keras      saved Keras model
```

Uploaded archives live at `data/uploads/{job_id}.zip` and extracted image folders at `data/uploads/{job_id}/`.

## Notes

- Job metadata is file-backed, not stored in a database
- The training monitor polls every 2 seconds
- Cancelling a running job marks it as failed with `Cancelled by user`
- Deleting a job removes its artifacts, extracted dataset, and uploaded ZIP

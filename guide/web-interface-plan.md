# Simple Web Interface Plan

## Goal

Build a simple web interface for the CNN training workflow using:

- FastAPI
- Jinja2
- Uvicorn
- FastAPI routers
- Tailwind CSS

The interface should let a user:

1. upload a dataset
2. configure a training job
3. monitor training progress
4. view results
5. download the trained model


## Approach

Keep the frontend server-rendered with Jinja2 and use small client-side `fetch()` calls only where needed. This keeps the stack simple, avoids a frontend build system, and fits well with FastAPI.

Use FastAPI for two responsibilities:

- render HTML pages
- expose JSON APIs for upload, job creation, progress polling, and artifact download


## High-Level Flow

1. User opens the dashboard.
2. User uploads a ZIP dataset.
3. Backend validates the ZIP and creates a `job_id`.
4. User configures model architecture and hyperparameters.
5. Backend stores the config and submits a Prefect training flow.
6. Worker runs training in the background.
7. UI polls progress from backend endpoints.
8. Results page shows metrics, confusion matrix, and download link.


## Project Structure

Recommended layout:

```text
app/
  main.py
  config.py
  schemas.py
  job_store.py
  routes/
    pages.py
    upload.py
    jobs.py
    artifacts.py
  templates/
    base.html
    index.html
    upload.html
    configure.html
    training.html
    results.html
  pipelines/
    training_flow.py
    tasks/

data/
  uploads/
  artifacts/
```


## Implementation Plan

### 1. App Bootstrap

Create `app/main.py` to:

- initialize `FastAPI`
- register all routers
- keep app startup minimal

Use `uvicorn app.main:app --reload` in development.


## 2. Configuration

Create `app/config.py` for shared constants and paths:

- `BASE_DIR`
- `DATA_DIR`
- `UPLOADS_DIR`
- `ARTIFACTS_DIR`
- `PREFECT_API_URL`
- architecture labels/options

This keeps path and environment logic out of routes.


## 3. Schemas

Create `app/schemas.py` with Pydantic models for:

- `JobConfig`
- `JobStatus`
- `EpochMetric`
- `ProgressResponse`

These models define the contract between backend routes and frontend requests.


## 4. Job Persistence

Create `app/job_store.py` as a simple file-based persistence layer.

Each job should live under:

```text
data/artifacts/{job_id}/
```

Expected files:

- `job.json`
- `metrics.json`
- `confusion.json`
- `model.keras`

This avoids adding a database while keeping job state explicit and inspectable.


## 5. Routers

Split routes by responsibility.

### `app/routes/pages.py`

HTML pages rendered with Jinja2:

- `/` dashboard
- `/upload`
- `/configure/{job_id}`
- `/training/{job_id}`
- `/results/{job_id}`

### `app/routes/upload.py`

API for dataset upload:

- `POST /api/upload`

Responsibilities:

- accept ZIP upload
- validate file type
- inspect folder structure
- count classes/images
- create initial job record

### `app/routes/jobs.py`

APIs for training job lifecycle:

- `POST /api/jobs`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}/progress`

Responsibilities:

- accept training config
- store config in job record
- submit Prefect deployment
- expose current metrics and status

### `app/routes/artifacts.py`

Artifact download endpoints:

- `GET /api/jobs/{job_id}/download`


## 6. Templates

Use a shared base template plus one template per page.

### `base.html`

Should contain:

- main layout
- navigation
- Tailwind CDN
- Alpine.js if desired
- Chart.js for metrics visualization

### `index.html`

Dashboard page showing:

- list of jobs
- status badges
- links to configure, monitor, or results

### `upload.html`

Upload form with:

- drag-and-drop or file picker
- upload button
- validation/error feedback
- dataset preview after upload

### `configure.html`

Configuration form with:

- architecture selection
- epochs
- learning rate
- batch size
- image size
- augmentation toggle

### `training.html`

Training monitor with:

- current status
- polling logic
- live loss chart
- live accuracy chart
- epoch log table

### `results.html`

Results page with:

- final metrics
- confusion matrix
- full epoch history
- download button


## 7. Frontend Behaviour

Keep JavaScript small and page-specific.

Use `fetch()` for:

- uploading ZIP files
- submitting training configs
- polling progress

Use server-rendered HTML for the initial page structure and state, then enhance with JavaScript where interaction is needed.


## 8. Tailwind Styling

Use Tailwind through the CDN for simplicity.

Guidelines:

- keep a consistent layout shell
- use clear status colors for `pending`, `running`, `completed`, `failed`
- build readable tables and forms first
- keep charts and metrics visually secondary to workflow clarity

No separate frontend build pipeline is necessary for the first version.


## 9. Training Integration

The web app should not run training directly in the request cycle.

Instead:

1. user submits config
2. backend saves config to job state
3. backend submits a Prefect deployment
4. Prefect worker runs the training flow asynchronously

This keeps FastAPI responsive.


## 10. Progress and Results

During training, write progress to JSON files so the UI can poll lightweight endpoints.

The backend should expose a progress endpoint that returns:

- job status
- epoch metrics
- class names
- confusion matrix when available
- error message if failed


## 11. Validation and Error Handling

Add basic checks early:

- uploaded file must be `.zip`
- ZIP must contain class folders
- `job_id` must exist before configure/training/results
- architecture must be one of the allowed values
- download should only work after successful completion

Show user-facing errors inside the page, not only as raw API failures.


## 12. Development Runtime

Typical local startup:

```bash
.venv/bin/prefect server start
.venv/bin/prefect worker start --pool cnn-pool
PYTHONPATH=. .venv/bin/uvicorn app.main:app --reload --port 8000
```

Open:

- web app: `http://localhost:8000`
- Prefect UI: `http://localhost:4200`


## 13. Suggested Build Order

Implement in this order:

1. app bootstrap and config
2. schemas and job store
3. page router and base template
4. upload flow
5. configure flow
6. training submission API
7. training monitor page
8. results page
9. artifact download
10. validation polish and UI cleanup


## Why This Plan

This approach is intentionally simple:

- no React or separate frontend build
- no database required
- clear separation between pages, APIs, and pipeline logic
- easy to debug because state is stored on disk
- good fit for an internal ML training tool

It also leaves room to grow later if needed:

- replace file-based storage with a database
- move Tailwind from CDN to compiled assets
- add authentication
- add richer job management

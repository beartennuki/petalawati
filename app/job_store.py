import json
import shutil
from pathlib import Path
from datetime import datetime
from app.config import ARTIFACTS_DIR, UPLOADS_DIR
from app.schemas import JobStatus, JobConfig


def _job_dir(job_id: str) -> Path:
    d = ARTIFACTS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_dir_path(job_id: str) -> Path:
    return ARTIFACTS_DIR / job_id


def _job_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def create_job(job_id: str, classes: list[str], num_images: int) -> JobStatus:
    job = JobStatus(
        job_id=job_id,
        status="pending",
        classes=classes,
        num_images=num_images,
        created_at=datetime.utcnow().isoformat(),
    )
    _job_path(job_id).write_text(job.model_dump_json(indent=2))
    return job


def get_job(job_id: str) -> JobStatus | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    return JobStatus.model_validate_json(path.read_text())


def update_job(job_id: str, **kwargs) -> JobStatus:
    job = get_job(job_id)
    for k, v in kwargs.items():
        setattr(job, k, v)
    _job_path(job_id).write_text(job.model_dump_json(indent=2))
    return job


def list_jobs() -> list[JobStatus]:
    jobs = []
    for job_file in sorted(ARTIFACTS_DIR.glob("*/job.json"), reverse=True):
        try:
            jobs.append(JobStatus.model_validate_json(job_file.read_text()))
        except Exception:
            pass
    return jobs


def metrics_path(job_id: str) -> Path:
    return _job_dir(job_id) / "metrics.json"


def confusion_path(job_id: str) -> Path:
    return _job_dir(job_id) / "confusion.json"


def model_path(job_id: str) -> Path:
    return _job_dir(job_id) / "model.keras"


def upload_zip_path(job_id: str) -> Path:
    return UPLOADS_DIR / f"{job_id}.zip"


def extracted_upload_dir(job_id: str) -> Path:
    return UPLOADS_DIR / job_id


def append_metric(job_id: str, metric: dict) -> None:
    path = metrics_path(job_id)
    metrics = json.loads(path.read_text()) if path.exists() else []
    metrics.append(metric)
    path.write_text(json.dumps(metrics, indent=2))


def read_metrics(job_id: str) -> list[dict]:
    path = metrics_path(job_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def read_confusion(job_id: str) -> list[list[int]] | None:
    path = confusion_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def delete_job_files(job_id: str) -> None:
    for path in (_job_dir_path(job_id), extracted_upload_dir(job_id)):
        if path.exists():
            shutil.rmtree(path)

    zip_path = upload_zip_path(job_id)
    if zip_path.exists():
        zip_path.unlink()

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import JobConfig, ProgressResponse
from app.job_store import get_job, update_job, read_metrics, read_confusion
from app.config import PREFECT_API_URL

router = APIRouter(prefix="/api")


@router.post("/jobs")
async def create_training_job(config: JobConfig):
    job = get_job(config.job_id)
    if job is None:
        raise HTTPException(404, "Job not found — upload a dataset first")

    update_job(config.job_id, config=config, status="pending")

    # Submit to Prefect work pool (non-blocking)
    try:
        from prefect.deployments import run_deployment
        run_deployment(
            name="training-flow/cnn-deploy",
            parameters={"job_id": config.job_id},
            timeout=0,
        )
    except Exception as e:
        update_job(config.job_id, status="failed", error=str(e))
        raise HTTPException(500, f"Failed to submit Prefect job: {e}")

    return JSONResponse({"job_id": config.job_id, "status": "pending"})


@router.get("/jobs/{job_id}/progress", response_model=ProgressResponse)
async def get_progress(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    return ProgressResponse(
        status=job.status,
        metrics=read_metrics(job_id),
        classes=job.classes,
        confusion_matrix=read_confusion(job_id) if job.status == "completed" else None,
        error=job.error,
    )


@router.get("/jobs")
async def list_jobs_api():
    from app.job_store import list_jobs
    return [j.model_dump() for j in list_jobs()]

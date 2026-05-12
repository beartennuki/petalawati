from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import JobConfig, ProgressResponse
from app.job_store import get_job, update_job, read_metrics, read_confusion, delete_job_files, upload_zip_path
from app.dataset_archive import check_channel_consistency
router = APIRouter(prefix="/api")


@router.post("/jobs")
async def create_training_job(config: JobConfig):
    job = get_job(config.job_id)
    if job is None:
        raise HTTPException(404, "Job not found — upload a dataset first")

    update_job(config.job_id, config=config, status="pending", error=None)

    if config.run_consistency_check:
        zip_path = upload_zip_path(config.job_id)
        if zip_path.exists():
            consistency_errors = check_channel_consistency(zip_path)
            if consistency_errors:
                update_job(config.job_id, status="failed", error="; ".join(consistency_errors))
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": "; ".join(consistency_errors),
                        "errors": consistency_errors,
                    },
                )

    # Submit to Prefect work pool (non-blocking)
    try:
        from prefect.deployments import run_deployment
        flow_run = await run_deployment(
            name="training-flow/cnn-deploy",
            parameters={"job_id": config.job_id},
            timeout=0,
        )
        update_job(config.job_id, flow_run_id=str(flow_run.id))
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


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "running":
        raise HTTPException(409, f"Job is not running (status: {job.status})")
    if not job.flow_run_id:
        raise HTTPException(500, "No Prefect flow run ID recorded for this job")

    try:
        import uuid
        from prefect import get_client
        async with get_client() as client:
            await client.cancel_flow_run(flow_run_id=uuid.UUID(job.flow_run_id))
    except Exception as e:
        raise HTTPException(500, f"Failed to cancel Prefect flow run: {e}")

    update_job(job_id, status="failed", error="Cancelled by user")
    return JSONResponse({"job_id": job_id, "cancelled": True})


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status == "running":
        raise HTTPException(409, "Cannot delete a running job")

    delete_job_files(job_id)
    return JSONResponse({"job_id": job_id, "deleted": True})

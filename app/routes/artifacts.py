from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.job_store import model_path, get_job

router = APIRouter(prefix="/api")


@router.get("/jobs/{job_id}/download")
async def download_model(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "completed":
        raise HTTPException(400, "Model not ready yet")

    path = model_path(job_id)
    if not path.exists():
        raise HTTPException(404, "Model file not found")

    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=f"model_{job_id[:8]}.keras",
    )

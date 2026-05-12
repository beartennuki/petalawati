import uuid
import zipfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.config import UPLOADS_DIR
from app.dataset_archive import inspect_dataset_archive
from app.job_store import create_job

router = APIRouter(prefix="/api")


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only .zip files are accepted")

    job_id = str(uuid.uuid4())
    zip_path = UPLOADS_DIR / f"{job_id}.zip"

    content = await file.read()
    zip_path.write_bytes(content)

    # Peek inside to get class names without full extraction
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is not a valid ZIP")

    classes, num_images = inspect_dataset_archive(names)
    if not classes:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(400, "ZIP must contain class subdirectories")

    create_job(job_id, classes, num_images)

    return JSONResponse({
        "job_id": job_id,
        "classes": classes,
        "num_images": num_images,
    })

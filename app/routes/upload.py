import uuid
import zipfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.config import UPLOADS_DIR
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

    # Discover top-level directories = classes
    classes = sorted({
        p.split("/")[1] if p.count("/") >= 1 else p.split("/")[0]
        for p in names
        if "/" in p and not p.endswith("/")
    } - {""})

    # Fallback: single nested folder — go one level deeper
    if len(classes) == 1 and all(p.startswith(classes[0] + "/") for p in names if "/" in p):
        deeper = sorted({
            p.split("/")[2]
            for p in names
            if p.count("/") >= 2 and not p.endswith("/")
        } - {""})
        if deeper:
            classes = deeper

    if not classes:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(400, "ZIP must contain class subdirectories")

    num_images = sum(
        1 for n in names
        if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    create_job(job_id, classes, num_images)

    return JSONResponse({
        "job_id": job_id,
        "classes": classes,
        "num_images": num_images,
    })

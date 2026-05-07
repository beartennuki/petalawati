from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from app.job_store import list_jobs, get_job
from app.config import ARCHITECTURES

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    jobs = list_jobs()
    return templates.TemplateResponse("index.html", {"request": request, "jobs": jobs})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@router.get("/configure/{job_id}", response_class=HTMLResponse)
async def configure_page(request: Request, job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse("configure.html", {
        "request": request,
        "job": job,
        "architectures": ARCHITECTURES,
    })


@router.get("/training/{job_id}", response_class=HTMLResponse)
async def training_page(request: Request, job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse("training.html", {"request": request, "job": job})


@router.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse("results.html", {"request": request, "job": job})

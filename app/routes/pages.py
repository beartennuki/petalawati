from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from app.job_store import list_jobs, get_job
from app.config import ARCHITECTURES, ARCHITECTURE_CATALOG

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    jobs = list_jobs()
    return templates.TemplateResponse(request, "index.html", {"request": request, "jobs": jobs})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", {"request": request})


@router.get("/catalogue", response_class=HTMLResponse)
async def catalogue_page(request: Request):
    return templates.TemplateResponse(request, "catalogue.html", {
        "request": request,
        "architectures": ARCHITECTURES,
        "catalog": ARCHITECTURE_CATALOG,
    })


@router.get("/configure/{job_id}", response_class=HTMLResponse)
async def configure_page(request: Request, job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse(request, "configure.html", {
        "request": request,
        "job": job,
        "architectures": ARCHITECTURES,
        "catalog": ARCHITECTURE_CATALOG,
    })


@router.get("/training/{job_id}", response_class=HTMLResponse)
async def training_page(request: Request, job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse(request, "training.html", {"request": request, "job": job})


@router.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse(request, "results.html", {"request": request, "job": job})

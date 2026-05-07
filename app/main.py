from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.routes import pages, upload, jobs, artifacts

app = FastAPI(title="Petalawati — CNN Trainer")

app.include_router(pages.router)
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(artifacts.router)

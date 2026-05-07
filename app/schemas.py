from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class JobConfig(BaseModel):
    job_id: str
    architecture: str
    learning_rate: float = 0.001
    epochs: int = 10
    batch_size: int = 32
    image_size: int = 224
    val_split: float = 0.2
    augment: bool = True
    freeze_base: bool = True


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | completed | failed
    config: Optional[JobConfig] = None
    classes: list[str] = []
    num_images: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None


class EpochMetric(BaseModel):
    epoch: int
    loss: float
    val_loss: float
    accuracy: float
    val_accuracy: float


class ProgressResponse(BaseModel):
    status: str
    metrics: list[EpochMetric] = []
    classes: list[str] = []
    confusion_matrix: Optional[list[list[int]]] = None
    error: Optional[str] = None

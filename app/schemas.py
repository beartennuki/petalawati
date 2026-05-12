from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime
from app.config import ARCHITECTURE_KEYS

ArchitectureKey = Literal[
    ARCHITECTURE_KEYS[0],
    ARCHITECTURE_KEYS[1],
    ARCHITECTURE_KEYS[2],
    ARCHITECTURE_KEYS[3],
    ARCHITECTURE_KEYS[4],
]


class JobConfig(BaseModel):
    job_id: str
    architecture: ArchitectureKey
    learning_rate: float = Field(default=0.001, gt=0)
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1)
    image_size: int = Field(default=224, ge=32)
    val_split: float = Field(default=0.2, gt=0, lt=1)
    augment: bool = True


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | completed | failed
    config: Optional[JobConfig] = None
    classes: list[str] = []
    num_images: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None
    flow_run_id: Optional[str] = None


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

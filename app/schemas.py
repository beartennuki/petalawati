from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from app.config import ARCHITECTURES


class JobConfig(BaseModel):
    job_id: str
    architecture: str

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v: str) -> str:
        if v not in ARCHITECTURES:
            raise ValueError(f"must be one of: {', '.join(ARCHITECTURES)}")
        return v
    learning_rate: float = Field(default=0.001, gt=0)
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=1)
    image_size: int = Field(default=224, ge=32)
    val_split: float = Field(default=0.3, gt=0, lt=1)
    augment: bool = True
    run_consistency_check: bool = True


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
    epoch_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None


class ProgressResponse(BaseModel):
    status: str
    metrics: list[EpochMetric] = []
    classes: list[str] = []
    confusion_matrix: Optional[list[list[int]]] = None
    error: Optional[str] = None

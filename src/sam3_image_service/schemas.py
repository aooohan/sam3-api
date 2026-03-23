from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class MaskRLE(BaseModel):
    size: list[int] = Field(description="Mask size in [height, width] format.")
    counts: list[int] = Field(description="Uncompressed COCO-style RLE counts.")


class Detection(BaseModel):
    index: int
    label: str
    score: float | None = None
    area: int
    box: BoundingBox
    mask: MaskRLE | None = None


class ImageSize(BaseModel):
    width: int
    height: int


class RecognitionResponse(BaseModel):
    prompt: str
    device: str | None = None
    image_size: ImageSize
    detections: list[Detection]
    took_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    runtime_dependencies_available: bool
    device: str | None = None
    checkpoint_path: str | None = None
    message: str | None = None

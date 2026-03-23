from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from time import perf_counter

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from sam3_image_service.backend import (
    InferenceBackend,
    Sam3BackendError,
    Sam3ModelBackend,
)
from sam3_image_service.config import Settings, get_settings
from sam3_image_service.schemas import HealthResponse, RecognitionResponse


def get_backend(request: Request) -> InferenceBackend:
    return request.app.state.backend


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def create_app(
    *,
    settings: Settings | None = None,
    backend: InferenceBackend | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    backend = backend or Sam3ModelBackend(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.sam3_download_on_startup:
            try:
                app.state.backend.prepare()
            except Sam3BackendError:
                # The health endpoint will surface the reason if the runtime is not ready.
                pass
        if settings.sam3_warmup_on_startup:
            try:
                app.state.backend.ensure_loaded()
            except Sam3BackendError:
                # The health endpoint will surface the reason if the runtime is not ready.
                pass
        yield

    app = FastAPI(
        title="SAM3 Image Recognition Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.backend = backend

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz(active_backend: InferenceBackend = Depends(get_backend)) -> HealthResponse:
        return HealthResponse.model_validate(active_backend.describe())

    @app.post("/v1/recognize", response_model=RecognitionResponse)
    async def recognize(
        file: UploadFile = File(...),
        prompt: str = Form(..., min_length=1, max_length=200),
        include_masks: bool | None = Form(default=None),
        score_threshold: float | None = Form(default=None, ge=0.0, le=1.0),
        mask_threshold: float | None = Form(default=None, ge=0.0, le=1.0),
        active_backend: InferenceBackend = Depends(get_backend),
        active_settings: Settings = Depends(get_app_settings),
    ) -> RecognitionResponse:
        try:
            payload = await file.read()
            image = Image.open(BytesIO(payload))
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

        started_at = perf_counter()
        try:
            result = active_backend.recognize(
                image=image,
                prompt=prompt.strip(),
                score_threshold=(
                    score_threshold
                    if score_threshold is not None
                    else active_settings.sam3_score_threshold
                ),
                mask_threshold=(
                    mask_threshold
                    if mask_threshold is not None
                    else active_settings.sam3_mask_threshold
                ),
                include_masks=(
                    include_masks
                    if include_masks is not None
                    else active_settings.sam3_include_masks_by_default
                ),
            )
        except Sam3BackendError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        took_ms = round((perf_counter() - started_at) * 1000, 2)
        return RecognitionResponse.model_validate(
            {
                "prompt": result.prompt,
                "device": result.device,
                "image_size": {
                    "width": result.image_width,
                    "height": result.image_height,
                },
                "detections": [
                    {
                        "index": detection.index,
                        "label": detection.label,
                        "score": detection.score,
                        "area": detection.area,
                        "box": detection.box,
                        "mask": detection.mask,
                    }
                    for detection in result.detections
                ],
                "took_ms": took_ms,
            }
        )

    return app

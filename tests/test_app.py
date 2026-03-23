from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from sam3_image_service.app import create_app
from sam3_image_service.backend import DetectionResult, RecognitionResult, Sam3BackendError
from sam3_image_service.config import Settings


class FakeBackend:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    def describe(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model_loaded": True,
            "runtime_dependencies_available": True,
            "device": "cpu",
            "checkpoint_path": None,
            "message": None,
        }

    def ensure_loaded(self) -> None:
        return None

    def recognize(
        self,
        image: Image.Image,
        prompt: str,
        *,
        score_threshold: float,
        mask_threshold: float,
        include_masks: bool,
    ) -> RecognitionResult:
        if self.should_fail:
            raise Sam3BackendError("runtime unavailable")

        mask = {"size": [image.height, image.width], "counts": [0, 4]}
        return RecognitionResult(
            prompt=prompt,
            device="cpu",
            image_width=image.width,
            image_height=image.height,
            detections=[
                DetectionResult(
                    index=0,
                    label=prompt,
                    score=0.92,
                    area=4,
                    box={"x1": 1.0, "y1": 2.0, "x2": 5.0, "y2": 6.0},
                    mask=mask if include_masks else None,
                )
            ],
        )


def build_client(backend: FakeBackend) -> TestClient:
    settings = Settings(
        APP_HOST="127.0.0.1",
        APP_PORT=8000,
        SAM3_INCLUDE_MASKS_BY_DEFAULT=False,
    )
    app = create_app(settings=settings, backend=backend)
    return TestClient(app)


def create_test_png() -> bytes:
    image = Image.new("RGB", (8, 8), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_healthz_returns_backend_status() -> None:
    client = build_client(FakeBackend())

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["device"] == "cpu"


def test_recognize_returns_detection_payload() -> None:
    client = build_client(FakeBackend())

    response = client.post(
        "/v1/recognize",
        files={"file": ("demo.png", create_test_png(), "image/png")},
        data={"prompt": "white dog", "include_masks": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt"] == "white dog"
    assert payload["image_size"] == {"width": 8, "height": 8}
    assert len(payload["detections"]) == 1
    assert payload["detections"][0]["mask"] is not None


def test_recognize_returns_503_when_backend_is_unavailable() -> None:
    client = build_client(FakeBackend(should_fail=True))

    response = client.post(
        "/v1/recognize",
        files={"file": ("demo.png", create_test_png(), "image/png")},
        data={"prompt": "white dog"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "runtime unavailable"


def test_recognize_rejects_invalid_image_upload() -> None:
    client = build_client(FakeBackend())

    response = client.post(
        "/v1/recognize",
        files={"file": ("broken.txt", b"not-an-image", "text/plain")},
        data={"prompt": "white dog"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."

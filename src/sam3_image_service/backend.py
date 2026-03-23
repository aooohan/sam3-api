from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from sam3_image_service.config import Settings


class Sam3BackendError(RuntimeError):
    """Raised when the SAM3 runtime cannot serve a request."""


@dataclass(slots=True)
class DetectionResult:
    index: int
    label: str
    score: float | None
    area: int
    box: dict[str, float]
    mask: dict[str, list[int]] | None


@dataclass(slots=True)
class RecognitionResult:
    prompt: str
    device: str | None
    image_width: int
    image_height: int
    detections: list[DetectionResult]


class InferenceBackend(Protocol):
    def describe(self) -> dict[str, Any]:
        ...

    def prepare(self) -> None:
        ...

    def ensure_loaded(self) -> None:
        ...

    def recognize(
        self,
        image: Image.Image,
        prompt: str,
        *,
        score_threshold: float,
        mask_threshold: float,
        include_masks: bool,
    ) -> RecognitionResult:
        ...


DownloadFile = Callable[..., str]


def _hf_download_file(**kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


class LocalCheckpointResolver:
    def __init__(
        self,
        settings: Settings,
        *,
        download_file: DownloadFile | None = None,
    ) -> None:
        self._settings = settings
        self._download_file = download_file or _hf_download_file

    @property
    def expected_checkpoint_path(self) -> Path:
        if self._settings.sam3_checkpoint_path:
            return Path(self._settings.sam3_checkpoint_path).expanduser().resolve()
        return (
            Path(self._settings.sam3_model_dir).expanduser().resolve()
            / self._settings.sam3_checkpoint_filename
        )

    def cached_checkpoint_path(self) -> Path | None:
        checkpoint_path = self.expected_checkpoint_path
        if checkpoint_path.exists():
            return checkpoint_path
        return None

    def resolve(self, *, download_if_missing: bool) -> Path:
        checkpoint_path = self.expected_checkpoint_path
        if self._settings.sam3_checkpoint_path:
            if checkpoint_path.exists():
                return checkpoint_path
            raise Sam3BackendError(
                f"SAM3 checkpoint path does not exist: {checkpoint_path}. "
                "Set SAM3_CHECKPOINT_PATH to a valid file."
            )

        if checkpoint_path.exists() and not self._settings.sam3_force_download:
            return checkpoint_path

        if not download_if_missing or not self._settings.sam3_load_from_hf:
            raise Sam3BackendError(
                "SAM3 checkpoint is not available locally. "
                "Set SAM3_CHECKPOINT_PATH or enable SAM3_LOAD_FROM_HF."
            )

        model_dir = checkpoint_path.parent
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._download_file(
                repo_id=self._settings.sam3_hf_model_id,
                filename="config.json",
                local_dir=model_dir,
                token=self._settings.sam3_hf_token,
                endpoint=self._settings.sam3_hf_endpoint,
                force_download=self._settings.sam3_force_download,
                local_files_only=self._settings.sam3_local_files_only,
            )
            self._download_file(
                repo_id=self._settings.sam3_hf_model_id,
                filename=self._settings.sam3_checkpoint_filename,
                local_dir=model_dir,
                token=self._settings.sam3_hf_token,
                endpoint=self._settings.sam3_hf_endpoint,
                force_download=self._settings.sam3_force_download,
                local_files_only=self._settings.sam3_local_files_only,
            )
        except ImportError as exc:
            raise Sam3BackendError(
                "huggingface_hub is not available. Run `uv sync --extra runtime` first."
            ) from exc
        except Exception as exc:
            raise Sam3BackendError(
                "Failed to download SAM3 model files from the configured Hugging Face "
                f"endpoint `{self._settings.sam3_hf_endpoint}`. Make sure this machine "
                f"has access to the gated `{self._settings.sam3_hf_model_id}` repository, "
                "then set `HF_TOKEN` or run `hf auth login`."
            ) from exc

        if checkpoint_path.exists():
            return checkpoint_path

        raise Sam3BackendError(
            f"SAM3 files were downloaded but `{checkpoint_path.name}` was not found in "
            f"{checkpoint_path.parent}."
        )


class Sam3ModelBackend:
    def __init__(
        self,
        settings: Settings,
        *,
        checkpoint_resolver: LocalCheckpointResolver | None = None,
    ) -> None:
        self._settings = settings
        self._checkpoint_resolver = checkpoint_resolver or LocalCheckpointResolver(settings)
        self._processor: Any | None = None
        self._device: str | None = None
        self._checkpoint_path: str | None = None
        self._last_error: str | None = None

    def describe(self) -> dict[str, Any]:
        deps_available = (
            find_spec("sam3") is not None
            and find_spec("torch") is not None
            and find_spec("psutil") is not None
            and find_spec("huggingface_hub") is not None
        )
        checkpoint_path = self._checkpoint_path
        cached_checkpoint = self._checkpoint_resolver.cached_checkpoint_path()
        if checkpoint_path is None and cached_checkpoint is not None:
            checkpoint_path = str(cached_checkpoint)

        message = self._last_error
        if not deps_available:
            message = "SAM3 runtime dependencies are missing. Run `uv sync --extra runtime`."
        elif (
            message is None
            and checkpoint_path is None
            and not self._settings.sam3_load_from_hf
            and self._settings.sam3_checkpoint_path is None
        ):
            message = (
                "SAM3 checkpoint is not available locally. Set SAM3_CHECKPOINT_PATH or "
                "enable SAM3_LOAD_FROM_HF."
            )
        elif message is None and checkpoint_path is None and self._settings.sam3_load_from_hf:
            message = (
                "SAM3 checkpoint is not cached yet. It will be downloaded into "
                f"{self._checkpoint_resolver.expected_checkpoint_path.parent}."
            )

        status = "ok" if deps_available and message is None else "degraded"

        return {
            "status": status,
            "model_loaded": self._processor is not None,
            "runtime_dependencies_available": deps_available,
            "device": self._device,
            "checkpoint_path": checkpoint_path,
            "message": message,
        }

    def prepare(self) -> None:
        try:
            checkpoint_path = self._checkpoint_resolver.resolve(download_if_missing=True)
        except Sam3BackendError as exc:
            self._last_error = str(exc)
            raise

        self._checkpoint_path = str(checkpoint_path)
        self._last_error = None

    def ensure_loaded(self) -> None:
        if self._processor is not None:
            return

        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise Sam3BackendError(
                "SAM3 runtime dependencies are missing. Run `uv sync --extra runtime` "
                "before calling the recognition endpoint."
            ) from exc

        device = self._resolve_device(torch)
        try:
            checkpoint_path = self._checkpoint_resolver.resolve(download_if_missing=True)
        except Sam3BackendError as exc:
            self._last_error = str(exc)
            raise

        self._checkpoint_path = str(checkpoint_path)

        builder_kwargs: dict[str, Any] = {
            "device": device,
            "compile": self._settings.sam3_enable_compile,
            "checkpoint_path": str(checkpoint_path),
            "load_from_HF": False,
        }

        try:
            model = build_sam3_image_model(**builder_kwargs)
        except TypeError:
            # Be tolerant of minor API differences across sam3 releases.
            fallback_kwargs = {
                "device": device,
                "checkpoint_path": str(checkpoint_path),
            }
            model = build_sam3_image_model(**fallback_kwargs)

        self._processor = Sam3Processor(model)
        self._device = device
        self._last_error = None

    def recognize(
        self,
        image: Image.Image,
        prompt: str,
        *,
        score_threshold: float,
        mask_threshold: float,
        include_masks: bool,
    ) -> RecognitionResult:
        self.ensure_loaded()

        assert self._processor is not None
        image = image.convert("RGB")
        inference_state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = self._normalize_masks(output.get("masks"), mask_threshold)
        boxes = self._normalize_boxes(output.get("boxes"), len(masks))
        scores = self._normalize_scores(output.get("scores"), len(masks))

        detections: list[DetectionResult] = []
        for index, mask in enumerate(masks):
            score = scores[index] if index < len(scores) else None
            if score is not None and score < score_threshold:
                continue

            box = boxes[index] if index < len(boxes) else self._box_from_mask(mask)
            detections.append(
                DetectionResult(
                    index=index,
                    label=prompt,
                    score=score,
                    area=int(mask.sum()),
                    box={
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    },
                    mask=self._encode_mask(mask) if include_masks else None,
                )
            )

        return RecognitionResult(
            prompt=prompt,
            device=self._device,
            image_width=image.width,
            image_height=image.height,
            detections=detections,
        )

    def _resolve_device(self, torch: Any) -> str:
        configured = self._settings.sam3_device.lower()
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return np.empty((0,))
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
        return np.asarray(value)

    def _normalize_masks(self, masks: Any, mask_threshold: float) -> list[np.ndarray]:
        array = self._to_numpy(masks)
        if array.size == 0:
            return []

        while array.ndim > 3:
            array = array[0]

        if array.ndim == 2:
            array = array[np.newaxis, ...]

        if array.ndim != 3:
            raise Sam3BackendError(f"Unexpected mask shape returned by SAM3: {array.shape!r}")

        return [np.asarray(mask > mask_threshold, dtype=np.uint8) for mask in array]

    def _normalize_boxes(self, boxes: Any, expected: int) -> list[np.ndarray]:
        array = self._to_numpy(boxes)
        if array.size == 0:
            return [np.asarray(self._empty_box(), dtype=np.float32) for _ in range(expected)]

        if array.ndim == 1 and array.shape[0] == 4:
            array = array[np.newaxis, ...]

        if array.ndim != 2 or array.shape[1] != 4:
            return [np.asarray(self._empty_box(), dtype=np.float32) for _ in range(expected)]

        return [np.asarray(box, dtype=np.float32) for box in array]

    def _normalize_scores(self, scores: Any, expected: int) -> list[float | None]:
        array = self._to_numpy(scores)
        if array.size == 0:
            return [None] * expected

        flat = np.asarray(array).reshape(-1)
        return [float(score) for score in flat.tolist()]

    @staticmethod
    def _empty_box() -> tuple[float, float, float, float]:
        return (0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def _box_from_mask(mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return np.asarray((0.0, 0.0, 0.0, 0.0), dtype=np.float32)
        return np.asarray(
            (
                float(xs.min()),
                float(ys.min()),
                float(xs.max()),
                float(ys.max()),
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _encode_mask(mask: np.ndarray) -> dict[str, list[int]]:
        flattened = mask.astype(np.uint8).T.flatten()
        counts: list[int] = []
        current_value = 0
        run_length = 0

        for pixel in flattened.tolist():
            if pixel == current_value:
                run_length += 1
                continue
            counts.append(run_length)
            run_length = 1
            current_value = pixel

        counts.append(run_length)
        return {
            "size": [int(mask.shape[0]), int(mask.shape[1])],
            "counts": counts,
        }

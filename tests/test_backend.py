from __future__ import annotations

from pathlib import Path

import pytest

from sam3_image_service.backend import LocalCheckpointResolver, Sam3BackendError, Sam3ModelBackend
from sam3_image_service.config import Settings


def test_settings_default_to_new_sam3_repo() -> None:
    settings = Settings(SAM3_DOWNLOAD_ON_STARTUP=False)

    assert settings.sam3_hf_model_id == "AnantP78/sam3_pt"


def test_settings_normalize_full_hf_repo_url() -> None:
    settings = Settings(
        SAM3_HF_MODEL_ID="https://huggingface.co/AnantP78/sam3_pt/",
        SAM3_DOWNLOAD_ON_STARTUP=False,
    )

    assert settings.sam3_hf_model_id == "AnantP78/sam3_pt"


def test_checkpoint_resolver_uses_existing_explicit_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "custom.pt"
    checkpoint_path.write_bytes(b"weights")
    settings = Settings(
        SAM3_CHECKPOINT_PATH=str(checkpoint_path),
        SAM3_DOWNLOAD_ON_STARTUP=False,
    )

    resolver = LocalCheckpointResolver(settings)

    assert resolver.resolve(download_if_missing=False) == checkpoint_path.resolve()


def test_checkpoint_resolver_downloads_missing_checkpoint_to_model_dir(tmp_path: Path) -> None:
    download_calls: list[dict[str, str]] = []
    model_dir = tmp_path / "models" / "sam3"

    def fake_download_file(**kwargs: str) -> str:
        filename = kwargs["filename"]
        download_calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(filename, encoding="utf-8")
        return str(target)

    settings = Settings(
        SAM3_MODEL_DIR=str(model_dir),
        SAM3_DOWNLOAD_ON_STARTUP=False,
    )
    resolver = LocalCheckpointResolver(settings, download_file=fake_download_file)

    checkpoint_path = resolver.resolve(download_if_missing=True)

    assert checkpoint_path == (model_dir / "sam3.pt").resolve()
    assert checkpoint_path.exists()
    assert [call["filename"] for call in download_calls] == ["config.json", "sam3.pt"]
    assert all(call["endpoint"] == "https://hf-mirror.com" for call in download_calls)


def test_checkpoint_resolver_uses_custom_endpoint_from_env(tmp_path: Path) -> None:
    download_calls: list[dict[str, str]] = []
    model_dir = tmp_path / "models" / "sam3"

    def fake_download_file(**kwargs: str) -> str:
        download_calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("weights", encoding="utf-8")
        return str(target)

    settings = Settings(
        SAM3_MODEL_DIR=str(model_dir),
        HF_ENDPOINT="https://custom-hf-mirror.example",
        SAM3_DOWNLOAD_ON_STARTUP=False,
    )

    resolver = LocalCheckpointResolver(settings, download_file=fake_download_file)
    resolver.resolve(download_if_missing=True)

    assert all(call["endpoint"] == "https://custom-hf-mirror.example" for call in download_calls)


def test_checkpoint_resolver_raises_when_download_is_disabled(tmp_path: Path) -> None:
    settings = Settings(
        SAM3_MODEL_DIR=str(tmp_path / "models" / "sam3"),
        SAM3_LOAD_FROM_HF=False,
        SAM3_DOWNLOAD_ON_STARTUP=False,
    )
    resolver = LocalCheckpointResolver(settings)

    with pytest.raises(Sam3BackendError, match="checkpoint is not available locally"):
        resolver.resolve(download_if_missing=True)


def test_backend_describe_reports_cached_checkpoint_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "models" / "sam3" / "sam3.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"weights")
    settings = Settings(
        SAM3_MODEL_DIR=str(checkpoint_path.parent),
        SAM3_DOWNLOAD_ON_STARTUP=False,
    )

    backend = Sam3ModelBackend(settings)
    description = backend.describe()

    assert description["checkpoint_path"] == str(checkpoint_path.resolve())
    assert description["status"] in {"ok", "degraded"}

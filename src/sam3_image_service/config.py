from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_log_level: str = Field(default="info", alias="APP_LOG_LEVEL")

    sam3_device: str = Field(default="auto", alias="SAM3_DEVICE")
    sam3_checkpoint_path: str | None = Field(default=None, alias="SAM3_CHECKPOINT_PATH")
    sam3_model_dir: str = Field(default="models/sam3", alias="SAM3_MODEL_DIR")
    sam3_checkpoint_filename: str = Field(default="sam3.pt", alias="SAM3_CHECKPOINT_FILENAME")
    sam3_hf_model_id: str = Field(default="AnantP78/sam3_pt", alias="SAM3_HF_MODEL_ID")
    sam3_hf_endpoint: str | None = Field(
        default="https://hf-mirror.com",
        validation_alias=AliasChoices("SAM3_HF_ENDPOINT", "HF_ENDPOINT"),
    )
    sam3_hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SAM3_HF_TOKEN", "HF_TOKEN"),
    )
    sam3_load_from_hf: bool = Field(default=True, alias="SAM3_LOAD_FROM_HF")
    sam3_download_on_startup: bool = Field(default=True, alias="SAM3_DOWNLOAD_ON_STARTUP")
    sam3_force_download: bool = Field(default=False, alias="SAM3_FORCE_DOWNLOAD")
    sam3_local_files_only: bool = Field(default=False, alias="SAM3_LOCAL_FILES_ONLY")
    sam3_enable_compile: bool = Field(default=False, alias="SAM3_ENABLE_COMPILE")
    sam3_score_threshold: float = Field(default=0.25, alias="SAM3_SCORE_THRESHOLD")
    sam3_mask_threshold: float = Field(default=0.5, alias="SAM3_MASK_THRESHOLD")
    sam3_include_masks_by_default: bool = Field(
        default=False,
        alias="SAM3_INCLUDE_MASKS_BY_DEFAULT",
    )
    sam3_warmup_on_startup: bool = Field(
        default=False,
        alias="SAM3_WARMUP_ON_STARTUP",
    )

    @field_validator("sam3_hf_model_id", mode="before")
    @classmethod
    def normalize_repo_id(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        cleaned = value.strip()
        if cleaned.startswith(("http://", "https://")):
            parsed = urlparse(cleaned)
            path_parts = [part for part in parsed.path.split("/") if part]
            if len(path_parts) >= 2:
                return "/".join(path_parts[:2])

        return cleaned.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

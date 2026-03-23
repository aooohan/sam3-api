from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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
    sam3_load_from_hf: bool = Field(default=True, alias="SAM3_LOAD_FROM_HF")
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

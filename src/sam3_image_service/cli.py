from __future__ import annotations

import uvicorn

from sam3_image_service.app import create_app
from sam3_image_service.config import get_settings


def main() -> None:
    settings = get_settings()
    app = create_app(settings=settings)
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.app_log_level,
    )

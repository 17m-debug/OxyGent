import sys
import os
from pathlib import Path

# Remove loguru default handler BEFORE importing any other modules
# This prevents err.log files from being created by modules imported later
from loguru import logger
logger.remove()  # Remove default stderr handler

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from app.api.router import api_router
from core.config import settings

# Configure loguru: remove default handler, only output to console
# (Already removed above, but keeping comment for clarity)

# Get log level from environment variable, default is INFO
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
handler_id = logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=log_level,
    colorize=True
)

# Store handler_id in module variable for use by log_config module
import app.api.log.log_config as log_config_module
log_config_module._handler_id = handler_id
log_config_module._log_level = log_level

app = FastAPI()

app.include_router(api_router, prefix="/api/v1")

# Initialize endpoint registry (singleton pattern) and restore all registered endpoints
from core.interface.endpoint_registry import EndpointRegistry
endpoint_registry = EndpointRegistry(app=app)
endpoint_registry.restore_all_endpoints()


ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT_DIR / "web" / "dist"


def _mount_frontend_dist() -> None:
    index_file = DIST_DIR / "index.html"
    if not index_file.exists():
        raise RuntimeError("web/dist not found; run pnpm build first.")

    @app.middleware("http")
    async def spa_static_middleware(request: Request, call_next):
        path = request.url.path
        if path.startswith(("/api", "/kb", "/docs", "/openapi.json", "/redoc")):
            return await call_next(request)

        file_path = DIST_DIR / path.lstrip("/")
        if file_path.is_file():
            return FileResponse(file_path)
        if "." in Path(path).name:
            return await call_next(request)
        return FileResponse(index_file)

    logger.info("Serving web/dist from FastAPI.")


if __name__ == "__main__":
    _mount_frontend_dist()
    uvicorn.run(app, host=settings.host, port=settings.port)

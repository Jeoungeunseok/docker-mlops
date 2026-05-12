from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.domains.mlops.bootstrap import bootstrap_mlops_components


def create_app() -> FastAPI:
    configure_logging(settings)
    bootstrap_mlops_components()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()

"""FastAPI application entrypoint for the MVP API service."""

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.observability.middleware.http_asgi import ObservabilityHTTPMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        FastAPI: Configured FastAPI application.
    """
    app = FastAPI(title="Nano GraphRAG API", version="0.1.0")
    app.add_middleware(ObservabilityHTTPMiddleware)
    app.include_router(api_router)
    return app


app = create_app()

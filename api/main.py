import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import torch

from api.routes.v1_router import v1_router
from api.settings import api_settings
from api.state import ml_models  # <-- Import from the neutral state file
from .core.embedding_manager import OptimizedEmbeddingManager

# A simple dictionary to hold the loaded model manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Server starting up ---")

    # Instantiate the optimized embedding manager
    manager = OptimizedEmbeddingManager()
    print(f"Embedding manager initialized on device: {manager.device}")

    # Store manager in shared state
    ml_models["embedding_manager"] = manager
    print("✅ Embedding manager loaded successfully.")
    print(ml_models)

    yield

    print("--- Server shutting down ---")
    ml_models.clear()


def create_app() -> FastAPI:
    """Create a FastAPI App"""
    app: FastAPI = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url="/openapi.json" if api_settings.docs_enabled else None,
        lifespan=lifespan,  # <-- Use the new lifespan manager
    )

    app.include_router(v1_router, prefix="/api")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()

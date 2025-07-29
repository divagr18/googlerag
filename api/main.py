# main.py (or app/main.py)
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import torch
from sentence_transformers import SentenceTransformer

from api.routes.v1_router import v1_router
from api.settings import api_settings
from api.state import ml_models  # <-- Import from the neutral state file


# A simple dictionary to hold the loaded model

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Server starting up ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_name = 'nomic-ai/nomic-embed-text-v1.5'
    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name, device=device,trust_remote_code=True)
    model.max_seq_length = 512  # optional
    ml_models["embedding_model"] = model
    print("✅ Embedding model loaded successfully.")
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
        lifespan=lifespan  # <-- ADD THIS LIFESPAN MANAGER
    )

    app.include_router(v1_router,prefix="/api")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

app = create_app()
from fastapi import APIRouter

from api.routes.health import health_router

# Corrected: Standardized to absolute import
from api.routes.ragsys import ragsys_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(health_router)
v1_router.include_router(ragsys_router)

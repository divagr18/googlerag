from fastapi import APIRouter

from api.routes.health import health_router

# Corrected: Standardized to absolute import
from api.routes.hackrx import hackrx_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(health_router)
v1_router.include_router(hackrx_router)

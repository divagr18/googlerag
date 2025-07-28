from fastapi import APIRouter

from api.routes.health import health_router
from .hackrx import hackrx_router # <-- IMPORT

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(health_router)
v1_router.include_router(hackrx_router) # <-- INCLUDE


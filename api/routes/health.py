from fastapi import APIRouter

######################################################
## Routes for the API Health
######################################################

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
def get_health():
    """Check the health of the Api"""

    return {
        "status": "success",
    }

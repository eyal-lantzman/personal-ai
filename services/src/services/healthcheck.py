from fastapi import status, APIRouter, HTTPException
from pydantic import BaseModel
from services.registry import REGISTRY

healthcheck_router = APIRouter()

class HealthCheck(BaseModel):
    status: str


@healthcheck_router.get(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck
)
def get_health() -> HealthCheck:
    if len(REGISTRY.models) == 0:
        raise HTTPException(status_code=500, detail="No Models")
    
    return HealthCheck(status="OK")

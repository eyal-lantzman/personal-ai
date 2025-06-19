# TODO: move towards https://github.com/xregistry/spec
from fastapi import status, APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.internal.registry import ModelCard, Models
import logging

logger = logging.getLogger(__name__)

registry_router = APIRouter()

REGISTRY = Models()

class ImportModel(BaseModel):
    model_id:str
    token:Optional[str] = None


@registry_router.get(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=List[str]
)
def get_models() -> List[str]:
    return list(REGISTRY.models.keys())

@registry_router.get(
    "/{provider_id}",
    status_code=status.HTTP_200_OK,
    response_model=List[str]
)
def get_provider_models(provider_id:str) -> List[str]:
    provider_id = provider_id.rstrip("/")

    response = list()
    for model_id in REGISTRY.models.keys():
        if model_id.startswith(provider_id + "/"):
            response.append(model_id[len(provider_id)+1:])

    if not response:
        raise HTTPException(status_code=404, detail="No such provider.")

    return response

@registry_router.get(
    "/{provider_id}/{model_name}",
    status_code=status.HTTP_200_OK,
    response_model=ModelCard
)
def get_model(provider_id:str, model_name:str) -> ModelCard:
    model_id = f"{provider_id}/{model_name}"
    if model_id not in REGISTRY.models:
        raise HTTPException(status_code=404, detail=f"No such model.")

    return REGISTRY.models[model_id]


@registry_router.post(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=ModelCard
)
def import_model(request:ImportModel) -> ModelCard:
    # TODO: add support for long running operations
    if not REGISTRY.import_model(request.model_id, token=request.token):
        logger.warning("Failed to import a model: %s", request.model_id)
        
    return REGISTRY.models[request.model_id]
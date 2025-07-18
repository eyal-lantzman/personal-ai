from fastapi import status, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal, List
from services.internal.registry import Models
import logging

logger = logging.getLogger(__name__)

models_router = APIRouter()

REGISTRY = Models()

class Model(BaseModel):
    id: str
    """The model identifier, which can be referenced in the API endpoints."""

    created: int
    """The Unix timestamp (in seconds) when the model was created."""

    object: Literal["model"]
    """The object type, which is always "model"."""

    owned_by: str
    """The organization that owns the model."""


class ModelsResponse(BaseModel):
    data: List[Model]
    """The list of embeddings generated by the model."""

    object: Literal["list"]
    """The object type, which is always "list"."""

@models_router.get(
    "/",
    status_code=status.HTTP_200_OK,
    response_model=ModelsResponse
)
def get_models() -> ModelsResponse:
    response = ModelsResponse(data=list[Model](), object="list")
    for k,v in REGISTRY.models.items():
        response.data.append(Model(id=k, created=v.created, object="model", owned_by="me"))
    return response


@models_router.get(
    "/{provider}/{model}",
    status_code=status.HTTP_200_OK,
    response_model=Model
)
def get_model(provider:str, model:str=None) -> Model:
    model_id = "/".join([provider, model])
    if model_id not in REGISTRY.models:
        raise HTTPException(status_code=404, detail=f"No such model.")
    
    return Model(
        id=model_id, 
        created=REGISTRY.models[model_id].created, 
        object="model", 
        owned_by="me")
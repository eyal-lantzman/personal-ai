from fastapi import APIRouter,Request
from requests import Session
import httpx
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

import logging

logger = logging.getLogger(__name__)

# Create a FastAPI app
lm_studio_lb_router = APIRouter()

_LM_STUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234"

def get_base_url():
    return _LM_STUDIO_DEFAULT_BASE_URL

def set_base_url(base_url:str):
    global _LM_STUDIO_DEFAULT_BASE_URL
    _LM_STUDIO_DEFAULT_BASE_URL = base_url

def get_local_models() -> list[dict]:
    session = Session()
    response = session.get(f"{get_base_url()}/api/v0/models")
    response.raise_for_status()
    models = response.json()["data"]
    logger.debug("Models found: %s", models)
    return models

def get_model_state(model_id):
    session = Session()
    response = session.get(f"{get_base_url()}/api/v0/models/{model_id}")
    response.raise_for_status()
    model = response.json()
    logger.debug("Model found: %s", model)
    return model["state"]

def build_model_id_to_alternatives_map() -> dict[str, set]:
    map = dict[str, set]()

    for lm_studio_model in get_local_models():
        parts = lm_studio_model["id"].split(":")
        mode_class_id = parts[0]
        if mode_class_id not in map:
            map[mode_class_id] = set()
            map[mode_class_id].add(mode_class_id)
        else:
            map[mode_class_id].add(lm_studio_model["id"])

    return map

lm_studio_client = httpx.AsyncClient(timeout=15.0*60.0)

@lm_studio_lb_router.post("/{path:path}")
@lm_studio_lb_router.get("/{path:path}")
async def proxy(request: Request, path: str):

    data = await request.body()
    json = None
    if data != b"":
        import json
        json = json.loads(data.decode("unicode_escape"))

    #if (path == "chat/completions" or path == "embeddings") and "model" in data:
        # TODO: add load balancing based on json["model"]
        # map = build_model_id_to_alternatives_map()
        # alternatives = map.get(json["model"])


    url = f"{_LM_STUDIO_DEFAULT_BASE_URL}/v1/{path}"

    # Forward the request
    if json and "stream" in json and json["stream"]:
        proxied_request = lm_studio_client.build_request(request.method, url, headers=request.headers.raw, data=data)
        response = await lm_studio_client.send(proxied_request, stream=True)
        return StreamingResponse(
            response.aiter_text(),
            media_type="application/x-ndjson",
            background=BackgroundTask(response.aclose)
        )
    else:
        response = await lm_studio_client.request(
            request.method, 
            url, 
            headers=request.headers.raw, 
            data=data,
        )
        try:
            return response.json()
        finally:
            await response.aclose()
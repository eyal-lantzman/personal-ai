from requests import Session
import logging

logger = logging.getLogger(__name__)

DEFAULT_FAST_MODEL = "qwen3-0.6b"
_DEFAULT_BASE_URL = "http://localhost:1234/v1"

def get_base_url():
    return _DEFAULT_BASE_URL

def set_base_url(base_url:str):
    global _DEFAULT_BASE_URL
    _DEFAULT_BASE_URL = base_url

def get_local_models():
    session = Session()
    response = session.get(f"{get_base_url()}/models")
    response.raise_for_status()
    models = response.json()["data"]
    logger.debug("Models found: %s", models)
    return models

def get_loaded_model():
    default_model = get_local_models()[0]["id"]
    logger.debug("Default model found: %s", default_model)
    return default_model
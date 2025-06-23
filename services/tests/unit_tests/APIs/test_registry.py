import pytest
from unittest import mock
import os
from pathlib import Path
from fastapi.testclient import TestClient

@pytest.fixture
def model_cache():
    return Path(__file__).parent.parent.parent.parent.parent / "models"

@pytest.fixture
def hf_token():
    # If the env variable is set, great!
    token = os.getenv("HF_TOKEN", None)
    if token:
        return token
    
    # Otherwise, look for token.env outside of the scope of this project (to avoid leaking it accidentally)
    token_file = Path(__file__).parent.parent.parent.parent.parent.parent / "token.env"
    if token_file.exists():
        with token_file.open() as f:
            setting = f.readline()
            # expecting something like HF_TOKEN=....
            return setting.split("HF_TOKEN=")[-1]
    else:
        raise Exception("HF_TOKEN is required for import tests and we could not bootstrap it")

@pytest.fixture
def setenvvar(monkeypatch, model_cache, hf_token):
    with mock.patch.dict(os.environ, clear=True):
        envvars = {
            "HF_HOME": str(model_cache),
            "HF_TOKEN": hf_token
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)
        yield # This is the magical bit which restore the environment after 

@pytest.fixture
def client(setenvvar):
    from services.app import app
    return TestClient(app)

def test_list_models(client):
    from services.registry import REGISTRY

    response = client.get("/registry/")
    assert response.status_code == 200
    assert response.json() == list(REGISTRY.models.keys())

def test_list_provider_models_no_provider(client):
    response = client.get("/registry/" + "TESTTEST")
    assert response.status_code == 404
    assert response.json() == {"detail": "No such provider."}

def test_list_provider_models(client):
    from services.registry import REGISTRY

    model_id = list(REGISTRY.models.keys())[0]
    provider = model_id.split("/")[0]
    response = client.get("/registry/" + provider)
   
    expected = list()
    for model_id in REGISTRY.models.keys():
        parts = model_id.split("/")
        if parts[0] == provider:
            expected.append(parts[1])

    assert response.status_code == 200
    assert response.json() == expected

def test_get_model_no_model(client):
    from services.registry import REGISTRY

    model_id = list(REGISTRY.models.keys())[0]
    provider = model_id.split("/")[0]
    response = client.get("/registry/" + f"{provider}/TESTTEST")
    assert response.status_code == 404
    assert response.json() == {"detail": "No such model."}

def test_get_model(client):
    from services.registry import REGISTRY, ModelCard

    model_id = list(REGISTRY.models.keys())[0]
    response = client.get("/registry/" + model_id)
    
    assert response.status_code == 200
    assert ModelCard.model_validate(response.json()) == REGISTRY.models[model_id]

def test_import_model(client):
    from services.registry import REGISTRY, ImportModel, ModelCard

    model_id = list(REGISTRY.models.keys())[0]
    request = ImportModel(model_id=model_id, token=os.getenv("HF_TOKEN"))
    response = client.post("/registry", json=request.model_dump())
    
    assert response.status_code == 200
    assert ModelCard.model_validate(response.json()) == REGISTRY.models[model_id]

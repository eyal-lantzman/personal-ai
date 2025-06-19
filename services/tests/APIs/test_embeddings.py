import pytest
from unittest import mock
import os
from pathlib import Path
from fastapi.testclient import TestClient

@pytest.fixture()
def model_cache():
    return Path(__file__).parent.parent.parent.parent / "models"

@pytest.fixture()
def setenvvar(monkeypatch, model_cache):
    with mock.patch.dict(os.environ, clear=True):
        envvars = {
            "HF_HOME": str(model_cache),
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)
        yield # This is the magical bit which restore the environment after 

@pytest.fixture()
def client(setenvvar):
    from services.app import app
    return TestClient(app)

def test_create_embeddings_not_supported(client):
    from services.embeddings import CreateEmbeddingsRequest

    request = CreateEmbeddingsRequest(model="sentence-transformers/TESTTEST", input="Hi")
    response = client.post("/embeddings", json=request.model_dump())
    assert response.status_code == 404

def test_create_embeddings_single(client):
    from services.embeddings import CreateEmbeddingsRequest

    request = CreateEmbeddingsRequest(model="sentence-transformers/all-MiniLM-L6-v2", input="Hi")
    response = client.post("/embeddings", json=request.model_dump())
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1
    assert response.json()["data"][0]["index"] == 0
    assert response.json()["model"] == "sentence-transformers/all-MiniLM-L6-v2"

def test_create_embeddings_single_base64(client):
    from services.embeddings import CreateEmbeddingsRequest

    request = CreateEmbeddingsRequest(model="sentence-transformers/all-MiniLM-L6-v2", input="Hi", encoding_format="base64")
    response = client.post("/embeddings", json=request.model_dump())
    assert response.status_code == 200
    assert response.json() == {
        "data": [{
            "embedding": "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",
            "index": 0,
            "object": "embedding",
       }],
       "model": "sentence-transformers/all-MiniLM-L6-v2",
       "object": "list",
       "usage": {
           "prompt_tokens": 2,
           "total_tokens": 386,
       },
    }

def test_create_embeddings_multiple(client):
    from services.embeddings import CreateEmbeddingsRequest

    request = CreateEmbeddingsRequest(model="sentence-transformers/all-MiniLM-L6-v2", input=["Hi", "Bye"])
    response = client.post("/embeddings", json=request.model_dump())
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
    assert response.json()["data"][0]["index"] == 0
    assert response.json()["data"][1]["index"] == 1
    assert response.json()["data"][0]["embedding"] != response.json()["data"][1]["embedding"]
    assert response.json()["model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_create_embeddings_multiple_base64(client):
    from services.embeddings import CreateEmbeddingsRequest

    request = CreateEmbeddingsRequest(model="sentence-transformers/all-MiniLM-L6-v2", input=["Hi", "Bye"], encoding_format="base64")
    response = client.post("/embeddings", json=request.model_dump())
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
    assert response.json()["data"][0]["index"] == 0
    assert response.json()["data"][1]["index"] == 1
    assert response.json()["data"][0]["embedding"] != response.json()["data"][1]["embedding"]
    assert response.json()["model"] == "sentence-transformers/all-MiniLM-L6-v2"


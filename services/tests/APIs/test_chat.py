import pytest
from unittest import mock
import os
from pathlib import Path
import json
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


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

@pytest.fixture()
def async_client(client):
    return AsyncClient(
        transport=ASGITransport(app=client.app), base_url="http://test"
    )

def test_create_completion_not_supported(client):
    from services.chat import ChatCompletionRequest, Message

    request = ChatCompletionRequest(model="meta-llama/TESTTEST", messages=[Message(role="user", content="Hi")])
    response = client.post("/chat/completions", json=request.model_dump())
    assert response.status_code == 404

def test_create_completion_single(client):
    from services.chat import ChatCompletionRequest, Message

    request = ChatCompletionRequest(model="meta-llama/Llama-3.2-1B-Instruct", messages=[Message(role="user", content="Continue until 10: 1,2,4, ")])
    response = client.post("/chat/completions", json=request.model_dump())
    assert response.status_code == 200
    assert response.json()["model"] == request.model
    assert len(response.json()["choices"]) == 1
    assert response.json()["id"] is not None
    assert response.json()["created"] is not None
    assert response.json()["object"] == "chat.completion"

@pytest.mark.asyncio
async def test_create_completion_single_stream(async_client):
    from services.chat import ChatCompletionRequest, Message

    request = ChatCompletionRequest(model="meta-llama/Llama-3.2-1B-Instruct", messages=[Message(role="user", content="Continue until 10: 1,2,4, ")], stream=True)
    async with async_client:
        response = await async_client.post("/chat/completions", json=request.model_dump())
    assert response.status_code == 200
    if response.encoding is None:
        response.encoding = "utf-8"
    line_count = 0
    for line in response.iter_lines():
        if line:
            part = json.loads(line)
            assert part["model"] == request.model
            assert len(part["choices"]) == 1
            assert part["id"] is not None
            assert part["created"] is not None
            assert part["object"] == "chat.completion.chunk"
            line_count += 1
    assert line_count > 0

def test_create_completion_multiple(client):
    from services.chat import ChatCompletionRequest, Message

    request = ChatCompletionRequest(model="meta-llama/Llama-3.2-1B-Instruct", messages=[Message(role="user", content="Continue the sentence - mother, father, ")], n=2)
    response = client.post("/chat/completions", json=request.model_dump())
    assert response.status_code == 200
    assert response.json()["model"] == request.model
    assert len(response.json()["choices"]) == 2
    assert response.json()["id"] is not None
    assert response.json()["created"] is not None
    assert response.json()["object"] == "chat.completion"

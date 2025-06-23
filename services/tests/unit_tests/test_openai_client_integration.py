import pytest
from unittest import mock
import os
from pathlib import Path
from openai import OpenAI
import logging
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)

@pytest.fixture
def model_cache():
    return Path(__file__).parent.parent.parent.parent / "models"

@pytest.fixture
def setenvvar(monkeypatch, model_cache):
    with mock.patch.dict(os.environ, clear=True):
        envvars = {
            "HF_HOME": str(model_cache),
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)
        yield # This is the magical bit which restore the environment after 

@pytest.fixture
def testserver(setenvvar):
    from services.app import app
    return TestClient(app)

@pytest.fixture
def client(testserver):
    return OpenAI(
        base_url=testserver.base_url, 
        api_key="my-key",
        http_client=testserver
    )

def test_openai_completions(client):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
        {"role": "system", "content": "Talk like a pirate."},
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
    )

    logger.info(response)
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.role == "assistant"

def test_openai_embeddings(client):
    input = "Embed this"
    response = client.embeddings.create(
        encoding_format="float",
        model="sentence-transformers/all-MiniLM-L6-v2",
        input=input
    )

    logger.info(response)
    assert len(response.data[0].embedding) == 384
    assert response.usage.total_tokens == 384 + len(input)

def test_openai_models(client):
    models_response = client.models.list()

    logger.info(models_response)
    assert len(models_response.data) > 0 

    model_response = client.models.retrieve(models_response.data[0].id)

    logger.info(model_response)
    assert model_response.owned_by == "me"

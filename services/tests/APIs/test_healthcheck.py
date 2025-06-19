import pytest
from unittest import mock
import os
from pathlib import Path
from unittest.mock import patch
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

def test_healthy(client):
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_unhealthy(client):
    from services.healthcheck import REGISTRY
    with patch.dict(REGISTRY.models, dict(), clear=True):
        response = client.get("/health/")
        assert response.status_code == 500
        assert response.json() == {"detail": "No Models"}

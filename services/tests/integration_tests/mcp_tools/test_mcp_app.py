import pytest
import os
import importlib
from unittest import mock
from fastapi.testclient import TestClient
from pathlib import Path
from fastmcp import Client
import services.app

@pytest.fixture()
def model_cache():
    return Path(__file__).parent.parent.parent.parent.parent / "models"

@pytest.fixture()
def setenvvar(monkeypatch, model_cache):
    with mock.patch.dict(os.environ, clear=False):
        envvars = {
            "HF_HOME": str(model_cache),
        }
        for k, v in envvars.items():
            monkeypatch.setenv(k, v)
        yield # This is the magical bit which restore the environment after 

@pytest.fixture
@pytest.mark.asyncio
async def streaming_http_server(monkeypatch, setenvvar):
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    importlib.reload(services.app)
    yield TestClient(services.app.app)

@pytest.mark.asyncio
async def test_mcp_app_on_streamable_http(streaming_http_server):
    with streaming_http_server:
        response = streaming_http_server.get("/mcp/mcp")
        assert response.status_code == 406

        response = streaming_http_server.get("/mcp/sse/")
        assert response.status_code == 404

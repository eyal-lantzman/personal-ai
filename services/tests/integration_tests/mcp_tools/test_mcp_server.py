import pytest
import logging
from urllib.parse import urlparse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import tempfile
from services.mcp_tools.mcp_server import MCPProjectServer

logger = logging.getLogger(__name__)

@pytest.fixture
def client_url() -> str:
    return "http://127.0.0.1:9876/mcp"

def wait_for_connection(client_url):
    import requests
    import time
    server_endpoint = f"{client_url.scheme}://{client_url.hostname}:{client_url.port}{client_url.path}"
    for i in range(10):
        try:
            response = requests.get(server_endpoint)
            response.raise_for_status()
            return True
        except requests.exceptions.ConnectionError as e:
            time.sleep(1)
        except requests.exceptions.HTTPError  as e:
            # When getting http errors, it means the server is running!
            return True

@pytest.fixture
@pytest.mark.asyncio
async def server(client_url):
    url = urlparse(client_url)
    server = MCPProjectServer(root=tempfile.gettempdir(), enable_fs=True, log_level="DEBUG", host=url.hostname, port=url.port)
    server.start_uvicorn(transport="streamable-http")
    wait_for_connection(url)
    yield server
    server.stop_uvicorn()

@pytest.mark.asyncio
async def test_mcp_langchain_client_integration(server, client_url):
    client = MultiServerMCPClient({
        "test": {
            "url": client_url,
            "transport": "streamable_http",
        }
    })

    async with client.session("test") as session:
        tools = await load_mcp_tools(session)

        assert [x.name for x in tools] == [
            "fs_list_dir",
            "fs_read_text",
            "fs_read_bytes",
            "fs_write_to_file",
            "fs_create_file",
            "fs_create_folder", 
            "fs_is_folder_empty",
            "fs_search_folder"
        ]
import pytest
from fastmcp import Client
from urllib.parse import urlparse
import tempfile
from services.mcp_tools.mcp_server import MCPProjectServer

@pytest.fixture
def client_url():
    return urlparse("http://127.0.0.1:9876/mcp")


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
def server(client_url):
    server = MCPProjectServer(tempfile.gettempdir(), enable_fs=True, log_level="DEBUG", host=client_url.hostname, port=client_url.port)
    server.start_uvicorn(transport="streamable-http") 
    wait_for_connection(client_url)
    yield server
    server.stop_uvicorn()

@pytest.mark.asyncio
async def test_all_tools(client_url, server):
    async with Client(client_url.geturl()) as client:
        tools = await client.list_tools()
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
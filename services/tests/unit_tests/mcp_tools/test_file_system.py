import pytest
import os
import json
from pathlib import Path
from fastmcp import Client
from fastmcp.exceptions import ToolError
from services.mcp_tools.file_system import FileSystem

TEST_FOLDER = Path(__file__).parent / "test_resources"
OTHER_ROOT = Path(__file__).parent.parent / "APIs"

@pytest.fixture
def test_subject():
    return FileSystem(TEST_FOLDER)

def test_validate_relative_path(test_subject:FileSystem):
    assert test_subject.validate_relative_path(str(TEST_FOLDER / "folder")) == Path(TEST_FOLDER / "folder")
    assert test_subject.validate_relative_path(str(TEST_FOLDER / "file.txt")) == Path(TEST_FOLDER / "file.txt")
    assert test_subject.validate_relative_path(str(TEST_FOLDER / "folder/file.txt")) == Path(TEST_FOLDER / "folder/file.txt")
    assert test_subject.validate_relative_path(str(TEST_FOLDER)) == Path(TEST_FOLDER)
    
    with pytest.raises(ValueError):
        test_subject.validate_relative_path(str(OTHER_ROOT))

    with pytest.raises(ValueError):
        test_subject.validate_relative_path(str(TEST_FOLDER)[:-1]) 

@pytest.mark.asyncio
async def test_all_tools(test_subject:FileSystem):
    async with Client(test_subject.mcp) as client:
        result = await client.list_tools()
        assert set([t.name for t in result]) == set([
            "list_dir",
            "read_text",
            "read_bytes",
            "write_to_file",
            "create_file",
            "create_folder",
            "is_folder_empty",
            "search_folder"
        ])

@pytest.mark.asyncio
async def test_list_dir(test_subject:FileSystem):
    async with Client(test_subject.mcp) as client:
        result = await client.call_tool("list_dir", {"relative_path": str(TEST_FOLDER)})
        assert len(result) == 1
        assert json.loads(result[0].text) == ["file1", "file2", "file3"]

        result = await client.call_tool("list_dir", {"relative_path": "."})
        assert len(result) == 1
        assert json.loads(result[0].text) == ["file1", "file2", "file3"]

        result = await client.call_tool("list_dir", {"relative_path": ".\\"})
        assert len(result) == 1
        assert json.loads(result[0].text) == ["file1", "file2", "file3"]

        result = await client.call_tool("list_dir", {"relative_path": "file3"})
        assert len(result) == 0

        with pytest.raises(ToolError):
            await client.call_tool("list_dir", {"relative_path": "..\\"})


@pytest.mark.asyncio
async def test_read_text(test_subject:FileSystem):
    async with Client(test_subject.mcp) as client:
        result = await client.call_tool("read_text", {"file_name": "file1", "relative_path": str(TEST_FOLDER)})
        assert len(result) == 1
        assert result[0].text == "file 1 text"

@pytest.mark.asyncio
async def test_read_bytes(test_subject:FileSystem):
    async with Client(test_subject.mcp) as client:
        result = await client.call_tool("read_bytes", {"file_name": "file1", "relative_path": str(TEST_FOLDER)})
        assert len(result) == 1
        assert result[0].text == '"file 1 text"'

@pytest.mark.asyncio
async def test_write_to_file(test_subject:FileSystem):
    async with Client(test_subject.mcp) as client:
        content = "hi world"
        result = await client.call_tool("write_to_file", {"file_name": "file2", "relative_path": str(TEST_FOLDER), "content": content})
        assert len(result) == 1
        assert result[0].text == str(len(content))

@pytest.mark.asyncio
async def test_create_file(test_subject:FileSystem):
    async with Client(test_subject.mcp) as client:
        file_name = "file___" + str(os.getpid())
        result = await client.call_tool("create_file", {"file_name": file_name, "relative_path": str(TEST_FOLDER)})
        assert len(result) == 1
        assert result[0].text == str(TEST_FOLDER / file_name)
        os.remove(result[0].text)

@pytest.mark.asyncio
async def test_is_folder_empty(test_subject:FileSystem):
    async with Client(test_subject.mcp) as client:

        result = await client.call_tool("is_folder_empty", {"relative_path": str(TEST_FOLDER)})
        assert len(result) == 1
        assert result[0].text == "false"

        result = await client.call_tool("is_folder_empty", {"relative_path": str(TEST_FOLDER / "file3")})
        assert len(result) == 1
        assert result[0].text == "true"

@pytest.mark.asyncio
async def test_search_folder(test_subject:FileSystem):
    async with Client(test_subject.mcp) as client:

        result = await client.call_tool("search_folder", {"relative_path": str(TEST_FOLDER), "pattern": "file*"})
        assert len(result) == 1
        assert json.loads(result[0].text) == ["file1", "file2", "file3"]

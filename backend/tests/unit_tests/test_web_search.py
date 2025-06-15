import pytest
from unittest.mock import patch, MagicMock, call
from typing import Generator
from agent.web_search import search_tool, RATE_CALLS, RATE_PER_SECONDS
from ratelimit.exception import RateLimitException
import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def ddg_service_mock() -> Generator[None, MagicMock, None]:
    patcher = patch("langchain_community.tools.DuckDuckGoSearchResults")
    yield patcher.start()
    patcher.stop()

@pytest.fixture
def happy_path_value() -> dict:
    return {
        "snippet": "test snippet",
        "title": "test title",
        "link": "http://test.link"
    }

@pytest.mark.asyncio
async def test_search_tool_happy_path(ddg_service_mock:MagicMock, happy_path_value:dict):
    # Arrange
    ddg_service_mock.return_value = MagicMock()
    invoke = ddg_service_mock.return_value.invoke
    invoke.return_value=happy_path_value

    # Act
    result = await search_tool.ainvoke({"query": "my query", "num_results": 1})

    # Assert
    ddg_service_mock.assert_called_once_with(output_format="json", backend="text", region="wt-wt", num_results=1)
    invoke.assert_called_once_with("my query")
    assert result == happy_path_value

@pytest.mark.asyncio
async def test_search_tool_sleep_and_retry(ddg_service_mock:MagicMock, happy_path_value:dict):
    # Arrange
    ddg_service_mock.return_value = MagicMock()
    invoke:MagicMock = ddg_service_mock.return_value.invoke
    invoke.side_effect = [RateLimitException("boom", 0), happy_path_value]

    # Act
    result = await search_tool.ainvoke({"query": "my query", "num_results": 1})

    # Assert
    ddg_service_mock.assert_has_calls([
        call(output_format="json", backend="text", region="wt-wt", num_results=1),
        call().invoke('my query'),
        call(output_format="json", backend="text", region="wt-wt", num_results=1),
        call().invoke('my query'),
        ])
    assert result == happy_path_value

@pytest.mark.asyncio
async def test_search_tool_rate_limit(ddg_service_mock:MagicMock, happy_path_value:dict):
    import time
    from asyncio import gather
    # Arrange
    ddg_service_mock.return_value = MagicMock()
    invoke = ddg_service_mock.return_value.invoke

    call_times = []
    def clock_it(x):
        call_times.append(time.monotonic())
        return happy_path_value
    invoke.side_effect = clock_it

    tasks = []
    for i in range(RATE_CALLS + 1):
        tasks.append(search_tool.ainvoke({"query": f"my query{i}", "num_results": 1}))

    # Act
    result = await gather(*tasks)
    logger.debug(call_times)

    # Assert
    # RATE_CALLS + 1 is too many calls, should throttle (with some 10% margin error)
    assert max(call_times) - min(call_times) >= RATE_PER_SECONDS * .9 
    assert result == [happy_path_value] * (RATE_CALLS + 1)
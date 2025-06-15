import pytest
import logging
from typing import Generator
from agent.memory_recollection import MemoryManager, WebSearchMemory

logger = logging.getLogger(__name__)

TEST_RUNS = 3

@pytest.fixture
def namespace() -> tuple[str,...]:
    import random
    return ("test", str(random.randint(0, 100)))

@pytest.fixture
def memory(namespace) -> Generator[None,MemoryManager,None]:
    manager = MemoryManager(namespace)
    yield manager

@pytest.fixture
def example1() -> WebSearchMemory:
    return WebSearchMemory(
       prefix=1,
       search_query="How should do I water my plants?",
       sources_gathered=["http://source1"],
       web_research_result=["http://nature/watering/plants"],
    )

@pytest.fixture
def example2() -> WebSearchMemory:
    import uuid
    return WebSearchMemory(
        id=str(uuid.uuid4()),
        prefix=2,
        search_query="what is the spec for a quantum machine?",
        sources_gathered=["http://source2"],
        web_research_result=["http://quantum/compute/specs"],
    )

@pytest.mark.parametrize("run", range(TEST_RUNS))
def test_memory_interaction(run, memory, example1, example2):
    # Recoll example1 that is not in the memory
    result = memory.recoll_search(1, example1.search_query)
    assert result.search_query == example1.search_query
    assert result.sources_gathered == []
    assert result.web_research_results == []

    # Remember example1 and try recoll again
    id1 = memory.remember_search(example1)
    assert id1 is not None
    result = memory.recoll_search(2, example1.search_query)
    assert result.search_query == example1.search_query
    assert result.sources_gathered == example1.sources_gathered
    assert result.web_research_results == example1.web_research_result

    # Recoll example2 that is not in the memory
    result = memory.recoll_search(3, example2.search_query)
    assert result.search_query == example2.search_query
    assert result.sources_gathered == []
    assert result.web_research_results == []

    # Remember example2 and try recoll again
    id2 = memory.remember_search(example2)
    assert id2 is not None
    result = memory.recoll_search(4, example2.search_query)
    assert result.search_query == example2.search_query
    assert result.sources_gathered == example2.sources_gathered
    assert result.web_research_results == example2.web_research_result

    # Forget example2 and search again
    assert memory.forget_search(id2)
    result = memory.recoll_search(5, example2.search_query)
    assert result.search_query == example2.search_query
    assert result.sources_gathered == []
    assert result.web_research_results == []

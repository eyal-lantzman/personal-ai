import pytest
from unittest.mock import Mock, patch
from langgraph.types import Send
from langchain_core.messages import BaseMessage, AIMessage
from agent.configuration import Configuration
from agent.state import SearchTaskInput, RecollTaskInput, InitialState, IntermediateState, FinalState
from agent.tools_and_schemas import AggregatedSearchResults, WebSearchMemory
from agent.graph import generate_query, reconcile_research, recollection, web_research, reflection, evaluate_research, finalize_answer, continue_to_parallel_recollection, continue_to_parallel_web_research, graph


# Fixtures for mocking dependencies
@pytest.fixture
def configuration():
    return Configuration(
            small_model="sml",
            medium_model="med",
            large_model="lrg",
            max_search_results=5,
            max_research_loops=3,
            number_of_initial_queries=2)

@pytest.fixture
def mock_from_runnable_config(configuration):
    patcher = patch("agent.configuration.Configuration.from_runnable_config")
    mock = patcher.start()
    mock.return_value = configuration
    yield mock
    patcher.stop()

@pytest.fixture
def mock_simple_search():
    patcher = patch("agent.graph.simple_search")
    mock = patcher.start()
    mock.return_value = Mock(
        search_query="mock_search_query",
        sources_gathered=["source1", "source2"],
        web_research_results=["result1", "result2"]
    )
    yield mock
    patcher.stop()

@pytest.fixture
def mock_identify_region():
    patcher = patch("agent.graph.identify_region")
    mock = patcher.start()
    mock.return_value = "eu-us"
    yield mock
    patcher.stop()

@pytest.fixture
def mock_recoll_search():
    patcher = patch("agent.memory_recollection.MemoryManager.recoll_search")
    mock = patcher.start()
    yield mock
    patcher.stop()

@pytest.fixture
def mock_remember_search():
    patcher = patch("agent.memory_recollection.MemoryManager.remember_search")
    mock = patcher.start()
    yield mock
    patcher.stop()

@pytest.fixture
def mock_get_chat():
    patcher = patch("agent.graph.get_chat")
    mock = patcher.start()
    # invoke().content
    mock.return_value.invoke.return_value = BaseMessage(content="<think>mock thinking</think>\n\nmock answer", type="assistant")
    # with_structured_output(Reflection).invoke()
    mock.return_value.with_structured_output.return_value.invoke.return_value.is_sufficient = False
    mock.return_value.with_structured_output.return_value.invoke.return_value.knowledge_gap = "mock gap"
    mock.return_value.with_structured_output.return_value.invoke.return_value.follow_up_queries = ["follow1", "follow2"]
    # with_structured_output(SearchQueryList).invoke()
    mock.return_value.with_structured_output.return_value.invoke.return_value.query = ["query1", "query2"]
    mock.return_value.with_structured_output.return_value.invoke.return_value.rationale = "mock rationale"
    yield mock
    patcher.stop()

# Test nodes
def test_generate_query(configuration, mock_from_runnable_config, mock_get_chat, mock_identify_region):
    """Test the generate_query node"""
    # Arrange
    config = Mock()
    input = InitialState(
        research_query="test research query",
        messages=["initial message"]
    )

    # Act
    output = generate_query(input, config)
    
    # Assert
    assert output == IntermediateState(
        initial_search_query_count=configuration.number_of_initial_queries,
        reasoning_model=configuration.medium_model,
        research_query="test research query",
        search_query=frozenset(["query1", "query2"]),
        is_knowledge_sufficient=False,
        search_query_result_limit=configuration.max_search_results,
        search_region="eu-us",
        max_research_loops=configuration.max_research_loops)
    mock_get_chat.assert_called_once_with(model=configuration.medium_model, temperature=1.0, max_retries=2)
    mock_identify_region.assert_called_once_with(input["research_query"])

@pytest.mark.parametrize("input_research_loop_count", (None, 0, 1))
def test_reconcile_research(input_research_loop_count):
    """Test the reconcile_research node"""
    # Arrange
    config = Mock()
    input = IntermediateState(
        search_query=frozenset({"query1", "query2"}),
        research_loop_count = input_research_loop_count,
        follow_up_queries=["follow1", "query1"],
        number_of_ran_queries = 2
    )

    # Act
    output = reconcile_research(input, config)
    
    # Assert
    assert output == IntermediateState(
        search_query=frozenset(["query1", "query2", "follow1"]),
        follow_up_queries=[],
        research_loop_count= 0 if input_research_loop_count is None else 1,
        number_of_ran_queries=input["number_of_ran_queries"]
    )

def test_continue_to_parallel_recollection():
    """Test the continue_to_parallel_recollection node"""
    # Arrange
    input = IntermediateState(
        search_query=frozenset({"query1", "query2"}),
        search_query_result_limit = 3
    )
    
    # Act
    output = continue_to_parallel_recollection(input)
    
    # Assert
    assert len(output) == len(input["search_query"])
    assert output == [
        Send(node="recollection", 
             arg=RecollTaskInput(
                    search_query = "query1", 
                    id = 0, 
                    search_query_result_limit = input["search_query_result_limit"]
                )
        ),
        Send(node="recollection", 
             arg=RecollTaskInput(
                    search_query = "query2", 
                    id = 1, 
                    search_query_result_limit = input["search_query_result_limit"]
                )
        )
    ]

def test_continue_to_parallel_web_research():
    """Test the continue_to_parallel_web_research node"""
    # Arrange
    input = IntermediateState(
        search_query=frozenset({"query1", "query2", "query3"}),
        search_query_result_limit = 4
    )
    
    # Act
    output = continue_to_parallel_web_research(input)
    
    # Assert
    assert len(output) == len(input["search_query"])
    assert output == [
        Send(node="web_research", 
             arg=SearchTaskInput(
                    search_query = "query1", 
                    id = 0, 
                    search_query_result_limit = input["search_query_result_limit"]
                )
        ),
        Send(node="web_research", 
             arg=SearchTaskInput(
                    search_query = "query2", 
                    id = 1, 
                    search_query_result_limit = input["search_query_result_limit"]
                )
        ),
        Send(node="web_research", 
             arg=SearchTaskInput(
                    search_query = "query3", 
                    id = 2, 
                    search_query_result_limit = input["search_query_result_limit"]
                )
        )
    ]

def test_web_research(configuration, mock_simple_search, mock_remember_search):
    """Test the web_research node"""
    # Arrange
    config = Mock()
    input = SearchTaskInput(
        id=3,
        search_query="web research query",
        search_query_result_limit=configuration.max_search_results
    )

    # Act
    output = web_research(input, config)
    
    # Assert
    assert output.goto == "reflection"
    assert output.update == IntermediateState(
        search_query="mock_search_query",
        sources_gathered=frozenset(["source1", "source2"]),
        web_research_result=frozenset(["result1", "result2"])
    )

    mock_simple_search.assert_called_once_with(
        input["id"],
        "web research query",
        num_results=configuration.max_search_results,
        region=None
    )
    mock_remember_search.assert_called_once_with(
        WebSearchMemory(prefix=input["id"], 
                        search_query='mock_search_query', 
                        web_research_result=['result1', 'result2'],
                        sources_gathered=['source1', 'source2']))

@pytest.mark.parametrize("has_memories", (True, False))
def test_recollection(configuration, mock_recoll_search, has_memories):
    """Test the recollection node"""
    # Arrange
    config = Mock()
    input = RecollTaskInput(
        id=2,
        search_query="recollection query",
        search_query_result_limit=configuration.max_search_results
    )
    if has_memories:
        mock_recoll_search.return_value = AggregatedSearchResults(
            search_query="mock_memory_manager",
            sources_gathered=["source4", "source5"],
            web_research_results=["result3", "result6"]
        )
    else:
        mock_recoll_search.return_value = None


    # Act
    output = recollection(input, config)
    
    # Assert
    assert output.goto == "reflection"
    if has_memories:
        assert output.update == IntermediateState(
            search_query="mock_memory_manager",
            sources_gathered=frozenset(["source4", "source5"]),
            web_research_result=frozenset(["result3", "result6"])
        )
    else:
        assert output.update is None

    mock_recoll_search.assert_called_once_with(
        input["id"],
        "recollection query",
        num_results=configuration.max_search_results
    )

def test_reflection(configuration, mock_from_runnable_config, mock_get_chat):
    """Test the reflection node"""
    # Arrange
    config = Mock()
    input = IntermediateState(
        web_research_result=frozenset(["reflection result 1", "reflection result 2"]),
        research_query="reflection topic",
        search_query=frozenset(["q1", "q2", "q3"])
    )
    
    # Act
    output = reflection(input, config)
    
    # Assert
    assert output == IntermediateState(
        is_knowledge_sufficient=False,
        follow_up_queries=sorted(["follow1", "follow2"]),
        number_of_ran_queries=len(input["search_query"])
    )
    mock_get_chat.assert_called_once_with( model=configuration.large_model, temperature=1.0, max_retries=2,)

@pytest.mark.parametrize("research_loop_count", (0, 10))
@pytest.mark.parametrize("is_knowledge_sufficient", (True, False))
def test_evaluate_research(configuration, mock_from_runnable_config, research_loop_count, is_knowledge_sufficient):
    """Test the evaluate_research node"""
    # Arrange
    config = Mock()
    input = IntermediateState(
        is_knowledge_sufficient=is_knowledge_sufficient,
        research_loop_count=research_loop_count,
        max_research_loops=configuration.max_research_loops
    )

    # Act
    output = evaluate_research(input, config)
    
    # Assert
    if is_knowledge_sufficient or research_loop_count >= configuration.max_research_loops:
        assert output == "finalize_answer"
    else:
        assert output == "reconcile_research"

def test_finalize_answer(configuration, mock_from_runnable_config, mock_get_chat):
    """Test the finalize_answer node"""
    # Arrange
    config = Mock()
    input = IntermediateState(
        reasoning_model=configuration.small_model,
        web_research_result=frozenset(["final result 1", "final result 2"]),
        research_query="final topic",
        sources_gathered=frozenset(["source2", "source3", "source1"])
    )

    # Act
    output = finalize_answer(input, config)
    
    # Assert
    assert output == FinalState(
        summary="mock answer",
        sources=list(sorted(input["sources_gathered"])),
        messages=[AIMessage(content="Summary:\n---------------\nmock answer\n\nSources:\n---------------\n source1\n source2\n source3", additional_kwargs={}, response_metadata={})]
    )
    mock_get_chat.assert_called_once_with(model=configuration.small_model, temperature=0, max_retries=2)

# Test transitions
@pytest.mark.parametrize("max_research_loops", (1, 2, 3, 4))
def test_end_to_end(configuration, 
                    max_research_loops,
                    mock_from_runnable_config, 
                    mock_get_chat, 
                    mock_identify_region,
                    mock_recoll_search, 
                    mock_remember_search, 
                    mock_simple_search):
    # Arrange
    configuration.max_research_loops = max_research_loops
    configuration.max_search_results = 1
    configuration.number_of_initial_queries = 1
    input = InitialState(
        research_query="test research query",
        messages=["initial message"]
    )
    mock_recoll_search.return_value = AggregatedSearchResults(
            search_query="mock_memory_manager",
            sources_gathered=["source4", "source5"],
            web_research_results=["result3", "result6"]
    )

    # Act
    output = graph.invoke(input, configuration.model_dump())

    # Assert
    assert output["summary"]  == "mock answer"
    assert output["sources"]  == ["source1", "source2", "source4", "source5"]
    assert len(output["messages"]) == 2
    assert output["messages"][0].content == input["messages"][0]
    assert output["messages"][1].content == "Summary:\n---------------\nmock answer\n\nSources:\n---------------\n source1\n source2\n source4\n source5"
    assert mock_identify_region.call_count == 1
    assert mock_recoll_search.call_count == 2 + (6 * (configuration.max_research_loops - 1))
    assert mock_remember_search.call_count == 2 + (6 * (configuration.max_research_loops - 1))
    assert mock_simple_search.call_count == 2 + (6 * (configuration.max_research_loops - 1))
    assert mock_get_chat.call_count == 3 + (configuration.max_research_loops - 1)
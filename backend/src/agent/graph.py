from typing import Literal
from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send, Command
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from agent.state import (
    InitialState,
    IntermediateState,
    FinalState,
    OverallState,
    SearchTaskInput,
    SearchTaskOutput,
    RecollTaskInput,
    RecollTaskOutput,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.openai_compatible_models import get_chat, remove_thinking
from agent.web_search import simple_search, identify_region
from agent.memory_recollection import MemoryManager, WebSearchMemory
from agent.utils import (
    get_research_topic,
)

load_dotenv()

from langchain.globals import set_debug
set_debug(True)

import logging
logger = logging.getLogger(__name__)

memory_manager = MemoryManager()

# Nodes
def generate_query(state: InitialState, config: RunnableConfig) -> IntermediateState:
    configurable = Configuration.from_runnable_config(config)

    logger.debug("------------->generate_query: %s", str(state))

    # init Generator Model
    llm = get_chat(
        model=configurable.medium_model,
        temperature=1.0,
        max_retries=2,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    research_query = state.get("research_query") or get_research_topic(state["messages"])
    current_date = get_current_date()
    initial_search_query_count = state.get("initial_search_query_count", configurable.number_of_initial_queries)
    
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=research_query,
        number_queries=initial_search_query_count,
    )

    # TODO: parallelize 

    # Generate the search queries
    result:SearchQueryList = structured_llm.invoke(formatted_prompt)

    # Identify the search region
    search_region  = identify_region(research_query)

    return_state = IntermediateState(
        research_query = research_query,
        search_query = result.query,
        is_sufficient = False,
        initial_search_query_count = initial_search_query_count,
        search_query_result_limit = state.get("search_query_result_limit", configurable.max_search_results),
        max_research_loops = state.get("max_research_loops", configurable.max_research_loops),
        reasoning_model = state.get("reasoning_model", configurable.medium_model),
        search_region = search_region
    )
    
    logger.debug("generate_query-------------> %s", str(return_state))
    return return_state

def reconcile_research(state: IntermediateState, config: RunnableConfig) -> IntermediateState:
    search_queries = state["search_query"]
    if state.get("follow_up_queries"):
        search_queries.union(state.get("follow_up_queries"))
    
    return_state = IntermediateState(
        search_query=search_queries,
        follow_up_queries = set(),
        research_loop_count = 0 if state.get("research_loop_count", 0) == 0 is None else 1, #Auto increment
        number_of_ran_queries = state.get("number_of_ran_queries", 0)
    )
    
    logger.debug("reconcile_research-------------> %s", str(return_state))
    return return_state

def continue_to_parallel_recollection(state: IntermediateState):
    logger.debug("------------->continue_to_parallel_recollection: %s", str(state))

    return_state = [ 
            Send(
                node="recollection", 
                arg=RecollTaskInput(
                    search_query = search_query, 
                    id = int(idx), 
                    search_query_result_limit = state["search_query_result_limit"]
                )
            )
        for idx, search_query in enumerate(state["search_query"])
    ]

    logger.debug("continue_to_parallel_recollection-------------> %s", str(return_state))
    return return_state

def continue_to_parallel_web_research(state: IntermediateState):
    logger.debug("------------->continue_to_parallel_web_research: %s", str(state))

    return_state = [ 
            Send(
                node="web_research", 
                arg=SearchTaskInput(
                    search_query = search_query, 
                    id = int(idx), 
                    search_query_result_limit = state["search_query_result_limit"]
                )
            )
        for idx, search_query in enumerate(state["search_query"])
    ]

    logger.debug("continue_to_parallel_web_research-------------> %s", str(return_state))
    return return_state

def web_research(state: SearchTaskInput, config: RunnableConfig) -> Command[Literal["reflection"]]:
    logger.debug("------------->web_research: %s ", str(state))
    
    results = simple_search(
        state["id"], 
        state["search_query"], 
        num_results=state.get("search_query_result_limit"), 
        region=state.get("search_region", None))
    
    memory_manager.remember_search(WebSearchMemory(
        prefix=state["id"],
        search_query=results.search_query, 
        sources_gathered=results.sources_gathered, 
        web_research_result=results.web_research_results))

    return_state = SearchTaskOutput(
        search_query=results.search_query, 
        sources_gathered=results.sources_gathered, 
         web_research_result=results.web_research_results
    )

    command = Command(
        update=IntermediateState(
            search_query=return_state["search_query"],
            web_research_result=frozenset(return_state["web_research_result"]),
            sources_gathered=frozenset(return_state["sources_gathered"])), 
        goto="reflection")

    logger.debug("web_research-------------> %s", str(command))
    return command

def recollection(state: RecollTaskInput, config: RunnableConfig) -> Command[Literal["reflection"]]:
    logger.debug("------------->recollection: %s ", str(state))

    results = memory_manager.recoll_search(
        state["id"], 
        state["search_query"], 
        num_results=state.get("search_query_result_limit")
    )
    if results and results.web_research_results:
        return_state = RecollTaskOutput(
            search_query=results.search_query, 
            sources_gathered=results.sources_gathered, 
            web_research_result=results.web_research_results
        )
        command = Command(
            update=IntermediateState(
                search_query=return_state["search_query"],
                web_research_result=frozenset(return_state["web_research_result"]),
                sources_gathered=frozenset(return_state["sources_gathered"])), 
            goto="reflection")
    else:
        command = Command(goto="reflection")

    logger.debug("recollection-------------> %s", str(command))
    return command

def reflection(state: IntermediateState, config: RunnableConfig) -> IntermediateState:
    configurable = Configuration.from_runnable_config(config)
    logger.debug("------------->reflection: %s", str(state))

    reasoning_model = state.get("reasoning_model", configurable.large_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = get_chat(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
    )
    result:Reflection = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return_state = IntermediateState(
        is_knowledge_sufficient = result.is_sufficient,
        follow_up_queries = set(result.follow_up_queries),
        number_of_ran_queries = len(state["search_query"]),
    )

    logger.debug("reflection-------------> %s", str(return_state))
    return return_state


def evaluate_research(
    state: IntermediateState,
    config: RunnableConfig,
) -> IntermediateState:
    configurable = Configuration.from_runnable_config(config)

    logger.debug("------------->evaluate_research: %s", str(state))

    max_research_loops = state.get("max_research_loops")
   
    if state.get("is_sufficient", False) or state.get("research_loop_count") >= max_research_loops:
        return_state = "finalize_answer"
    else:
        return_state = "reconcile_research"

    logger.debug("evaluate_research-------------> %s", str(return_state))
    return return_state

def finalize_answer(state: IntermediateState, config: RunnableConfig) -> FinalState:
    configurable = Configuration.from_runnable_config(config)

    logger.debug("------------->finalize_answer: %s", str(state))

    reasoning_model = state.get("reasoning_model")

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=state["research_query"],
        summaries="\n---\n\n".join(state["web_research_result"]),
        sources="\n---\n\n".join(state["sources_gathered"]),
    )

    # init Reasoning Model
    llm = get_chat(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
    )
    result = llm.invoke(formatted_prompt)
    remove_thinking(result)

    final_content = "Summary:\n---------------\n" + result.content + "\n\nSources:\n---------------\n " + "\n ".join(state["sources_gathered"])
    return_state = FinalState(
        summary = result.content, 
        sources = state["sources_gathered"],
        messages = [AIMessage(content=final_content)],
    )

    logger.debug("finalize_answer-------------> %s", str(return_state))
    return return_state


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration, input=InitialState, output=FinalState)
# Set the entrypoint as `generate_query`
# This means that this node is the first one called

builder.add_edge(START, "generate_query")
builder.add_node("generate_query", generate_query)

# From from Generate to reconcile
builder.add_edge("generate_query", "reconcile_research")
builder.add_node("reconcile_research",reconcile_research)

# In parallel, try to search and recoll from existing memories
builder.add_conditional_edges(
    "reconcile_research", continue_to_parallel_recollection, ["recollection"])
builder.add_conditional_edges(
    "reconcile_research", continue_to_parallel_web_research, ["web_research"])

# Move from recollection to reflection
builder.add_node("recollection", recollection)

# Move from web_searc to reflection
builder.add_node("web_research", web_research)

# Move from reflection to finish or back to reconciliation
builder.add_node("reflection", reflection)
builder.add_conditional_edges(
    "reflection", evaluate_research, ["reconcile_research", "finalize_answer"]
)

builder.add_node("finalize_answer", finalize_answer)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Union, Optional

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated

from langchain_core.messages import MessageLikeRepresentation
from agent.openai_compatible_models import remove_thinking

Messages = Union[list[MessageLikeRepresentation], MessageLikeRepresentation]

def add_cleaned_messages(
    left: Messages,
    right: Messages,
    *,
    format:str = "langchain-openai",
) -> Messages:
    merged = add_messages(left, right)
    for message in merged:
        remove_thinking(message)
    return merged

def merge_results(
    left: dict,
    right: dict,
) -> dict:
    assert len(set(left.keys()).intersection(right.keys())) == 0
    merged = {**left, **right}
    return merged

def merge_sets(
    left: set,
    right: set,
) -> set:
    if not left:
        left_set = set()
    elif isinstance(left, frozenset) or isinstance(left, list):
        left_set = set(left)
    elif isinstance(left, set):
        left_set = left
    else:
        left_set = set([left])

    if not right:
        right_set = set()
    elif isinstance(right, frozenset) or isinstance(right, list):
        right_set = set(right)
    elif isinstance(right, set):
        right_set = right
    else:
        right_set = set([right])

    result = frozenset(left_set | right_set)
    return result

class BaseState(TypedDict):
    messages: Annotated[list, add_cleaned_messages]

class InitialState(BaseState):
    research_query: Optional[str]
    initial_search_query_count: Optional[int]
    search_query_result_limit: Optional[int]
    max_research_loops: Optional[int]
    reasoning_model: Optional[str]

class FinalState(BaseState):
    summary: str
    sources: list

class IntermediateState(InitialState):
    # Reflection
    research_loop_count: Annotated[int, operator.add]
    is_knowledge_sufficient : bool
    follow_up_queries : list
    number_of_ran_queries : int
    # Search
    search_region: Optional[str]
    search_query: Annotated[frozenset, merge_sets]
    web_research_result: Annotated[frozenset, merge_sets]
    sources_gathered: Annotated[frozenset, merge_sets]


class OverallState(InitialState, IntermediateState, FinalState):
    pass

class Query(TypedDict):
    query: str
    rationale: str

class SearchTaskInput(TypedDict):
    search_query: str
    id: int
    search_query_result_limit: int

class SearchTaskOutput(TypedDict):
    search_query: str
    web_research_result: list[str]
    sources_gathered: list[str]

class RecollTaskInput(TypedDict):
    search_query: str
    id: int
    search_query_result_limit: int

class RecollTaskOutput(SearchTaskOutput):
   pass

@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report

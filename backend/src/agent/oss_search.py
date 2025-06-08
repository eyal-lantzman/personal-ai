from pydantic import BaseModel, RootModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from ratelimit import limits, RateLimitException, sleep_and_retry

import logging

logger = logging.getLogger(__name__)


class AggregatedSimpleSearchResults(BaseModel):
    """Results from simple_search."""
    sources_gathered: list[str]
    search_query: str
    web_research_results: list[str]

class SimpleSearchResult(BaseModel):
    snippet: str
    title: str
    link: str

class SimpleSearchResults(RootModel):
    root: List[SimpleSearchResult] = Field(default_factory=list)

def _aggregate_results(query:str, search_results:SimpleSearchResults) -> AggregatedSimpleSearchResults:
    # Creates a citations map, so that can inseart [X], X: <link>
    citations = {}
    i = 1
    for result in search_results.root:
        link = result.link
        if link not in citations:
            citations[link] = i
            i += 1

    # Create search results by modifying the original one and inserting citation markers
    modified_search_results = []
    for result in search_results.root:
        link = result.link
        snippet = result.snippet
        modified_result_text = f"{snippet} [{citations[link]}]({link})"
        modified_search_results.append(modified_result_text)

    # Use the citation links as the sources
    sources_gathered = list(citations.keys())

    return AggregatedSimpleSearchResults(
        sources_gathered=sources_gathered,
        search_query=query, 
        web_research_results=modified_search_results
    )

def find_last_message_of_type(type:type, messages:list[BaseMessage]) -> BaseMessage:
    for message in reversed(messages):
        if isinstance(message, type):
            return message
    return None

def simple_search(query:str, model_id:str = None, max_iterations:int = 10, **kwargs) -> AggregatedSimpleSearchResults:
    """ kwargs: See also: https://python.langchain.com/docs/integrations/tools/ddg/"""

    @tool(description="Useful to for when you need to answer questions about current events. Input should be a search query.", 
          return_direct=True)
    @sleep_and_retry
    @limits(calls=10, period=60)
    def search_tool(query:str):
        from duckduckgo_search.exceptions import DuckDuckGoSearchException
        try:
            return DuckDuckGoSearchResults(output_format="json", **kwargs).invoke(query)
        except DuckDuckGoSearchException as e:
            raise RateLimitException(repr(e), 0)

    parser = PydanticOutputParser(pydantic_object=SimpleSearchResults)

    tool_result = search_tool.invoke(query)
    result = parser.parse(tool_result)

    # TODO: consider more expansive summary of the page.
    return _aggregate_results(query, result)

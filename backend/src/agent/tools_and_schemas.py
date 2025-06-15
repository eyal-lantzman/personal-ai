from typing import List, Optional
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )

class AggregatedSearchResults(BaseModel):
    """Results from simple_search"""
    sources_gathered: list[str] = Field(
        description="Sources (urls) which were gathered during search"
    )
    search_query: str = Field(
        description="Search query that was used"
    )
    web_research_results: list[str] = Field(
        description="The textual search results (snippets)"
    )


class WebSearchMemory(BaseModel):
    prefix: int = Field(description="The prefix or an id of the search at the time it was captured")
    search_query: str  = Field(description="The query that was used for search for the results")
    web_research_result: list[str] = Field(description="The textual results identified during the search")
    sources_gathered: list[str] = Field(description="The source urls for the textual results")
    # TODO: metadata
    id: Optional[str] =  Field(default=None, description="The id of the object in store, when applicable")

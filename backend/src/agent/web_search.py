from pydantic import BaseModel, RootModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from ratelimit import limits, RateLimitException, sleep_and_retry
from agent.tools_and_schemas import AggregatedSearchResults
from agent.openai_compatible_models import get_chat, remove_thinking

import logging

logger = logging.getLogger(__name__)

RATE_CALLS = 2
RATE_PER_SECONDS = 5

# See also:https://duckduckgo.com/duckduckgo-help-pages/settings/params
_REGIONS = {
    "Default": "wt-wt",
    "Argentina": "ar-es",
    "Austria": "at-de",
    "Belgium (fr)": "be-fr",
    "Belgium (nl)": "be-nl",
    "Brazil": "br-pt",
    "Bulgaria": "bg-bg",
    "Canada": "ca-en",
    "Canada (fr)": "ca-fr",
    "Catalan": "ct-ca",
    "Chile": "cl-es",
    "China": "cn-zh",
    "Colombia": "co-es",
    "Croatia": "hr-hr",
    "Czech Republic": "cz-cs",
    "Denmark": "dk-da",
    "Estonia": "ee-et",
    "Finland": "fi-fi",
    "France": "fr-fr",
    "Germany": "de-de",
    "Greece": "gr-el",
    "Hong Kong": "hk-tz",
    "Hungary": "hu-hu",
    "India": "in-en",
    "indonesia": "id-id",
    "Indonesia (en)": "id-en",
    "Ireland": "ie-en",
    "Israel": "il-he",
    "Italy": "it-it",
    "Japan": "jp-jp",
    "Korea": "kr-kr",
    "Latvia": "lv-lv",
    "Lithuania": "lt-lt",
    "Latin America": "xl-es",
    "Malaysia": "my-ms",
    "Malaysia (en)": "my-en",
    "Mexico": "mx-es",
    "Netherlands": "nl-nl",
    "New Zealand": "nz-en",
    "Norway": "no-no",
    "Peru": "pe-es",
    "Philippines": "ph-en",
    "Philippines (tl)": "ph-tl",
    "Poland": "pl-pl",
    "Portugal": "pt-pt",
    "Romania": "ro-ro",
    "Russia": "ru-ru",
    "Singapore": "sg-en",
    "Slovak Republic": "sk-sk",
    "Slovenia": "sl-sl",
    "South Africa": "za-en",
    "Spain": "es-es",
    "Sweden": "se-sv",
    "Switzerland (de)": "ch-de",
    "Switzerland (fr)": "ch-fr",
    "Switzerland (it)": "ch-it",
    "Taiwan": "tw-tzh",
    "Thailand": "th-th",
    "Turkey": "tr-tr",
    "Ukraine": "ua-uk",
    "United Kingdom": "uk-en",
    "United States": "us-en",
    "United States (es)": "ue-es",
    "Venezuela": "ve-es",
    "Vietnam": "vn-vi",
}
_REGIONS_PROMPT_VARIABLE = ""
for location, code in _REGIONS.items():
    _REGIONS_PROMPT_VARIABLE += f"- {code}: {location}\n"

region_inference_llm = get_chat(
    temperature=0.8,
    max_retries=2,
)

region_detection_prompt = """You are a linguist assistant specializes in identifying the writing style and associating it with a region (code).

Instructions:
-------------
- Based on the "Blurb," determine the region (geographical or cultural) the user is referring to. Focus solely on analyzing language, etymology, and contextual clues in the text, not direct answers.
- Prioritize simplicity: examine words, their meanings, and etymological roots within the "Blurb." If a geographic location is explicitly mentioned, use that; otherwise, rely on linguistic patterns or cultural terms.
- Analyze each word etymology to identify if it relates to specific countries/regions (e.g., terms unique to a region).
- Analyze each word spelling to identify if it relates to specific countries/regions
- Analyze each word as a location specific term to identify if it relates to specific countries/regions
- Use language style, vocabulary, spelling, slang, abbreviations and contextual hints (like location references) to infer the region. If none exist, reply: wt-wt.
- Only output the identified code (e.g., "wt-wt") — no explanations.

Region (and codes):
--------
{regions}

Blurb:
------
{query}

The recommeded code for the blurb is:"""


class SimpleSearchResult(BaseModel):
    """Single search result"""
    snippet: str = Field(description="Textual summary of a search result")
    title: str = Field(description="Title of the content for a search result)")
    link: str = Field(description="The source of the content for the search result")

class SimpleSearchResults(RootModel):
    root: List[SimpleSearchResult] = Field(default_factory=list, description="List of search results")

def _aggregate_results(prefix:int, query:str, search_results:SimpleSearchResults) -> AggregatedSearchResults:
    # Creates a citations map, so that can inseart [X], X: <link>
    citations = {}
    i = 1
    for result in search_results.root:
        link = result.link
        if link not in citations:
            citations[link] = f"{prefix}.{i}"
            i += 1

    # Create search results by modifying the original one and inserting citation markers
    modified_search_results = []
    sources_gathered = []
    for result in search_results.root:
        link = result.link
        snippet = result.snippet
        source = f"[{citations[link]}]({link})"
        modified_result_text = f"{snippet} {source}"
        sources_gathered.append(source)
        modified_search_results.append(modified_result_text)

    return AggregatedSearchResults(
        sources_gathered=sources_gathered,
        search_query=query, 
        web_research_results=modified_search_results
    )

# TODO: consider using async rate limiter with more strategies: https://asynciolimiter.readthedocs.io/en/latest/
@sleep_and_retry
@limits(calls=RATE_CALLS, period=RATE_PER_SECONDS)
def _search(query:str, num_results:int = 4, region:str = None):
    import duckduckgo_search.exceptions as ddg_exceptions
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
    # TODO:
    # consider news, image, video search 
    for backend in ["html", "lite"]:
        search_region = region=region or "wt-wt"
        search_source = "text"
        search_results = num_results 
        service = DuckDuckGoSearchResults(
            output_format="json", 
            backend=search_source, 
            num_results=search_results,
            region=search_region,
            api_wrapper=DuckDuckGoSearchAPIWrapper(
                region=search_region, 
                backend=backend,
                source=search_source, 
                max_results=search_results, 
                safesearch="off"),
            )
        try:
            return service.invoke(query)
        except (ddg_exceptions.DuckDuckGoSearchException, ddg_exceptions.TimeoutException) as e:
            logger.warning("Search err: %s", e)
            continue
        except ddg_exceptions.RatelimitException as e:
            import random
            raise RateLimitException(repr(e), random.randint(1, 10))
    # Failed, return an empty search result
    return "[]"

@tool(description="Useful to for when you need to answer questions about current events. Input should be a search query.", 
      return_direct=True)
def search_tool(query:str, num_results:int = 4):
    """Exposes a search as an AI tool. 
    Internally retry and rate limitted are applied already through 'RATE_CALLS' and 'RATE_PER_SECONDS' globals.
    
    Args:
        query (str): The query into search engine.
        num_results (int): Maximum of results returned.

    Returns:
        dict : Search results, see also: https://python.langchain.com/docs/integrations/tools/ddg
    """
    return _search(query, num_results)

def simple_search(query_id:int, query:str, num_results:int = 4, region:str = None) -> AggregatedSearchResults:
    """Searches the web.
    Internally retry and rate limitted are applied already through 'RATE_CALLS' and 'RATE_PER_SECONDS' globals.
    
    Args:
        query_id (int): Query identifier. Used as a prefix for citation id's e.g. <query_id>.<running number>.
        query (str): The query into search engine.
        num_results (int): Maximum of results returned.

    Returns:
        AggregatedSearchResults : Search results.
    """
    parser = PydanticOutputParser(pydantic_object=SimpleSearchResults)

    tool_result = search_tool.invoke({"query": query, "num_results": num_results, "region": region})
    result = parser.parse(tool_result)

    # TODO: consider more expansive summary of the page.
    return _aggregate_results(query_id, query, result)


def identify_region(query:str) -> str:
    """Uses AI to identify the search 'region' parameter based on a query. 
    This method should not fail and in worst case will return the default value for the search engine.
    
    Args:
        query (str): The user's query.

    Returns:
        Region code for that can be passed for searching region specific information.
    """

    formatted_prompt = region_detection_prompt.format(
        query=query,
        regions=_REGIONS_PROMPT_VARIABLE)
    try:
        message = region_inference_llm.invoke(formatted_prompt)
        reply = remove_thinking(message).content
        if reply in _REGIONS.values():
            return reply
    except Exception as e:
        logger.warning(e)

    return _REGIONS["Default"]
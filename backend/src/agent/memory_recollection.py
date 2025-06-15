import json
import uuid
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from agent.openai_compatible_models import get_text_embedding
from agent.tools_and_schemas import WebSearchMemory, AggregatedSearchResults
import logging

logger = logging.getLogger(__name__)

def aggregate_memories(query_id:int, search_query:str, memories:list[WebSearchMemory]) -> AggregatedSearchResults:
    # TODO: need to incorporate the query id and change the citations
    # TODO: need to detect dupes from current searches
    web_research_results = []
    sources_gathered = []
    for memory in memories:
        web_research_results.extend(memory.web_research_result)
        sources_gathered.extend(memory.sources_gathered)
    return AggregatedSearchResults(
        search_query=search_query, 
        web_research_results=web_research_results,
        sources_gathered=sources_gathered)

class MemoryManager:
    def __init__(self, 
                namespace:tuple[str, ...] | str = None, 
                store:BaseStore = None):
        
        self.store = store or InMemoryStore(
            index={
                "dims": 1024,
                "embed": get_text_embedding(),
            }
        ) 
        self.manage_tool = None
        self.search_tool = None
        self.namespace = namespace or ("memories", )
        self.init_tools()

    def init_tools(self):
        self.manage_tool = create_manage_memory_tool(
            namespace=self.namespace,
            store=self.store)
        
        self.search_tool = create_search_memory_tool(
            namespace=self.namespace,
            store=self.store, 
            response_format="content_and_artifact")

    def recoll_search(self, 
                             query_id:int, 
                             search_query:str, 
                             num_results:int = 4, 
                             min_search_score:float = 0.85) -> AggregatedSearchResults:
        """Recolls past memories based on search query minimal score similarity.
        
        Args:
            query_id (int): Query identifier. Used as a prefix for citation id's e.g. <query_id>.<running number>.
            search_query (str): The query to use for similarity search.
            num_results (int): Maximum of results returned.
            min_search_score(float): Minimal similarity score to use for filtering the initial results.

        Returns:
            AggregatedSearchResults : Search results.
        """
        response = self.search_tool.invoke({"query":search_query, "limit":num_results})
        memories = list[WebSearchMemory]()

        if isinstance(response, str):
            response = json.loads(response)

        if response:
            try:
                for hit in response:
                    if hit["score"] >= min_search_score:
                        memory = WebSearchMemory.model_validate_json(hit["value"]["content"])
                        memory.id = hit["key"]
                        memories.append(memory)
            except Exception as e:
                logger.warning("recollection error: %s", str(e))

        return aggregate_memories(query_id, search_query, memories)


    def remember_search(self,
                          research:WebSearchMemory) -> str:
        """Remembers a memory expressed as a research.
        
        Args:
            research (WebSearchMemory): The research to remember.
            
        Returns:
            str : Memory id or None if didn't save.
        """
        if research.search_query and research.web_research_result and research.sources_gathered:
            response:str = self.manage_tool.invoke({"content":research.model_dump_json(), "action": "create"})
            logger.debug(response)
            return response.split(" ")[-1]
        
        return None

    def forget_search(self, id:str) -> bool:
        """Forgets a memory represented by an id.
        
        Args:
            id (str): The memory id to forget.
            
        Returns:
            bool : weather forgot this item or not.
        """
        if id is not None:
            try:
                response = self.manage_tool.invoke({"id": uuid.UUID(id), "action": "delete"})
                logger.debug(response)
                return response.split(" ")[-1] == id
            except ValueError:
                pass
        
        return False

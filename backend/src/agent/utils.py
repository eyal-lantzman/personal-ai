from typing import List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from agent.openai_compatible_models import remove_thinking

def get_research_topic(messages: List[AnyMessage]) -> str:
    """
    Get the research topic from the messages.
    """
    # check if request has a history and combine the messages into a single string
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                remove_thinking(message)
                research_topic += f"Assistant: {message.content}\n"
    return research_topic

from agent.lm_server import get_loaded_model, get_base_url
import logging

logger = logging.getLogger(__name__)

def remove_thinking(message_or_content):
    def remove_thinking_between_tags(message:str) ->str:
        start_thinking = "<think>"
        stop_thinking = "</think>\n\n"
        lindex = -1
        rindex = -1
        if start_thinking in message:
            lindex = message.index(start_thinking)
        if stop_thinking in message:
            rindex = message.index(stop_thinking)
        if lindex>=0 and rindex >=0:
            logger.debug("Removing thinking section: %s", message[:len(stop_thinking) + rindex])
            return message[len(stop_thinking) + rindex:]
        return message
    
    from langchain_core.messages import BaseMessage
    if isinstance(message_or_content, BaseMessage):
        message_or_content.content = remove_thinking_between_tags(message_or_content.content)
        return message_or_content
    elif isinstance(message_or_content, str):
        return remove_thinking_between_tags(message_or_content)
    logger.warning("Unsupported input: %s", str(message_or_content))
    return message_or_content

def get_openai_client(**kwargs):
    """Returns OpenAI client."""
    from openai import Client
    args = {
        "api_key": "NOT USED",
        "base_url": get_base_url(),
    }
    merged_kwargs = {**args, **kwargs}
    return Client(**merged_kwargs)

def get_llm(**kwargs):
    """Returns langchain OpenAI completions compatible client."""
    from langchain_openai import OpenAI
    args = {
        "api_key": "NOT USED",
        "base_url": get_base_url(),
        "model": get_loaded_model(),
    }
    merged_kwargs = {**args, **kwargs}
    return OpenAI(
        **merged_kwargs
    )

def get_chat(**kwargs):
    """Returns langchain OpenAI completions compatible client."""
    from langchain_openai import ChatOpenAI
    args = {
        "api_key": "NOT USED",
        "base_url": get_base_url(),
        "model": get_loaded_model(),
    }
    merged_kwargs = {**args, **kwargs}
    return ChatOpenAI(
        **merged_kwargs
    )

def get_text_embedding(model:str = None, **kwargs):
    """Returns langchain OpenAI embeddings compatible client for general text embeddings."""
    default_model = model or "text-embedding-granite-embedding-278m-multilingual"
    return get_embeddings(model=default_model, **kwargs)

def get_code_embedding(model:str= None, **kwargs):
    """Returns langchain OpenAI embeddings compatible client for general code embeddings."""
    default_model = model or "jina-embeddings-v2-base-code"
    return get_embeddings(model=default_model, **kwargs)

def get_embeddings(model:str, **kwargs):
    """Returns langchain OpenAI embeddings compatible client for specified embeddings model."""
    from langchain_openai import OpenAIEmbeddings
    args = {
        "api_key": "NOT USED",
        "base_url": get_base_url(),
        "model": model,
        "check_embedding_ctx_length": False
    }
    merged_kwargs = {**args, **kwargs}
    return OpenAIEmbeddings(
        **merged_kwargs
    )
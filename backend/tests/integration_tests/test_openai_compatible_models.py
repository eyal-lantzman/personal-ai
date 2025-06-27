from agent.openai_compatible_models import get_chat, get_text_embedding, get_llm, get_openai_client, get_base_url

def test_get_chat():
    chat = get_chat(model="qwen/qwen3-1.7b")
    assert chat is not None

def test_get_llm():
    chat = get_llm(model="qwen/qwen3-1.7b")
    assert chat is not None

def test_get_openai_client():
    client = get_openai_client()
    assert str(client.base_url) == get_base_url() + "/"
    assert client.models.list() is not None

def test_get_text_embedding():
    embedding = get_text_embedding(model="text-embedding-granite-embedding-278m-multilingual")
    assert embedding is not None

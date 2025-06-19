import pytest
from pathlib import Path
from services.internal.registry import Models
from services.internal.loader import (
    SUPPORTED, 
    SentenceTransformerLoader,
    AutoModelForCausalLMLoader,
    ImageTextToTextModelLoader
)
from services.internal.inference import (
    TransformerInference, 
    EmbeddingsInference, 
    Dialog, 
    Message, 
    MultiModalMessage)

import logging

logger = logging.getLogger(__name__)

MODELS_CACHE = Path(__file__).parent.parent.parent / "models"
TEST_RESOURCES = Path(__file__).parent / "test_resources"

@pytest.fixture(scope="function")
def registry():
    return Models(MODELS_CACHE)

def test_inference_simple_model_as_non_streamer(registry:Models):
    under_test = TransformerInference(registry)

    under_test.serve("Qwen/Qwen2.5-Coder-3B-Instruct")
    dialog = Dialog(user_input=Message(role="user", content="Tell me a joke"))
    dialog.history.append(Message(role="system", content="You must reply in rhymes"))
    
    response = under_test.generate_response(dialog)
    logger.info(response)

    assert response.latest_response.role == "assistant"
    assert response.latest_response.content is not None
    assert response.thinking_response is None
    assert len(response.history) == 3

@pytest.mark.asyncio(loop_scope="function")
async def test_inference_reasoning_model_as_chat_streamer(registry:Models):
    under_test = TransformerInference(registry)

    under_test.serve("Qwen/Qwen3-4B")
    dialog = Dialog(user_input=Message(role="user", content="An increasing sequence: one, /no_think"))
    dialog.history.append(Message(role="system", content="Follow instructions succinctly"))

    async for response in under_test.async_generate_response(dialog):
        logger.info(response)

    assert dialog.latest_response.role == "assistant"
    assert "one, two, three, four, five, six" in dialog.latest_response.content
    assert dialog.thinking_response.role == "assistant_thinking"
    assert dialog.thinking_response.content != "<think>\n\n</think>"
    assert len(dialog.history) == 4

@pytest.mark.parametrize("model_id", [id for id, type in SUPPORTED.items() if type == SentenceTransformerLoader])
def test_inference_sentence_transformers_as_embeddings(registry:Models, model_id):
    under_test = EmbeddingsInference(registry)
    inputs = ["I like apples", "I like oranges", "I like HuggingFace"]

    under_test.reset()
    logger.info("Serving: %s", model_id)
    under_test.serve(model_id)

    embeddigns = under_test.encode(inputs)
    logger.info(embeddigns)
    assert embeddigns.shape == (len(inputs), under_test.loader.generation_kwargs["max_length"])

    similarity = under_test.similarity(embeddigns, embeddigns)
    logger.info(similarity)
    assert similarity.shape == (len(inputs), len(inputs))
    assert similarity[0][0] == -0.0
    assert similarity[1][1] == -0.0
    assert similarity[2][2] == -0.0

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("model_id", [id for id, type in SUPPORTED.items() if type == ImageTextToTextModelLoader])
async def test_inference_image_text_as_chat(registry:Models, model_id):
    under_test = TransformerInference(registry)
    under_test.serve(model_id)
    prompt = """Extract the text from the above document as if you were reading it naturally. 
    Return the tables in html format. 
    Return the equations in LaTeX representation. 
    If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. 
    Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. 
    Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. 
    Prefer using ☐ and ☑ for check boxes."""
    from PIL import Image
    image_path = TEST_RESOURCES / "test_image_1.jpg"
    image = Image.open(image_path)
    system = Message(role="system", content= "You are a helpful assistant.")
    user = MultiModalMessage(role="user" , content= [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},
    ])
    dialog = Dialog(user_input=user)
    dialog.history.append(system)
    dialog.images.append(image)
    
    async for response in under_test.async_generate_response(dialog, max_new_tokens=50):
        logger.info(response)

@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize("model_id", [id for id, type in SUPPORTED.items() if type == AutoModelForCausalLMLoader])
async def test_inference_causalLM_as_chat(registry:Models, model_id):
    under_test = TransformerInference(registry)
    logger.info("Serving: %s", model_id)
    under_test.serve(model_id)
    
    dialog = Dialog(user_input=Message(role="user", content="An increasing sequence: one, /no_think"))
    dialog.history.append(Message(role="system", content="Follow instructions succinctly"))

    async for response in under_test.async_generate_response(dialog, max_new_tokens=50):
        logger.info(response)

    if registry.models[model_id].supports_reasoning:
        if registry.models[model_id].supports_reasoning_onoff:
            assert dialog.thinking_response.role == "assistant_thinking"
            assert dialog.thinking_response.content != "<think>\n\n</think>"
            assert len(dialog.history) == 4
        else:
            assert len(dialog.thinking_response.content) > 0
            assert "<|" not in dialog.thinking_response.content
            assert "|>" not in dialog.thinking_response.content
            assert len(dialog.history) == 4
    else:
        assert dialog.latest_response.role == "assistant"
        assert len(dialog.latest_response.content) > 0
        assert "<|" not in dialog.latest_response.content
        assert "|>" not in dialog.latest_response.content
        assert dialog.thinking_response is None
        assert len(dialog.history) == 3

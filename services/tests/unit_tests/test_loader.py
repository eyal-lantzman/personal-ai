import pytest
from pathlib import Path
from services.internal.registry import Models
from services.internal.loader import (
    AutoModelForCausalLMLoader,
    SentenceTransformerLoader, 
    ImageTextToTextModelLoader,
    SUPPORTED, 
    )
import logging

logger = logging.getLogger(__name__)

MODELS_CACHE = Path(__file__).parent.parent.parent.parent / "models"

@pytest.fixture
def registry():
    return Models(str(MODELS_CACHE))

@pytest.mark.parametrize("model_id", [id for id, type in SUPPORTED.items() if type == AutoModelForCausalLMLoader])
def test_load_unload_AutoModelForCausalLM_ALL(registry, model_id):
    under_test = AutoModelForCausalLMLoader(registry)
        
    logger.info("Loading: %s", model_id)
    info = registry.models[model_id]
    loaded = under_test.load(info)
    
    assert loaded.entry_point_model is not None
    assert loaded.pre_processor_model is not None
    assert loaded.pre_processor_model == loaded.post_processor_model

    under_test.unload(loaded)

    assert loaded.entry_point_model is None
    assert loaded.pre_processor_model is None
    assert loaded.pre_processor_model  is None


@pytest.mark.parametrize("model_id", [id for id, type in SUPPORTED.items() if type == SentenceTransformerLoader])
def test_load_unload_SentenceTransformerLoader_ALL(registry, model_id):
    under_test = SentenceTransformerLoader(registry)

    logger.info("Loading: %s", model_id)
    info = registry.models[model_id]
    loaded = under_test.load(info)
    
    assert loaded.entry_point_model is not None
    assert loaded.pre_processor_model is not None
    assert loaded.pre_processor_model == loaded.post_processor_model

    under_test.unload(loaded)
    
    assert loaded.entry_point_model is None
    assert loaded.pre_processor_model is None
    assert loaded.pre_processor_model  is None


@pytest.mark.parametrize("model_id", [id for id, type in SUPPORTED.items() if type == ImageTextToTextModelLoader])
def test_load_unload_ImageTextToTextModelLoaderr_ALL(registry, model_id):
    under_test = ImageTextToTextModelLoader(registry)
        
    logger.info("Loading: %s", model_id)
    info = registry.models[model_id]
    loaded = under_test.load(info)
    
    assert loaded.entry_point_model is not None
    assert loaded.pre_processor_model is not None
    assert loaded.pre_processor_model is not None

    under_test.unload(loaded)
    
    assert loaded.entry_point_model is None
    assert loaded.pre_processor_model is None
    assert loaded.pre_processor_model  is None

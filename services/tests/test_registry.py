import os
from pathlib import Path
from services.internal.registry import Models, UNSUPPORTED_MODELS
from services.internal.loader import SUPPORTED
import logging

logger = logging.getLogger(__name__)

MODELS_CACHE = Path(__file__).parent.parent.parent / "models"

def test_init_loads_models_from_cache():
    expected = list()
    for dir in (MODELS_CACHE / "hub").iterdir():
        if not dir.name.startswith("models--") and not dir.name.startswith("datasets--") and dir.name != ".locks":
            for sub_dir in dir.iterdir():
                if f"{dir.name}/{sub_dir.name}" in UNSUPPORTED_MODELS:
                    logger.warning("Not supported yet: %s/%s", dir.name, sub_dir.name)
                else:
                    expected.append(dir.name + "/" + sub_dir.name)


    under_test = Models(str(MODELS_CACHE))
    assert set(under_test.models.keys()) == set(expected)
    for id, model in under_test.models.items():
        assert model.model_id == id
        assert str(under_test.cache_root) in str(model.location)


def test_import_models_stress():
    under_test = Models(str(MODELS_CACHE))

    for model_id, _ in SUPPORTED.items():
        model_info = under_test.import_model(model_id, token=os.getenv("HF_TOKEN"))
        assert model_info.model_id == model_id

from agent.lm_studio_server import get_loaded_model, get_local_models

def test_get_loaded_modell():
    model = get_loaded_model()
    assert model is not None

def test_get_local_models():
    models = get_local_models()
    assert len(models) > 0
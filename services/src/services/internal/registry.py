import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ModelCard(BaseModel):
    model_id: str
    location: Path = None
    created: int

    modalities: Optional[set[str]] = {"text"}
    supports_reasoning: bool = False
    supports_reasoning_onoff: bool = False

    language: Optional[set[str]] = None
    license_name : Optional[str] = None
    license_link : Optional[str] = None
    library_name : Optional[str] = None

    tags: Optional[set[str]] = None
    datasets: Optional[set[str]] = None
    metrics: Optional[set[str]] = None
    base_model: Optional[set[str]] = None

    # TODO: consider https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
    # TODO: prepopulate during import, and persist as part of the storage system

REASONING_MODELS = { # True - can start/stop reasoning using /think or /no_think suffixes, False - otherwise, If doesn't exist => not reasoning !
    "Qwen/Qwen3-4B":True,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":False,
}
UNSUPPORTED_MODELS = [
    "ByteDance-Seed/BAGEL-7B-MoT",
    "mistralai/Mixtral-8x7B-v0.1", # Too slow
    "ByteDance-Seed/Seed-Coder-8B-Reasoning", # Too slow
    "ByteDance-Seed/Seed-Coder-8B-Instruct", # Too slow
]

class Models:
    def __init__(self, cache_location=os.getenv("HF_HOME")):
        self.models = dict[str,ModelCard]()
        root = cache_location
        if root:
            root = Path(root)
        else:
            root = Path(__file__).parent.parent
            logger.warning("Using convention for cache_location: %s", root)
        self.cache_root = root / "hub"
        assert self.cache_root.exists() and self.cache_root.is_dir(), self.cache_root
        self.update_cached()

    def update_cached(self):
        for dir in (self.cache_root).iterdir():
            if dir.name.startswith("models--") or dir.name.startswith("datasets--") or dir.name == ".locks":
                continue # HF vanilla, which is not really supported by this module
            else:
                for sub_dir in dir.iterdir():
                    if sub_dir.is_dir():
                        model_id = f"{dir.name}/{sub_dir.name}"
                        if model_id not in UNSUPPORTED_MODELS:
                            if not self.ensure_model_info(model_id, sub_dir):
                                logger.warning("Non auto-configurable model: %s/%s", dir.name, sub_dir.name)

    def ensure_mode_info_from_cache(self, dir:Path) -> bool:
        parts = dir.name.split("--")
        if len(parts) == 3:
            model_id = f"{parts[1]}/{parts[2]}"
            return self.ensure_model_info(model_id, dir)
        return False
    
    def ensure_model_info(self, model_id:str, dir:Path) -> bool:
        if model_id not in self.models:
            for root, dirs, files in dir.walk(True, follow_symlinks=True):
                supports_reasoning = model_id in REASONING_MODELS
                supports_reasoning_onoff = supports_reasoning and REASONING_MODELS[model_id]
                if "config.json" or "model_index.json" in files:
                    self.models[model_id] = ModelCard(
                        model_id=model_id, 
                        created=int(os.stat(dir).st_birthtime),
                        location=root, 
                        supports_reasoning=supports_reasoning,
                        supports_reasoning_onoff=supports_reasoning_onoff)
                    return True
            
            return False
        return True
    
    def import_model(self, model_id:str, **kwargs) -> ModelCard:
        from huggingface_hub import snapshot_download

        downloaded = snapshot_download(repo_id=model_id, local_dir=str(self.cache_root / model_id), **kwargs)

        assert self.ensure_model_info(model_id, Path(downloaded))
        
        return self.models[model_id]

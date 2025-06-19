from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any
from services.internal.registry import Models, ModelCard
import logging

# TODO: add quantization support e.g. https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo

logger = logging.getLogger(__name__)

class LoadedModel(BaseModel, ABC):
    model_id: str
    entry_point_model:Any = None
    pre_processor_model:Any = None
    post_processor_model:Any = None
 
class Loader(ABC):
    def __init__(self, registry:Models, **generation_kwargs):
        self.cache_root = registry.cache_root
        self.registry = registry
        self.generation_kwargs = generation_kwargs


    def load_transformer_causalLM(self, model_id:str):
        from transformers import AutoModelForCausalLM 
        return AutoModelForCausalLM.from_pretrained(
            self.registry.models[model_id].location,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
            cache_dir=str(self.cache_root)
        )
    
    def load_transformer_tokenizer(self, model_id:str):
        from transformers import AutoTokenizer 
        return AutoTokenizer.from_pretrained(
            self.registry.models[model_id].location,
            local_files_only=True,
            cache_dir=str(self.cache_root)
        )
    
    def load_transformer_processor(self, model_id:str):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(
            self.registry.models[model_id].location,
            local_files_only=True,
            cache_dir=str(self.cache_root)
        )
    
    def load_sentence_transformer(self, model_id:str):
        from sentence_transformers import SentenceTransformer 
        return SentenceTransformer(
            str(self.registry.models[model_id].location), 
            device="cpu",
            local_files_only=True, 
            cache_folder=str(self.cache_root),
            trust_remote_code=True,
        )
    
    def load_model(self, model_id: str):
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            self.registry.models[model_id].location,
            local_files_only=True,
            cache_dir=str(self.cache_root))
    
    def load_image_to_text_model(self, model_id: str):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            self.registry.models[model_id].location,
            torch_dtype="auto", 
            device_map="auto", 
            local_files_only=True,
            cache_dir=str(self.cache_root))
        model.eval()
        return model
   
    def unload(self, model:LoadedModel):
        import gc
        import torch

        del model.entry_point_model
        model.entry_point_model = None
        del model.pre_processor_model
        model.pre_processor_model = None
        del model.post_processor_model
        model.post_processor_model = None
        gc.collect()
        torch.cuda.empty_cache()
        self.log_usage_metrics()

    def log_usage_metrics(self):
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            logger.debug(torch.cuda.memory_summary())
        else:
            logger.warning("CUDA not loaded")
    
    @abstractmethod
    def load(self, model_info:ModelCard, **kwargs) -> LoadedModel: 
        pass

    @staticmethod
    def generate_kwargs(args:dict, **kwargs):
        arguments = dict(kwargs)
        for k,v in args.items():
            arguments[k] = v
        return arguments
    
class AutoModelForCausalLMLoader(Loader):
    def __init__(self, 
                 registry:Models, 
                 return_tensors:str="pt",
                 **generation_kwargs):
        super().__init__(registry, **generation_kwargs)
        self.return_tensors = return_tensors

    def load(self, model_info:ModelCard) -> LoadedModel:
        loaded = LoadedModel(model_id=model_info.model_id)
        loaded.entry_point_model = self.load_transformer_causalLM(model_info.model_id)
        loaded.pre_processor_model = self.load_transformer_tokenizer(model_info.model_id)
        loaded.post_processor_model = loaded.pre_processor_model
        self.log_usage_metrics()
        return loaded

class SentenceTransformerLoader(Loader):
    def __init__(self, 
                 registry:Models, 
                 **generation_kwargs):
        super().__init__(registry, **generation_kwargs)

    def load(self, model_info:ModelCard)-> LoadedModel:
        loaded = LoadedModel(model_id=model_info.model_id)
        loaded.entry_point_model = self.load_sentence_transformer(model_info.model_id)
        loaded.pre_processor_model = loaded.entry_point_model.tokenizer
        loaded.post_processor_model = loaded.pre_processor_model
        return loaded

class ImageTextToTextModelLoader(Loader):
    def __init__(self, 
                 registry:Models, 
                 return_tensors:str="pt",
                 **generation_kwargs
        ):
        super().__init__(registry, **generation_kwargs)
        self.return_tensors = return_tensors

    def load(self, model_info:ModelCard)-> LoadedModel:
        loaded = LoadedModel(model_id=model_info.model_id)
        loaded.entry_point_model = self.load_image_to_text_model(model_info.model_id)
        loaded.pre_processor_model = self.load_transformer_processor(model_info.model_id)
        loaded.post_processor_model = loaded.pre_processor_model

        self.log_usage_metrics()
        return loaded
    
class NotImplemented(Loader):
    def __init__(self, 
                 registry:Models, 
                 **generation_kwargs
        ):
        super().__init__(registry, **generation_kwargs)

    def load(self, model_info:ModelCard)-> LoadedModel:
        self.log_usage_metrics()
        raise NotImplementedError()

SUPPORTED = {
    "meta-llama/Llama-3.2-1B-Instruct":AutoModelForCausalLMLoader,
    "Qwen/Qwen3-4B":AutoModelForCausalLMLoader,
    "Qwen/Qwen2.5-Coder-3B-Instruct":AutoModelForCausalLMLoader,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":AutoModelForCausalLMLoader,
    "deepseek-ai/deepseek-coder-6.7b-instruct":AutoModelForCausalLMLoader,

    "sentence-transformers/all-MiniLM-L6-v2":SentenceTransformerLoader,
    "ibm-granite/granite-embedding-278m-multilingual":SentenceTransformerLoader,

    "nanonets/Nanonets-OCR-s": ImageTextToTextModelLoader,
}
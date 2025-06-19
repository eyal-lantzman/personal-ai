from abc import ABC
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from threading import Thread
from transformers import TextIteratorStreamer
from typing import AsyncIterator, Self, Any, Union, Literal
from services.internal.registry import Models
from services.internal.loader import LoadedModel, SUPPORTED
import logging

logger = logging.getLogger(__name__)

DEFAULT_LOADER_CONFIG = {
    "ibm-granite/granite-embedding-278m-multilingual":  { "max_length":768 },
    "sentence-transformers/all-MiniLM-L6-v2":  { "max_length":384 },

    "Qwen/Qwen3-4B": {"max_new_tokens":32768},
    "meta-llama/Llama-3.2-1B-Instruct": {"max_new_tokens":32768},
    "Qwen/Qwen3-4B": {"max_new_tokens":32768},
    "Qwen/Qwen2.5-Coder-3B-Instruct":{"max_new_tokens":32768},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": { "temperature": 0.6, "max_new_tokens": 32768},
    "deepseek-ai/deepseek-coder-6.7b-instruct":{ "temperature": 0.6, "max_new_tokens": 32768},
}

CUSTOMIZE_PROMPTS = {
    
}

class Message(BaseModel):
    role:str
    content:Union[str,list[str]]

class ImageContentType(BaseModel):
    image:Any = None
    type:str = Literal["image"]

class TextContentType(BaseModel):
    text:str
    type:str = Literal["text"]

class AudioContentType(BaseModel):
    audio:Any = None
    type:str = Literal["audio"]

class MultiModalMessage(Message):
    content: list[Union[ImageContentType, TextContentType, AudioContentType]]

class Dialog(BaseModel):
    id:UUID = Field(default_factory=uuid4, frozen=True)
    user_input:Message
    thinking:bool = None
    latest_response:Message = None
    thinking_response:Message = None
    history:list[Message] = Field(default_factory=list)
    thinking_content:Union[str,list[str]] = ""
    content:Union[str,list[str]] = ""
    images:list = Field(default_factory=list)

    def process_part_of_iteration(self, content:str):
        if not content:
            return
        
        if "<think>" in content:
            self.thinking = True
            self.thinking_content += content
        elif "</think>" in content:
            self.thinking = False
            self.thinking_content += content
        else:
            if self.thinking:
                self.thinking_content += content
            else:
                self.content += content

    def initiate_iteration(self) -> Self:
        self.thinking = False
        self.content = ""
        self.latest_response = None
        if len(self.history) == 0 or self.history[-1] != self.user_input:
            self.history.append(self.user_input)

        return self

    def complete_iteration(self) -> Self:
        if isinstance(self.thinking_content, str) and self.thinking_content or any(self.thinking_content):
            self.thinking_response = Message(role="assistant_thinking", content=self.thinking_content)
            self.history.append(self.thinking_response)

        self.latest_response = Message(role="assistant", content=self.content)
        self.history.append(self.latest_response)

        return self
    
class BaseInference(ABC):
    def __init__(self, registry:Models):
        self.registry = registry
        self.loader = None
        self.model_id:str = None
        self.loaded:LoadedModel = None
        self.ready = False
    
    def reset(self):
        if self.ready:
            self.loader.unload(self.loaded)
            self.loader = None
            self.model_id = None
            self.loaded = None
            self.ready = False

    def serve(self, model_id:str, **kwargs):
        if model_id not in SUPPORTED:
            raise NotImplementedError("model_id: % is not supported yet", model_id)

        if self.ready:
            raise ValueError("Already serving %s", self.model_id)

        model_info = self.registry.models[model_id]

        loader_kwargs = DEFAULT_LOADER_CONFIG.get(model_id, {})
        self.loader = SUPPORTED[model_id](self.registry, **loader_kwargs)
        self.loaded = self.loader.load(model_info, **kwargs)
        self.model_id = model_id
        self.ready = True

class EmbeddingsInference(BaseInference):
    def encode(self, sentances:list[str], **kwargs) -> list:
        arguments = {**self.loader.generation_kwargs, **kwargs}

        return self.loaded.entry_point_model.encode(sentances, **arguments)
    
    def similarity(self, embedings1:list, embedings2:list, similarity_function:str = None) -> list[list]:
        if similarity_function is None:
            similarity_function = "euclidean"
        self.loaded.entry_point_model.similarity_fn_name = similarity_function
        return self.loaded.entry_point_model.similarity(embedings1, embedings2)

class TransformerInference(BaseInference):
    def __init__(self, registry:Models):
        super().__init__(registry)
        self.history = []

    def reset(self):
        if self.ready:
            super().reset()
            self.history.clear()

    def serve(self, model_id:str, **kwargs):
        super().serve(model_id, **kwargs)

    def process_messages_apply_prompt_template(self, dialog:Dialog):
        messages:list[Message] = [] + dialog.history
        if dialog.user_input not in dialog.history:
            messages.append(dialog.user_input)
        messages = [m.model_dump() for m in messages]

        if messages and self.model_id in CUSTOMIZE_PROMPTS:
            if "replace_role_system" in CUSTOMIZE_PROMPTS[self.model_id]:
                for message in messages:
                    if message["role"] == "system":
                        message["role"] = CUSTOMIZE_PROMPTS[self.model_id]["replace_role_system"]

            if "alternate_roles" in CUSTOMIZE_PROMPTS[self.model_id] and CUSTOMIZE_PROMPTS[self.model_id]["alternate_roles"]:
                previous = messages[0]
                i = 1
                while i < len(messages):
                    if previous["role"] == messages[i]["role"]:
                        previous["content"] += "\n" + messages[i]["content"]
                        messages.pop(i)
                    else:
                        i += 1


            return self.loaded.pre_processor_model.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
        else:

            return self.loaded.pre_processor_model.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )

    async def async_generate_response(self, dialog:Dialog, cancellable_callback:callable = None, **generation_kwargs) -> AsyncIterator[str]:
        dialog.initiate_iteration()
        try:
            text = self.process_messages_apply_prompt_template(dialog)

            streamer = TextIteratorStreamer(self.loaded.pre_processor_model, skip_special_tokens=True)
            config_kwargs = { **self.loader.generation_kwargs, **generation_kwargs }
            
            processor_kwargs = {}
            if dialog.images:
                processor_kwargs["images"] = dialog.images

            model_inputs = self.loaded.pre_processor_model(
                text=text, 
                return_tensors=self.loader.return_tensors, 
                truncation=True,
                **processor_kwargs
                ).to(self.loaded.entry_point_model.device)

            streaming_generation_kwargs = { **dict(model_inputs, streamer=streamer), **config_kwargs}

            thread = Thread(target=self.loaded.entry_point_model.generate, kwargs=streaming_generation_kwargs)

            thread.start()
            for message in streamer:
                if message:
                    dialog.process_part_of_iteration(message)
                    yield message
                if cancellable_callback and not cancellable_callback(message):
                    logger.info("Cancelling")
                    break
        except Exception as ex:
            logger.exception(ex)
            raise
        finally:
            # Update history
            dialog.complete_iteration()

    def generate_response(self, dialog:Dialog, processor_kwargs:dict=None, **generation_kwargs) -> Dialog:
        dialog.initiate_iteration()
        try:
            text = self.process_messages_apply_prompt_template(dialog)

            model_inputs = self.loaded.pre_processor_model(
                text, 
                return_tensors=self.loader.return_tensors,
                **(processor_kwargs or {})
                ).to(self.loaded.entry_point_model.device)
            
            config_kwargs = { **self.loader.generation_kwargs, **generation_kwargs }

            generated_ids = self.loaded.entry_point_model.generate(
                **model_inputs, 
                **config_kwargs
            )
            
            output_ids = generated_ids.tolist()

            if "num_return_sequences" in generation_kwargs and generation_kwargs["num_return_sequences"] > 1:
                dialog.content = [None] * generation_kwargs["num_return_sequences"]
                dialog.thinking_content = [None] * generation_kwargs["num_return_sequences"]
                for i in range(generation_kwargs["num_return_sequences"]):
                    if i >= len(output_ids):
                        while len(dialog.content) != len(output_ids):
                            dialog.content.pop(i)
                            dialog.thinking_content.pop(i)
                        break
                    content = self.loaded.post_processor_model.decode(output_ids[i], skip_special_tokens=True).strip("\n")
                    parts = content.split("</think>")
                    if len(parts) > 1:
                        dialog.thinking_content[i] = parts[0]
                        dialog.content[i] = parts[1]
                    else:
                        dialog.content[i] = parts[0]

            else:
                content = self.loaded.post_processor_model.decode(output_ids[0], skip_special_tokens=True).strip("\n")
                parts = content.split("</think>")
                if len(parts) > 1:
                    dialog.thinking_content = parts[0]
                    dialog.content = parts[1]
                else:
                    dialog.content = parts[0]

            return dialog
        except Exception as ex:
            logger.exception(ex)
            raise
        finally:
            # Update history
            dialog.complete_iteration()
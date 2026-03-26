import asyncio
import os
from threading import Thread
from typing import Any, Callable
from collections.abc import AsyncGenerator

import torch
from transformers import TextIteratorStreamer

from mindspeed_llm.fsdp2.utils.logging import get_logger
from .base_engine import BaseEngine, Response


logger = get_logger(__name__)


class HuggingfaceEngine(BaseEngine):
    def __init__(self, model, tokenizer, args) -> None:
        self.name = "huggingface"
        self.can_generate = True
        
        self.model = getattr(model, "model", model)
        self.tokenizer = tokenizer
        self.args = args

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()

        # Initialize asyncio loop safely
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        self.semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", "1")))

    @staticmethod
    def _get_target_device() -> torch.device:
        """Fetch the correct local NPU device based on environment variables."""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(torch.accelerator.current_accelerator().type, local_rank)

    @staticmethod
    def _process_args(
        model,
        tokenizer,
        args,
        messages: list[dict[str, str]],
        **input_kwargs,
    ) -> tuple[dict[str, Any], int]:
        
        target_device = HuggingfaceEngine._get_target_device()

        # Build prompt using chat template or fallback to raw text
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(target_device) 
        except Exception:
            raw_text = "\n".join([m["content"] for m in messages])
            input_ids = tokenizer(raw_text, return_tensors="pt").input_ids.to(target_device)

        prompt_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids).to(target_device)

        # Configure generation parameters
        gen_config = model.generation_config
        gen_config.max_new_tokens = getattr(args, "max_new_tokens", 512)
        gen_config.do_sample = getattr(args, "do_sample", False)  # Disabled to prevent FSDP sampling deadlocks

        gen_config.pad_token_id = tokenizer.pad_token_id

        gen_kwargs = dict(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
            synced_gpus=False, 
        )
        return gen_kwargs, prompt_length

    @staticmethod
    @torch.inference_mode()
    def _chat(
        model, tokenizer, args, messages, **input_kwargs
    ) -> list["Response"]:
        
        target_device = HuggingfaceEngine._get_target_device()
        torch.accelerator.set_device(target_device)

        gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
            model, tokenizer, args, messages, **input_kwargs
        )
        
        # Ensure generation runs strictly on the assigned NPU
        with torch.device(target_device):
            outputs = model.generate(**gen_kwargs)
        
        response_ids = outputs[:, prompt_length:]
        responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        results = [
            Response(
                response_text=res_text,
                response_length=len(response_ids[i]),
                prompt_length=prompt_length,
                finish_reason="stop" 
            ) for i, res_text in enumerate(responses)
        ]
        return results

    @staticmethod
    @torch.inference_mode()
    def _stream_chat(
        model, tokenizer, args, messages, **input_kwargs
    ) -> Callable[[], str]:
        
        target_device = HuggingfaceEngine._get_target_device()
        gen_kwargs, _ = HuggingfaceEngine._process_args(
            model, tokenizer, args, messages, **input_kwargs
        )
        
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        def generate():
            torch.accelerator.set_device(target_device)
            with torch.device(target_device):
                model.generate(**gen_kwargs)

        # Launch generation in background thread
        thread = Thread(target=generate, daemon=True)
        thread.start()

        def stream():
            return next(streamer)

        return stream

    async def chat(self, messages, system=None, **input_kwargs) -> list["Response"]:
        input_args = (self.model, self.tokenizer, self.args, messages)
        async with self.semaphore:
            return await asyncio.to_thread(self._chat, *input_args, **input_kwargs)

    async def stream_chat(self, messages, system=None, **input_kwargs) -> AsyncGenerator[str, None]:
        input_args = (self.model, self.tokenizer, self.args, messages)
        async with self.semaphore:
            stream_func = self._stream_chat(*input_args, **input_kwargs)
            while True:
                try:
                    yield await asyncio.to_thread(stream_func)
                except StopAsyncIteration:
                    break
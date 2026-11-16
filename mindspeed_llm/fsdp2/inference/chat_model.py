# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from enum import Enum
from collections.abc import Generator
from threading import Thread

from .engine.base_engine import BaseEngine, Response
from .engine.hf_engine import HuggingfaceEngine


def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class InferBackend(str, Enum):
    HUGGINGFACE = "huggingface"


class ChatModel:
    r"""General class for chat models. Bridges Async engine."""

    def __init__(self, model, tokenizer, args) -> None:
        # 1. Get the inference backend from args, default to huggingface
        infer_backend = getattr(args, "infer_backend", InferBackend.HUGGINGFACE).lower()

        # 2. Factory pattern to select the appropriate inference engine
        if infer_backend == InferBackend.HUGGINGFACE:
            self.engine: BaseEngine = HuggingfaceEngine(model, tokenizer, args)
        else:
            raise ValueError(f"Unknown infer_backend: {infer_backend}. Only choose from: {InferBackend.HUGGINGFACE}.")

        # 3. Start the background asyncio event loop
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(self, messages: list[dict[str, str]], **input_kwargs) -> list["Response"]:
        task = asyncio.run_coroutine_threadsafe(
            self.engine.chat(messages, **input_kwargs), self._loop
        )
        return task.result()

    def stream_chat(self, messages: list[dict[str, str]], **input_kwargs) -> Generator[str, None, None]:
        generator = self.engine.stream_chat(messages, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break
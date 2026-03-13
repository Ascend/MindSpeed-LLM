from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Response:
    response_text: str
    response_length: int
    prompt_length: int
    finish_reason: Literal["stop", "length"]


class BaseEngine(ABC):
    r"""Base class for inference engine of chat models."""
    name: str
    can_generate: bool

    @abstractmethod
    def __init__(self, model, tokenizer, args) -> None:
        r"""Initialize an inference engine."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        **input_kwargs,
    ) -> list["Response"]:
        r"""Get a list of responses of the chat model."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        r"""Get the response token-by-token of the chat model."""
        pass
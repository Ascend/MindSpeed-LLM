from functools import wraps

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.transformer_config import TransformerConfig


def fp8_context_wrapper(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Wraps the fp8_context_wrapper function."""

    def wrapper_fn(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with get_fp8_context(config, layer_no, is_init):
                return fn(*args, **kwargs)

        return wrapper

    return wrapper_fn
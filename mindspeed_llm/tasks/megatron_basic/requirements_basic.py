from functools import wraps

from typing import Any
from typing_extensions import Self


def version_wrapper(func):
    @wraps(func)
    def wrapper(name, *args, **kwargs):
        if name == 'transformer-engine':
            return '2.2.0'
        elif name == 'flash_attn':
            return '1.0'
        return func(name, *args, **kwargs)
    return wrapper


def _load_from_state_dict_wrapper(fn):
    def wrapper(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        from megatron.training import get_args
        args = get_args()
        if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
            if not any(('lora_a' in key.lower() or 'lora_b' in key.lower()) and key.endswith('weight') for key in state_dict):
                import warnings
                warnings.warn("The lora weights is missing from the checkpoint and will be randomly initialized.", RuntimeWarning)
                missing_keys[:] = [key for key in missing_keys if 'lora_a' not in key.lower() and 'lora_b' not in key.lower()]
            filter_keys = ('absmax', 'quant_map', 'bitsandbytes__nf4', '_extra_state')
            unexpected_keys[:] = [key for key in unexpected_keys if not key.endswith(filter_keys)]

    return wrapper
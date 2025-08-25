import signal
from functools import wraps
import torch


def communication_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args
        if get_args().enable_high_availability:
            from mindio_ttp.adaptor import tft_is_arf_reboot_node
            if tft_is_arf_reboot_node():
                return None
        return fn(*args, **kwargs)
    return wrapper


def new_group_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        backend = kwargs.get('backend', None)
        from mindio_ttp.adaptor import tft_is_arf_reboot_node
        if tft_is_arf_reboot_node() and isinstance(backend, str) and 'gloo' in backend:
            return None

        if (backend is None) or torch.distributed.distributed_c10d._is_barrier_after_init():
            kwargs['use_local_synchronization'] = True
        res = fn(*args, **kwargs)
        return res
    return wrapper

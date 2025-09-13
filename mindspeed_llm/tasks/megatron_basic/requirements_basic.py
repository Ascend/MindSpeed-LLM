from functools import wraps


def version_wrapper(func):
    @wraps(func)
    def wrapper(name, *args, **kwargs):
        if name == 'transformer-engine':
            return '2.2.0'
        elif name == 'flash_attn':
            return '1.0'
        return func(name, *args, **kwargs)
    return wrapper
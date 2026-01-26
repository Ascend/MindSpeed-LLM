"""FSDP backend global variables."""

_GLOBAL_ARGS = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    if var is None:
        raise ValueError(f'{name} is not initialized.')


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
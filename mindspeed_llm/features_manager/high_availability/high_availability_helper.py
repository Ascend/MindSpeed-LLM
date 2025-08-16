import os
import warnings
from functools import wraps


def get_env_args(args):
    env = os.getenv('HIGH_AVAILABILITY', '')
    if env.lower() in ('dump', 'recover', 'retry'):
        if not getattr(args, 'enable_high_availability', False):
            warnings.warn(
                "HIGH_AVAILABILITY environment variables enabled and args.enable_high_availability inactive"
            )
        args.enable_high_availability = True
    if env.lower() == 'recover':
        args.enable_worker_reboot = True
    if env.lower() == 'retry':
        args.enable_hbmfault_repair = True
    return args


def skip_reuse_register_patches(fn, argument):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not argument.enable_high_availability:
            fn(self, *args, **kwargs)
    return wrapper

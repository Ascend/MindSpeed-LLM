import sys
import types
import torch


def accelerator_getattr(module, fallback_module):
    def __getattr__(name):
        if hasattr(fallback_module, name):
            attr = getattr(fallback_module, name)
            setattr(module, name, attr)
            return attr
        else:
            raise AttributeError(f'module {module} and {fallback_module} has no attribute {name}.')

    return __getattr__


def set_accelerator_compatible(fallback_module):
    accelerator_module = types.ModuleType('torch.accelerator')
    for attr in dir(torch.accelerator):
        if attr.startswith('__'):
            continue
        setattr(accelerator_module, attr, getattr(fallback_module, attr, getattr(torch.accelerator, attr)))

    accelerator_module.__getattr__ = accelerator_getattr(accelerator_module, fallback_module)
    torch.accelerator = accelerator_module
    sys.modules['torch.accelerator'] = accelerator_module
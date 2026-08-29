import importlib
import sys

try:
    from megatron.core.datasets import helpers_cpp
except ImportError:
    from megatron.core.datasets.utils import compile_helpers
    compile_helpers()
    helpers_cpp = importlib.import_module("megatron.core.datasets.helpers_cpp")
    globals().update(helpers_cpp.__dict__)
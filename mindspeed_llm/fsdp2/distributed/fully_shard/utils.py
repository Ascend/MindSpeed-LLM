import re
from typing import Optional, Iterable, List

import torch.nn as nn
from megatron.training.utils import unwrap_model, print_rank_0

def get_nested_module(model: nn.Module, path: str) -> Optional[nn.Module]:
    """
    Get nested module in PyTorch model by dot-separated path.

    Args:
        model: PyTorch model instance
        path: Dot-separated path string, e.g., "model.visual.blocks.0"

    Returns:
        Found module object, or None if path doesn't exist
    """
    if path == "":
        return model

    modules = path.split('.')
    current_module = model
    for module_name in modules:
        if hasattr(current_module, module_name):
            current_module = getattr(current_module, module_name)
        else:
            return None
    return current_module


def expand_wildcard_pattern(model: nn.Module, module_path: str) -> List[str]:
    """
    Expand wildcard pattern {*} in module path based on actual model structure.
    """
    wildcard_pattern = r'\{\*\}'
    re_match = re.search(wildcard_pattern, module_path)
    if not re_match:
        return [module_path]

    wildcard_start = re_match.start()
    wildcard_end = re_match.end()

    # Get path before {*}
    base_path = module_path[:wildcard_start].rstrip('.')
    remaining_path = module_path[wildcard_end:].lstrip('.')

    base_module = get_nested_module(model, base_path)
    expanded_paths = []
    if base_module is not None and hasattr(base_module, '__len__'):
        total = len(base_module)
        for idx in range(total):
            new_path = f"{base_path}.{idx}"
            if remaining_path:
                new_path += f".{remaining_path}"
            expanded_paths.append(new_path)
    else:
        raise ValueError(f"Module at path '{base_path}' is None or does not have length attribute")

    return expanded_paths


def expand_range_pattern(module_path: str) -> List[str]:
    """
    Expand range pattern like {0-20,25,30-40} in module path.
    """
    pattern_regex = re.compile(r'^(.*?){(.*?)}(.*)$')
    re_match = pattern_regex.match(module_path)

    before_pattern = re_match.group(1).rstrip('.')
    pattern_content = re_match.group(2)
    after_pattern = re_match.group(3).lstrip('.')

    indices = set()
    parts = pattern_content.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like 0-20
            start, end = part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            indices.update(range(start, end + 1))
        else:
            # Single number
            indices.add(int(part))
    indices = [str(i) for i in sorted(indices)]

    expanded_paths = []
    for idx in indices:
        new_path = f"{before_pattern}.{idx}"
        if after_pattern:
            new_path += f".{after_pattern}"
        expanded_paths.append(new_path)

    return expanded_paths


def _validate_pattern(module_path: str) -> None:
    """
    Validate pattern string to ensure at most one pair of braces {}.
    """
    start_count = module_path.count("{")
    end_count = module_path.count("}")

    if start_count > 1 or end_count > 1:
        raise ValueError(
            f"Configuration string '{module_path}' contains multiple brace pairs. Only one pair is allowed per line")

    if start_count != end_count:
        raise ValueError(f"Configuration string '{module_path}' has mismatched braces")

    if start_count == 1:
        start_idx = module_path.find('{')
        end_idx = module_path.find('}')
        if end_idx <= start_idx:
            raise ValueError(f"Configuration string '{module_path}' has invalid brace order")


def get_submodules_by_path(model: nn.Module, config: Iterable[str]) -> List[nn.Module]:
    """
    Get submodules from model based on configuration paths.

    Args:
        model: The target model
        config: Configuration list containing module paths with patterns

    Returns:
        List of unique submodules in order of appearance
    """
    if config is None:
        return None
    model = unwrap_model(model)

    submodules = []

    for module_path in config:
        _validate_pattern(module_path)

        # Handle simple paths without patterns
        if "{" not in module_path and "}" not in module_path:
            target_module = get_nested_module(model, module_path)
            if target_module is not None:
                submodules.append(target_module)
            else:
                print_rank_0(f"Warning: Module not found at path '{module_path}'")
        else:
            if "{*}" in module_path:
                expanded_paths = expand_wildcard_pattern(model, module_path)
            else:
                expanded_paths = expand_range_pattern(module_path)

            # Get corresponding modules
            for path in expanded_paths:
                target_module = get_nested_module(model, path)
                if target_module is not None:
                    submodules.append(target_module)
                else:
                    print_rank_0(f"Warning: Module not found at expanded path '{path}'")

    # Remove duplicates while preserving order
    unique_submodules = []
    seen_modules = set()
    for module in submodules:
        module_id = id(module)
        if module_id not in seen_modules:
            seen_modules.add(module_id)
            unique_submodules.append(module)

    return unique_submodules
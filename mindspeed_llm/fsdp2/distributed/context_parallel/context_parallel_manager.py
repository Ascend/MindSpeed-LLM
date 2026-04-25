import importlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set, Counter
from collections import defaultdict, Counter

import torch
import torch.distributed as dist

from mindspeed_llm.fsdp2.distributed.context_parallel.context_parallel_mappings import COMMON_CP_MAPPINGS, MODEL_CP_CONFIG
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import CPPlanConfig


@dataclass
class CPTypeConfig:
    cp_type: str
    cp_specific_mappings: Optional[List[Tuple[str, str]]] = None

    def __post_init__(self):
        self.cp_specific_mappings = self.cp_specific_mappings or []


class ContextParallelMappingManager:
    COMMON_CP_MAPPINGS = COMMON_CP_MAPPINGS
    MODEL_CP_CONFIG = MODEL_CP_CONFIG

    def __init__(self):
        self.model_id_to_att_type: Dict[str, str] = self._generate_model_id_mapping()

    def _generate_model_id_mapping(self) -> Dict[str, str]:
        mapping = {}
        for att_type, config in self.MODEL_CP_CONFIG.items():
            for model_id in config["models"]:
                mapping[model_id] = att_type
        return mapping

    def detect_duplicate_mappings(self, mappings: List[Tuple[str, str]], model_id: str) -> None:
        if not mappings:
            return

        tuple_counter = Counter(mappings)
        duplicate_tuples = {item: cnt for item, cnt in tuple_counter.items() if cnt > 1}
        if duplicate_tuples:
            duplicate_info = "\n  - ".join([f"{item} (appeared {cnt} times)" for item, cnt in duplicate_tuples.items()])
            warning_msg = f"[Model {model_id}] Detected duplicate mapping tuples:\n  - {duplicate_info}"
            raise ValueError(f"WARNING: {warning_msg}")

        target_counter = defaultdict(list)
        for idx, (target, patch) in enumerate(mappings):
            target_counter[target].append((idx, patch))

        duplicate_targets = {t: patches for t, patches in target_counter.items() if len(patches) > 1}
        if duplicate_targets:
            duplicate_info = []
            for target, patches in duplicate_targets.items():
                patch_info = ", ".join([f"Mapping #{idx + 1}: {patch}" for idx, patch in patches])
                duplicate_info.append(f"{target} -> [{patch_info}]")
            warning_msg = f"[Model {model_id}] Detected duplicate target function (same target replaced multiple times):\n  - " + "\n  - ".join(
                duplicate_info)
            raise ValueError(f"ERROR: {warning_msg}")

    def _get_model_attention_config(self, model_id: str) -> Tuple[str, List[Tuple[str, str]]]:
        # ====================== Core validation: raise error if model_id is not supported ======================
        if model_id not in self.model_id_to_att_type:
            supported_models = sorted(list(self.model_id_to_att_type.keys()))
            raise ValueError(
                f"Unsupported model type: '{model_id}'\n"
                f"Context Parallel only supports the following models: {', '.join(supported_models)}"
            )

        att_type = self.model_id_to_att_type[model_id]
        model_specific_mappings = self.MODEL_CP_CONFIG[att_type]["model_specific"].get(model_id, [])
        return att_type, model_specific_mappings

    def get_model_cp_mappings(self, model_id: str, cp_type_config: CPTypeConfig) -> List[Tuple[str, str]]:
        """
        Core method: Generate mappings with override logic.
        Priority: model_specific > common templates
        """
        # Use dict to ensure unique targets (auto-override)
        mapping_dict: Dict[str, str] = {}
        cp_type = cp_type_config.cp_type

        # 1. Get model configuration
        att_type, model_specific_mappings = self._get_model_attention_config(model_id)

        # ======================== Step 1: Load common templates first ========================
        cp_att_templates = self.MODEL_CP_CONFIG[att_type]["cp_to_att_template"]
        for template_name in cp_att_templates:
            template = self.COMMON_CP_MAPPINGS[template_name]

            # Normalize to list
            mappings_to_add = template if isinstance(template, list) else [template]

            for target, patch in mappings_to_add:
                # Fill model_id placeholder
                target_filled = target.format(model_id=model_id) if "{model_id}" in target else target
                mapping_dict[target_filled] = patch

        # ======================== Step 2: Load model_specific (OVERRIDE if conflict) ========================
        for target, patch in model_specific_mappings:
            target_filled = target.format(model_id=model_id) if "{model_id}" in target else target

            # Check for conflict and print override message
            if target_filled in mapping_dict:
                old_patch = mapping_dict[target_filled]
                print(f"[Model {model_id}] OVERRIDE CONFLICT")
                print(f"             Target:    {target_filled}")
                print(f"             Old patch: {old_patch}")
                print(f"             New patch: {patch}")
                print(f"             Reason:    model_specific takes precedence over common templates\n")

            # Override (or add if new)
            mapping_dict[target_filled] = patch

        # ======================== Step 3: Convert back to list and add CP-specific mappings ========================
        final_mappings = [(target, patch) for target, patch in mapping_dict.items()]
        final_mappings.extend(cp_type_config.cp_specific_mappings)

        # Optional: Keep duplicate detection (though dict ensures unique targets)
        self.detect_duplicate_mappings(final_mappings, model_id)

        return final_mappings

    def apply_transformers_modules(self, modules: torch.nn.Module, cp_type_config: CPTypeConfig) -> None:
        model_id = modules.config.model_type
        model_patch_list = self.get_model_cp_mappings(model_id, cp_type_config)

        for target_name, func_patch_name in model_patch_list:
            # ========== Core Modification Part ==========
            # 1. Handle the replacement logic for target function/method
            # Split rule: The part before the last dot is "module.class/function",
            # the last dot is the method name (if it's a class method)
            parts = target_name.rsplit(".", 1)
            if len(parts) == 2:
                obj_path, method_name = parts
            else:
                obj_path = target_name
                method_name = None

            # Split module path and target object name (class/function)
            obj_parts = obj_path.rsplit(".", 1)
            if len(obj_parts) == 2:
                module_path, obj_name = obj_parts
            else:
                # If there's no class name, it's a direct function in the module
                module_path = obj_path
                obj_name = None

            # 2. Import the target module
            try:
                target_module = importlib.import_module(module_path)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"Failed to import module '{module_path}' for target '{target_name}'. "
                    f"Original error: {str(e)}"
                )

            # 3. Get the target object (class/function) and replace the method/function
            if obj_name:
                # Target is a class method
                target_obj = getattr(target_module, obj_name)
                # Import the function to be replaced
                patch_parts = func_patch_name.rsplit(".", 1)
                patch_module_path = patch_parts[0]
                patch_func_name = patch_parts[1]
                patch_module = importlib.import_module(patch_module_path)
                patch_func = getattr(patch_module, patch_func_name)

                # Replace the class method
                setattr(target_obj, method_name, patch_func)
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"CP Applied patch: {target_name} -> {func_patch_name}")
            else:
                # Target is a direct function in the module
                target_func_name = obj_path.split(".")[-1]
                patch_parts = func_patch_name.rsplit(".", 1)
                patch_module_path = patch_parts[0]
                patch_func_name = patch_parts[1]
                patch_module = importlib.import_module(patch_module_path)
                # Replace the function in the module
                target_module.__dict__[target_func_name] = patch_module.__dict__[patch_func_name]
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"CP Applied patch: {target_name} -> {func_patch_name}")

    @classmethod
    def cp_parallelize_modules(cls, modules: torch.nn.Module, plan: CPPlanConfig) -> None:
        if plan.context_parallel_type not in ["ulysses", "ring"]:
            raise ValueError(f"Unsupported CP type: {plan.context_parallel_type}, only 'ulysses'/'ring' are supported")

        manager = cls()
        cp_type_config = CPTypeConfig(cp_type=plan.context_parallel_type)
        manager.apply_transformers_modules(modules, cp_type_config)


def apply_context_parallelize_modules(modules: torch.nn.Module, plan: CPPlanConfig):
    """Compatibility wrapper for original ulysses function"""
    ContextParallelMappingManager.cp_parallelize_modules(modules, plan)

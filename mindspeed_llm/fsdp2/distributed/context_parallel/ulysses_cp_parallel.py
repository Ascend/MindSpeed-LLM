import importlib
import torch

from mindspeed_llm.fsdp2.distributed.parallel_engine_config import CPPlanConfig

MODEL_CP_MAPPING = {
    "gpt_oss":
        [("transformers.models.gpt_oss.modeling_gpt_oss.eager_attention_forward",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.flash_attention_forward_fa"),
         ("mindspeed_llm.fsdp2.models.gpt_oss.modeling_gpt_oss.flash_attention_forward",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.flash_attention_forward_fa"),
         ("transformers.loss.loss_utils.fixed_cross_entropy",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.fixed_cross_entropy_with_cp"),
         ],
    "qwen3_moe":
        [("transformers.models.qwen3_moe.modeling_qwen3_moe.eager_attention_forward",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.flash_attention_forward_fa_gqa"),
         ("transformers.loss.loss_utils.fixed_cross_entropy",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.fixed_cross_entropy_with_cp")],
    "qwen2":
        [("transformers.models.qwen2.modeling_qwen2.eager_attention_forward",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.flash_attention_forward_fa_gqa"),
         ("transformers.loss.loss_utils.fixed_cross_entropy",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.fixed_cross_entropy_with_cp"
          )
         ],
    "deepseek_v3":
        [("transformers.models.deepseek_v3.modeling_deepseek_v3.eager_attention_forward",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.flash_attention_forward_fa_gqa"),
         ("transformers.loss.loss_utils.fixed_cross_entropy",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.fixed_cross_entropy_with_cp"
          )
         ],
    "glm_moe_dsa":
        [("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.eager_attention_forward",
          "mindspeed_llm.fsdp2.distributed.context_parallel.dsa_attention.flash_attention_forward_fa_dsa"),
         ("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa.GlmMoeDsaAttention.forward",
          "mindspeed_llm.fsdp2.distributed.context_parallel.dsa_attention.dsa_forward"),
         ("transformers.masking_utils.sdpa_mask",
          "mindspeed_llm.fsdp2.distributed.context_parallel.dsa_attention.sdpa_mask"),
         ("transformers.loss.loss_utils.fixed_cross_entropy",
          "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.fixed_cross_entropy_with_cp")
         ]

}


def ulysses_parallelize_modules(modules: torch.nn.Module, plan: CPPlanConfig):
    if plan.context_parallel_type == "ulysses":
        apply_transformers_modules(modules)


def apply_transformers_modules(modules):
    model_type = modules.config.model_type
    if model_type not in MODEL_CP_MAPPING:
        supported_models = list(MODEL_CP_MAPPING.keys())
        raise ValueError(
            f"Context parallel does not support model type '{model_type}'. "
            f"Supported model types: {supported_models}"
        )

    model_patch_list = MODEL_CP_MAPPING[model_type]
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
        else:
            # Target is a direct function in the module
            target_func_name = obj_path.split(".")[-1]
            patch_parts = func_patch_name.rsplit(".", 1)
            patch_module_path = patch_parts[0]
            patch_func_name = patch_parts[1]
            patch_module = importlib.import_module(patch_module_path)
            # Replace the function in the module
            target_module.__dict__[target_func_name] = patch_module.__dict__[patch_func_name]




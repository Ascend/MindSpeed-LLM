import importlib
import torch


from mindspeed_llm.fsdp2.distributed.parallel_engine_config import CPPlanConfig



MODEL_CP_MAPPING = {
    "gpt_oss":
        [ ("transformers.models.gpt_oss.modeling_gpt_oss.eager_attention_forward",
           "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.flash_attention_forward_fa"),
         ("mindspeed_llm.fsdp2.models.gpt_oss.modeling_gpt_oss.flash_attention_forward",
           "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.flash_attention_forward_fa"),
          ("transformers.loss.loss_utils.fixed_cross_entropy",
           "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_function.fixed_cross_entropy_with_cp"),
        ],

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
        target_module = importlib.import_module(target_name.rsplit(".",1)[0])
        target_func_name = target_name.rsplit(".",1)[-1]
        func_module = importlib.import_module(func_patch_name.rsplit(".",1)[0])
        func_name = func_patch_name.rsplit(".",1)[-1]
        target_module.__dict__[target_func_name] = func_module.__dict__[func_name]


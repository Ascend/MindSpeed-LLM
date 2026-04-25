from typing import Tuple, Dict, List

"""
Global Common Mapping Templates
===============================
Purpose: Reusable mappings for all models. DO NOT modify unless adding new templates.

Each template is identified by a unique name and maps original functions to their
Context Parallel replacements.

Fields:
  - Key: Template name (referenced in MODEL_CP_CONFIG -> cp_to_att_template)
  - Value: Either a single tuple or a list of tuples
      - Single tuple (target_path, patch_path): One original function to replace
      - List of tuples: Multiple original functions to replace

Example:
    "template_name": (
        "original.module.function_name",      # Function to replace
        "patch.module.new_function"          # Replacement function
    )

Available templates:
  fixed_cross_entropy   - Replaces cross entropy loss with CP version
  cp_attention_unified - Replaces attention forward with unified CP version
  dsa_attention      - Replaces attention for DSA models
"""

COMMON_CP_MAPPINGS: Dict[str, Tuple[str, str] | List[Tuple[str, str]]] = {
    "fixed_cross_entropy": (
        "transformers.loss.loss_utils.fixed_cross_entropy",
        "mindspeed_llm.fsdp2.distributed.context_parallel.context_parallel_functions.fixed_cross_entropy_with_cp"
    ),
    "cp_attention_unified": (
        "transformers.models.{model_id}.modeling_{model_id}.eager_attention_forward",
        "mindspeed_llm.fsdp2.distributed.context_parallel.context_parallel_functions.context_parallel_attention_forward"
    ),
    "dsa_attention": [
        (
            "transformers.models.{model_id}.modeling_{model_id}.eager_attention_forward",
            "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_context_parallel.dsa_attention.flash_attention_forward_fa_dsa"
        ),
        (
            "transformers.masking_utils.sdpa_mask",
            "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_context_parallel.dsa_attention.sdpa_mask"
        )
    ]
}

"""
Model Group Configuration
==================
Purpose: Group models by type, bind common templates, and define model-specific overrides.

Each model group contains:
  models: Set of supported model names that use this group configuration
  cp_to_att_template: Template names from COMMON_CP_MAPPINGS to apply
  model_specific: Per-model overrides that take highest priority over common templates

Configuration fields explained:
  group_name:
    models              - Model identifiers (e.g., "gpt_oss", "qwen2")
    cp_to_att_template - List of template names to apply
    model_specific:
      model_name        - List of (target, patch) tuples for this specific model only

Priority order (highest to lowest):
  1. model_specific (per-model overrides)
  2. cp_to_att_template (common templates)
"""

MODEL_CP_CONFIG: Dict[str, Dict] = {
    # MLA GQA MHA
    "cp_attention_common": {
        "models": {"gpt_oss", "qwen3_moe", "qwen2", "deepseek_v3", "qwen3"},
        "cp_to_att_template": {"fixed_cross_entropy", "cp_attention_unified"},
        "model_specific": {
            "gpt_oss": [
                (
                    "mindspeed_llm.fsdp2.models.gpt_oss.modeling_gpt_oss.flash_attention_forward",
                    "mindspeed_llm.fsdp2.distributed.context_parallel.context_parallel_functions.context_parallel_attention_forward"
                )
            ]
        }
    },
    "dsa": {
        "models": {"glm_moe_dsa"},
        "cp_to_att_template": {"fixed_cross_entropy", "dsa_attention"},
        "model_specific": {
            "glm_moe_dsa": [
                (
                    "transformers.models.{model_id}.modeling_{model_id}.GlmMoeDsaAttention.forward",
                    "mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_context_parallel.dsa_attention.dsa_forward"
                )
            ]
        }
    }
}

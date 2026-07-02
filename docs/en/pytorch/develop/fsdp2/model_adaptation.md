# MindSpeed LLM FSDP2 Backend Model Adaptation Guide

This guide helps developers integrate new large language models (LLMs) into the MindSpeed LLM FSDP2 training backend. The repository provides two adaptation paths to meet different levels of customization needs.

---

## Adaptation Path Overview

| Path | Applicable Scenario | Complexity | Advantages |
| --- | --- | --- | --- |
| **Path 1: Native Transformers Adaptation** | Standard Hugging Face models that do not require changes to the model structure or operators. | ⭐ (low) | Zero-code development, plug and play, and automatic loading through `AutoModel`. |
| **Path 2: Custom Registry Adaptation** | Models that need changes to underlying operators, such as NPU-fused operators, a full rewrite, Monkey Patch support, or inherited overrides of `forward` logic. | ⭐⭐⭐ (medium) | Deep customization, higher performance, and support for hardware-specific optimizations, such as Ascend NPU. |

---

## Path 1: Native Transformers Adaptation

This is the fastest way to integrate a model. As long as the model is supported by the Hugging Face `transformers` library and does not require source changes, you can use it directly.

### Preparation

The model weights and configuration file (`config.json`) must follow the standard Hugging Face format.
Take Qwen3 as an example:

```json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

### Startup Configuration

In the startup script or YAML configuration, do not set the `model_id` parameter. The framework applies `AutoModelForCausalLM` logic and creates the model automatically from the `config.json` file.

```yaml
# config.yaml example.
model:
  model_name_or_path: "/path/to/your/new_model"  # Enter the path to the Hugging Face model weights and configuration files.
  trust_remote_code: true
  train_from_scratch: false # If true, initialize the weights randomly.
  # model_id:  # Remove this line.

```

### Internal Processing Flow

`ModelFactory` follows this logic:

1. Detect that `model_id` is empty.
2. Call `AutoModelForCausalLM.from_pretrained(...)` to load the model.
3. Apply the FSDP2 parallel strategy automatically for wrapping.

---

## Path 2: Custom Registry Adaptation

Use this path only when the native Transformers implementation cannot meet your needs. For example, use it when you need to inject NPU-friendly fused operators, modify the attention logic, or adapt special MoE routing.

### Defining the Model Class

Create a new model folder under `mindspeed_llm/fsdp2/models/` (for example, `custom_model`) and place the Transformers-style model file in that directory. Then complete the second-step registration. If you are developing further based on an open-source model, you can copy the native model file directly into that directory and rewrite it there.

**File Structure Example:**

```text
mindspeed_llm/fsdp2/models/
└── custom_model/
    ├── __init__.py
    └── modeling_custom_model.py

```

**Code Example (`mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py`):**

Take the gpt-oss model as an example and build on the native open-source implementation for secondary development. The example adapts expert parallelism and the GMM fused operator.

```python
class GptOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        args = get_args()
        if args.moe_grouped_gemm or args.ep_dispatcher == 'fused':
            self.experts = GptOssFusedExperts(config)   # Custom implementation, adapted for Ascend GMM fused operators and expert parallelism.
        else:
            self.experts = GptOssExperts(config)        # Transformers open-source implementation.

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        routed_out = self.experts(hidden_states, router_indices, router_scores)
        return routed_out, router_scores

```

### Registering the Model

Register the new model in the `ModelRegistry` class.

**Modify File:** `mindspeed_llm/fsdp2/model_registry.py`

```python
# 1. Import the custom class.
from mindspeed_llm.fsdp2.models.custom_model.modeling_custom_model import CustomModelForCausalLM

class ModelRegistry:
    # ...
    _REGISTRY: Dict[str, Type[Any]] = {
        "gpt_oss": GptOssForCausalLM,
        "qwen3": Qwen3ForCausalLM,
        # 2. Add the registry entry.
        "custom_model": CustomModelForCausalLM,
    }

```

### Startup Configuration

When you start training, explicitly set `model_id` to the registered key.

```yaml
# config.yaml example.
model:
  model_name_or_path: "/path/to/your/new_model"
  model_id: "custom_model"  # Activate the custom adaptation logic.

```

---

## Notes

1. **Naming Convention**: Keep `model_id` concise, such as `qwen3` or `gpt_oss`, and ensure that it matches the key in the registry dictionary exactly.

---

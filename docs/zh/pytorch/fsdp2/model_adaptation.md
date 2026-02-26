# MindSpeed LLM FSDP2后端模型适配指南

本指南旨在帮助开发者将新的大语言模型（LLM）接入 MindSpeed LLM FSDP2 训练后端。仓库提供了两种的适配路径，以满足不同层次的定制需求。

---

## 适配路径概览

| 路径 | 适用场景 | 复杂度 | 优势 |
| --- | --- | --- | --- |
| **路径一：原生 Transformers 适配** | 标准 Hugging Face 模型，无需修改模型结构或算子。 | ⭐ (低) | 零代码开发，即插即用，通过 `AutoModel` 自动加载。 |
| **路径二：自定义注册适配** | 需要修改底层算子（如 NPU 融合算子）、完全重写（推荐）、打 Monkey Patch、或通过继承重写 Forward 逻辑的模型。 | ⭐⭐⭐ (中) | 深度定制，性能更高，支持特定硬件优化（如 Ascend NPU）。 |

---

## 路径一：原生 Transformers 适配

这是最快的接入方式。只要模型在 Hugging Face `transformers` 库中受支持，且不需要修改其源码，即可直接使用。

### 准备工作

模型权重和配置文件（`config.json`）需符合 Hugging Face 标准格式。
以Qwen3为例：

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

### 启动配置

在启动脚本或 YAML 配置中，【**不要** 设置 `model_id` 参数】。框架会应用 `AutoModelForCausalLM` 逻辑，自动根据config.json文件创建模型。

```yaml
# config.yaml 示例
model:
  model_name_or_path: "/path/to/your/new_model"  #<-- 填入HF模型权重和配置文件路径
  trust_remote_code: true
  train_from_scratch: false # <-- 如果是True，随机初始化权重
  # model_id:  <-- 删除

```

### 内部处理流程

`ModelFactory` 会执行以下逻辑：

1. 检测到 `model_id` 为空。
2. 调用 `AutoModelForCausalLM.from_pretrained(...)` 加载模型。
3. 自动应用 FSDP2 并行策略进行包裹。

---

## 路径二：自定义注册适配

当且仅当原生 Transformers 实现无法满足需求（例如需要注入 NPU 亲和的融合算子、修改 Attention 逻辑或适配特殊的 MoE 路由）时，请使用此路径。

### 定义模型类

在 `mindspeed_llm/fsdp2/models/` 目录下新建模型文件夹（例如 `custom_model`），并将Transformers风格的模型文件放入该目录下，进行第二步的注册即可。如果是基于开源模型进行二次开发，可以将原生的模型文件（例如GPT-OSS：transformers/models/gpt_oss/modeling_gpt_oss.py）直接复制到该目录下，在此基础上进行改写。

**文件结构示例：**

```text
mindspeed_llm/fsdp2/models/
└── custom_model/
    ├── __init__.py
    └── modeling_custom_model.py

```

**代码示例 (`mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py`)：**

以GPT-OSS模型为例，基于原生开源实现进行二次开发：专家并行和GMM融合算子适配

```python
class GptOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        args = get_args()
        if args.moe_grouped_gemm or args.ep_dispatcher == 'fused':
            self.experts = GptOssFusedExperts(config)   # 自定义实现，适配昇腾GMM融合算子和专家并行，
        else:
            self.experts = GptOssExperts(config)        # transformers开源实现

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)  # (num_experts, seq_len)
        routed_out = self.experts(hidden_states, router_indices, router_scores)
        return routed_out, router_scores

```

### 注册模型

在 `ModelRegistry` 类中注册新模型。

**修改文件：** `mindspeed_llm/fsdp2/model_registry.py` 

```python
# 1. 导入自定义类
from mindspeed_llm.fsdp2.models.custom_model.custom_model import CustomModelForCausalLM

class ModelRegistry:
    # ...
    _REGISTRY: Dict[str, Type[Any]] = {
        "gpt_oss": GptOssForCausalLM,
        "qwen3": Qwen3ForCausalLM,
        # 2. 添加注册项
        "custom_model": CustomModelForCausalLM, 
    }

```

### 启动配置

在启动训练时，显式指定 `model_id` 刚才注册的 Key。

```yaml
# config.yaml 示例
model:
  model_name_or_path: "/path/to/your/new_model"
  model_id: "custom_model"  # <-- 激活自定义适配逻辑

```

---

## 注意事项

1. **命名规范**：建议 `model_id` 保持简洁（如 `qwen3`, `gpt_oss`），且与注册字典中的 Key 严格一致。

---

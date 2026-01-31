本指南旨在帮助开发者将新的大语言模型（LLM）接入 MindSpeed-LLM FSDP2 训练框架。框架提供了两种灵活的适配路径，以满足不同层次的定制需求。

---

## 适配路径概览

| 路径 | 适用场景 | 复杂度 | 优势 |
| --- | --- | --- | --- |
| **路径一：原生 Transformers 适配** | 标准 Hugging Face 模型，无需修改模型结构或算子。 | ⭐ (低) | 零代码开发，即插即用，通过 `AutoModel` 自动加载。 |
| **路径二：自定义注册适配** | 需要修改底层算子（如 NPU 融合算子）、完全重写（推荐）、打 Monkey Patch、或通过继承重写 Forward 逻辑的模型。 | ⭐⭐⭐ (中) | 深度定制，性能更高，支持特定硬件优化（如 Ascend NPU）。 |

---

## 路径一：原生 Transformers 适配 (Zero-Code)

这是最快的接入方式。只要模型在 Hugging Face `transformers` 库中受支持，且您不需要修改其源码，即可直接使用。

### 1. 准备工作

确保您的模型权重和配置文件（`config.json`）符合 Hugging Face 标准格式。

### 2. 启动配置

在启动脚本或 YAML 配置中，【**不要** 设置 `model_id` 参数】，或者将其留空。框架会自动回退到 `AutoModelForCausalLM` 逻辑。

```yaml
# config.yaml 示例
model:
  model_name_or_path: "/path/to/your/new_model"  #<-- 填入HF模型权重和配置文件路径
  trust_remote_code: true
  train_from_scratch: false # <-- 如果是True，随机初始化权重
  # model_id:  <-- 删除或留空

```

### 3. 框架内部流转

`ModelFactory` 会执行以下逻辑：

1. 检测到 `model_id` 为空。
2. 调用 `AutoModelForCausalLM.from_pretrained(...)` 加载模型。
3. 自动应用 FSDP2 并行策略进行包裹。

---

## 路径二：自定义注册适配 (Custom Registry)

当且仅当原生 Transformers 实现无法满足需求（例如需要注入 NPU 亲和的融合算子、修改 Attention 逻辑或适配特殊的 MoE 路由）时，请使用此路径。

### 步骤 1：定义模型类

在 `mindspeed_llm/fsdp2/models/` 目录下新建您的模型文件夹（例如 `my_new_model`），并创建一个继承自 Hugging Face 原生类的 Wrapper 类，或者将原生的模型文件（以gpt-oss为例：transformers/models/gpt_oss/modeling_gpt_oss.py）直接复制到该目录下进行二次开发。

**文件结构示例：**

```text
mindspeed_llm/fsdp2/models/
└── my_new_model/
    ├── __init__.py
    └── modeling_my_model.py

```

**代码示例 (`my_new_model.py`)：**

```python
from transformers import LlamaForCausalLM 
from mindspeed_llm.fsdp2.utils import patch_manager

class MyNewModelForCausalLM(LlamaForCausalLM):
    """
    自定义模型类，继承自 HF 原生类，用于注入特定逻辑。
    """

    @classmethod
    def register_patches(cls, model_args):
        """
        [可选] 在模型加载前执行 Monkey Patch 操作。
        例如：替换 Flash Attention，修改 RMSNorm 等。
        """
        print(f"> Applying NPU patches for {cls.__name__}...")
        # 示例：替换 Attention 实现
        # patch_manager.apply_patch(target_module, new_implementation)
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        [可选] 重写加载逻辑，如果需要特殊的初始化流程。
        通常直接调用 super() 即可。
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(self, input_ids, ...):
        """
        [可选] 如果需要完全自定义前向传播（例如修改 Loss 计算），请重写此方法。
        """
        return super().forward(input_ids, ...)

```

### 步骤 2：注册模型

在 `ModelRegistry` 类中注册您的新模型。

**修改文件：** `mindspeed_llm/fsdp2/model_factory.py` (或 registry 定义所在文件)

```python
# 1. 导入您的自定义类
from mindspeed_llm.fsdp2.models.my_new_model.my_new_model import MyNewModelForCausalLM

class ModelRegistry:
    # ...
    _REGISTRY: Dict[str, Type[Any]] = {
        "gpt_oss": GptOssForCausalLM,
        "qwen3": Qwen3ForCausalLM,
        # 2. 添加注册项
        "my_new_model": MyNewModelForCausalLM, 
    }

```

### 步骤 3：启动配置

在启动训练时，显式指定 `model_id` 为您刚才注册的 Key。

```yaml
# config.yaml 示例
model:
  model_name_or_path: "/path/to/your/new_model"
  model_id: "my_new_model"  # <-- 激活自定义适配逻辑

```

---

## 最佳实践与注意事项

1. **优先使用原生适配**：除非遇到性能瓶颈或算子不支持 NPU 的情况，否则首选路径一。这能最大程度保持与 Hugging Face 生态的同步。
2. **`register_patches` 的使用**：这是路径二的核心优势。利用此钩子函数，您可以在模型实例化**之前**修改 `transformers` 的底层函数（如 `LlamaAttention.forward`），从而在不破坏模型结构的前提下实现算子加速。
3. **命名规范**：建议 `model_id` 保持简洁（如 `llama3`, `mixtral`），且与注册字典中的 Key 严格一致。

---

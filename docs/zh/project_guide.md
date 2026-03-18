# MindSpeed LLM 项目导读

---

## 项目介绍

MindSpeed LLM 项目代码按照模块化设计原则进行组织，主要包含以下核心模块：

- **mindspeed_llm/**：核心代码目录，包含模型训练、特性管理、权重转换、在线推理和评估工具链等核心功能实现
- **docs/**：项目文档目录，提供项目介绍、快速入门、安装指南和特性说明等
- **configs/**：配置文件目录，提供权重转换、评估、微调、FSDP2和RLHF等相关配置
- **examples/**：模型示例脚本，涵盖fsdp2、mcore、mindspore和rlhf等多种训练后端和场景
- **tests/**：测试用例目录，包含单元测试、系统测试和覆盖率测试等

### 代码目录结构

项目根目录包含多个关键工具脚本，如权重转换工具、模型评估工具、训练流程等，支持从数据预处理到模型训练、评估和在线推理的全流程。项目代码目录层级详细介绍如下所示：

``` shell
MindSpeed-LLM/
 ├── ci                        # CI 测试脚本
 ├── configs                   # 配置文件目录
 │   ├── checkpoint/           # 权重相关配置
 │   ├── evaluate/             # 评估相关配置
 │   ├── finetune/             # 微调相关配置
 │   ├── fsdp2/                # FSDP2相关配置
 │   └── rlhf/                 # RLHF相关配置
 ├── docs                      # 项目文档目录
 ├── examples                  # 模型示例脚本
 │   ├── fsdp2/                # FSDP2后端示例
 │   ├── mcore/                # mcore后端示例
 │   ├── mindspore/            # MindSpore后端示例
 │   └── rlhf/                 # RLHF示例
 ├── mindspeed_llm             # 核心代码目录
 │   ├── __init__.py           # 核心模块包初始化
 │   ├── core/                 # 核心功能模块
 │   │   ├── context_parallel/         # 上下文并行
 │   │   ├── datasets/                 # 数据集处理
 │   │   ├── distributed/              # 分布式训练
 │   │   ├── high_availability/        # 高可用性
 │   │   ├── models/                   # 模型定义
 │   │   │   ├── common/               # 通用模型组件
 │   │   │   │   ├── embeddings/       # 嵌入层
 │   │   │   │   └── language_module/  # 语言模块
 │   │   │   └── gpt/                  # GPT模型实现
 │   │   ├── optimizer/           # 优化器
 │   │   ├── pipeline_parallel/   # 流水线并行
 │   │   ├── ssm/                 # 状态空间模型
 │   │   ├── tensor_parallel/     # 张量并行
 │   │   └── transformer/         # Transformer实现
 │   │       ├── custom_layers/   # 自定义层
 │   │       ├── moe/             # MoE实现
 │   │       ├── alibi_attention.py         # ALiBi注意力
 │   │       ├── attention.py               # 注意力机制
 │   │       ├── custom_dot_product_attention.py  # 自定义点积注意力
 │   │       ├── mlp.py                     # 多层感知机
 │   │       ├── multi_token_prediction.py  # 多token预测
 │   │       ├── transformer_block.py       # Transformer块
 │   │       ├── transformer_config.py      # Transformer配置
 │   │       └── transformer_layer.py       # Transformer层
 │   │   ├── __init__.py          # 核心模块初始化
 │   │   ├── fp8_utils.py         # FP8工具函数
 │   │   ├── optimizer_param_scheduler.py  # 优化器参数调度器
 │   │   ├── parallel_state.py    # 并行状态管理
 │   │   └── timers.py            # 计时器工具
 │   ├── features_manager/        # 特性管理器
 │   │   ├── __init__.py          # 特性管理器初始化
 │   │   ├── affinity/            # 亲和性优化
 │   │   ├── ai_framework/        # AI框架支持
 │   │   ├── arguments/           # 参数管理
 │   │   ├── common/              # 通用功能
 │   │   ├── context_parallel/    # 上下文并行特性
 │   │   ├── convert_checkpoint/  # 权重转换
 │   │   ├── dataset/             # 数据集特性
 │   │   ├── dpo/                 # DPO训练
 │   │   ├── evaluation/          # 评估特性
 │   │   ├── finetune/            # 微调特性
 │   │   ├── fsdp2/               # FSDP2特性
 │   │   ├── functional/          # 功能特性
 │   │   ├── high_availability/   # 高可用特性
 │   │   ├── inference/           # 推理特性
 │   │   ├── low_precision/       # 低精度训练
 │   │   ├── megatron_basic/      # Megatron基础
 │   │   ├── memory/              # 内存优化
 │   │   ├── models/              # 模型特性
 │   │   ├── moe/                 # MoE特性
 │   │   ├── pipeline_parallel/   # 流水线并行
 │   │   ├── tensor_parallel/     # 张量并行
 │   │   ├── tokenizer/           # 分词器
 │   │   └── transformer/         # Transformer特性
 │   │       ├── flash_attention/           # Flash Attention
 │   │       ├── multi_latent_attention/    # 多潜在注意力
 │   │       ├── qwen3_next_attention/      # Qwen3 Next注意力
 │   │       ├── mtp.py                     # 多token预测
 │   │       └── transformer_block.py       # Transformer块
 │   ├── legacy/                # 遗留代码模块
 │   │   ├── data/              # 遗留数据处理
 │   │   └── __init__.py        # 遗留模块初始化
 │   ├── mindspore/             # MindSpore后端模块
 │   │   ├── core/              # MindSpore核心功能
 │   │   ├── tasks/             # MindSpore任务模块
 │   │   ├── training/          # MindSpore训练模块
 │   │   ├── convert_ckpt.py    # MindSpore权重转换
 │   │   ├── convert_ckpt_v2.py # MindSpore权重转换 v2
 │   │   ├── mindspore_adaptor_v2.py  # MindSpore适配器
 │   │   └── utils.py           # MindSpore工具函数
 │   ├── tasks/                 # 任务模块
 │   │   ├── checkpoint/        # 检查点任务
 │   │   ├── common/            # 通用任务
 │   │   ├── dataset/           # 数据集任务
 │   │   ├── evaluation/        # 评估任务
 │   │   ├── high_availability/ # 高可用任务
 │   │   ├── inference/         # 推理任务
 │   │   ├── megatron_basic/    # Megatron基础任务
 │   │   ├── models/            # 模型任务
 │   │   ├── posttrain/         # 后训练任务
 │   │   ├── preprocess/        # 预处理任务
 │   │   ├── utils/             # 任务工具
 │   │   └── __init__.py        # 任务模块初始化
 │   ├── training/              # 训练模块
 │   │   ├── tokenizer/         # 分词器模块
 │   │   ├── __init__.py        # 训练模块初始化
 │   │   ├── arguments.py       # 训练参数
 │   │   ├── checkpointing.py   # 检查点管理
 │   │   ├── initialize.py      # 训练初始化
 │   │   ├── one_logger_utils.py  # 日志工具
 │   │   ├── training.py        # 训练主逻辑
 │   │   └── utils.py           # 训练工具函数
 │   └── fsdp2/                # FSDP2相关实现
 │       ├── __init__.py         # FSDP2模块初始化
 │       ├── checkpoint/        # 权重管理
 │       ├── data/              # 数据处理
 │       │   ├── megatron_data/   # Megatron数据集
 │       │   └── processor/       # 数据处理器
 │       ├── distributed/       # 分布式训练
 │       │   ├── context_parallel/   # 上下文并行
 │       │   ├── expert_parallel/    # 专家并行
 │       │   └── fully_shard/        # 完全分片
 │       ├── features/          # FSDP2特性
 │       ├── models/            # 模型实现
 │       │   ├── common/          # 通用模型组件
 │       │   ├── gpt_oss/         # GPT OSS模型
 │       │   ├── qwen3/           # Qwen3模型
 │       │   ├── qwen3_next/      # Qwen3 Next模型
 │       │   └── step35/          # Step3.5模型
 │       ├── optim/             # 优化器
 │       ├── train/             # 训练器
 │       └── utils/             # 工具函数
 ├── tests                     # 测试用例目录
 ├── convert_ckpt.py           # 权重转换工具
 ├── convert_ckpt_v2.py        # 权重转换工具 v2
 ├── evaluation.py             # 模型评估工具
 ├── inference.py              # 模型推理工具
 ├── posttrain_gpt.py          # 后训练流程
 ├── pretrain_gpt.py           # 预训练流程
 ├── pretrain_mamba.py         # 预训练流程
 ├── preprocess_data.py        # 数据预处理工具
 ├── preprocess_prompt.py      # 提示词预处理工具
 ├── rlhf_gpt.py               # RLHF 训练流程
 ├── setup.py                  # 安装配置文件
 ├── train_fsdp2.py            # FSDP2 训练流程
 ├── requirements.txt          # Python依赖文件
 ├── LICENSE                   # 许可证文件
 ├── OWNERS                    # 维护者列表
 ├── README.md                 # 项目说明文档
 ├── SECURITYNOTE.md           # 安全声明
 └── Third_Party_Open_Source_Software_Notice  # 第三方开源软件声明
```

### 模型运行样例

`examples/` 目录包含丰富的模型运行示例脚本，涵盖不同训练框架、模型架构和训练场景，帮助用户快速上手和使用 MindSpeed LLM。该目录运行样例分类如下：

``` shell
examples/
├── fsdp2/                # FSDP2 训练后端
│   ├── gpt_oss/          # GPT OSS 模型示例
├── mcore/                # mcore 训练后端
│   ├── qwen3/            # Qwen3 模型示例，包含预训练、微调、评估等脚本
├── mindspore/            # MindSpore 训练框架
│   ├── qwen25/           # Qwen2.5 MindSpore 示例，包含预训练、微调、评估脚本
└── rlhf/                 # RLHF 相关示例，包含数据预处理和训练脚本
```

每个训练框架下都提供了核心模型的完整训练脚本，用户可以根据需求选择相应的脚本进行训练：

1. FSDP2 训练后端GPT OSS 模型示例

    - 微调：运行 `examples/fsdp2/gpt_oss/tune_gpt_oss_20b_a3b_4K_fsdp2_mindspeed.sh` 脚本进行模型微调    
    - 详细指南：参考 [finetune_fsdp2.md](pytorch/training/finetune/fsdp2/finetune_fsdp2.md) 获取完整使用说明

2. mcore 训练后端 Qwen3 8B模型预训练示例

    - 数据处理：运行 `data_convert_qwen3_pretrain.sh` 脚本进行预训练数据处理
    - 预训练：运行 `pretrain_qwen3_8b_4k.sh` 脚本进行模型预训练
    - 详细指南：参考 [pretrain.md](pytorch/training/pretrain/mcore/pretrain.md) 获取完整使用说明

3. mcore 训练后端Qwen3 8B模型微调示例

    - 数据处理：运行 `data_convert_qwen3_instruction.sh` 脚本进行微调数据处理
    - 权重转换：运行 `ckpt_convert_qwen3_hf2mcore.sh` 脚本进行权重转换
    - 全参微调：运行 `tune_qwen3_8b_4k_full.sh` 脚本进行全参数微调
    - LoRA微调：运行 `tune_qwen3_8b_4k_lora.sh` 脚本进行LoRA微调
    - 详细指南：参考 [single_sample_finetune.md](pytorch/training/finetune/mcore/single_sample_finetune.md) 获取完整使用说明

4. mcore 训练后端Qwen3 模型工具链示例

    - 在线推理：运行 `generate_qwen3_8b_ptd.sh` 脚本进行模型在线推理
    - 评估：运行 `evaluate_qwen3_8b.sh` 脚本进行模型评估

5. MindSpore 训练框架Qwen2.5 预训练模型示例

    - 数据处理：运行 `data_convert_qwen25_pretrain.sh` 脚本进行预训练数据处理
    - 预训练：运行 `pretrain_qwen25_7b_32k_ms.sh` 脚本进行模型预训练

所有示例脚本都提供了完整的命令行参数和配置示例，用户可以根据自己的需求进行修改和扩展。

## 总结

MindSpeed LLM 项目提供了一个完整的大语言模型训练解决方案，具有以下核心特点：

- **多框架支持**：同时支持 PyTorch（含 mcore 和 fsdp2 两种训练后端）和 MindSpore 训练框架
- **模块化设计**：代码按照功能模块进行组织，便于维护和扩展
- **丰富的模型支持**：涵盖 dense、MoE 、SSM 、Linear等多种模型架构，支持主流开源大模型
- **完整的工具链**：提供从数据预处理、模型训练、评估到在线推理的全流程工具
- **完善的文档体系**：按照训练框架和功能模块组织文档，便于用户快速上手

项目通过清晰的目录结构、详细的文档说明和丰富的示例脚本，帮助开发者高效地进行大语言模型的训练、微调和部署工作。

# MindSpeed LLM 项目导读

---

## 代码介绍

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
    - 详细指南：参考 [fsdp2_quickstart.md](pytorch/fsdp2/fsdp2_quickstart.md) 获取完整使用说明

2. mcore 训练后端 Qwen3 8B模型预训练示例

    - 数据处理：运行 `data_convert_qwen3_pretrain.sh` 脚本进行预训练数据处理
    - 预训练：运行 `pretrain_qwen3_8b_4k.sh` 脚本进行模型预训练
    - 详细指南：参考 [pretrain.md](pytorch/solutions/pretrain/pretrain.md) 获取完整使用说明

3. mcore 训练后端Qwen3 8B模型微调示例

    - 数据处理：运行 `data_convert_qwen3_instruction.sh` 脚本进行微调数据处理
    - 权重转换：运行 `ckpt_convert_qwen3_hf2mcore.sh` 脚本进行权重转换
    - 全参微调：运行 `tune_qwen3_8b_4k_full.sh` 脚本进行全参数微调
    - LoRA微调：运行 `tune_qwen3_8b_4k_lora.sh` 脚本进行LoRA微调
    - 详细指南：参考 [single_sample_finetune.md](pytorch/solutions/finetune/single_sample_finetune.md) 获取完整使用说明

4. mcore 训练后端Qwen3 模型工具链示例

    - 在线推理：运行 `generate_qwen3_8b_ptd.sh` 脚本进行模型在线推理
    - 评估：运行 `evaluate_qwen3_8b.sh` 脚本进行模型评估

5. MindSpore 训练框架Qwen2.5 预训练模型示例

    - 数据处理：运行 `data_convert_qwen25_pretrain.sh` 脚本进行预训练数据处理
    - 预训练：运行 `pretrain_qwen25_7b_32k_ms.sh` 脚本进行模型预训练

所有示例脚本都提供了完整的命令行参数和配置示例，用户可以根据自己的需求进行修改和扩展。

## 文档介绍

MindSpeed LLM 文档按照不同的训练框架进行组织，主要包含以下核心目录：

- **pytorch/**：基于 PyTorch 训练框架的文档，支持 mcore 和 fsdp2 两种训练后端，包含安装指南、模型清单、特性说明、训练方案和工具链等
- **mindspore/**：基于 MindSpore 训练框架的文档，提供 MindSpore 框架下的使用指南和特性说明

### 文档目录结构

pytorch文档目录层级介绍如下：

``` shell
docs/zh/pytorch

├── introduction.md         # 项目介绍
├── guide.md               # 文档导航
├── quick_start.md          # 快速入门指南
├── pytorch/                # PyTorch 训练框架相关文档
│   ├── install_guide.md    # PyTorch 安装指南
│   ├── develop/            # 开发相关文档（mcore 训练后端）
│   ├── features/           # PyTorch 特性文档（mcore 训练后端）
│   ├── fsdp2/              # PyTorch FSDP2 训练后端文档
│   ├── models/             # PyTorch 支持的模型
│   └── solutions/          # 解决方案文档（mcore 训练后端）
│       ├── checkpoint/     # 权重转换
│       ├── evaluation/     # 模型评估
│       ├── finetune/       # 模型微调
│       ├── inference/      # 模型推理
│       ├── preference-alignment/ # 偏好对齐
│       └── pretrain/       # 预训练
└── mindspore/              # MindSpore 训练框架相关文档
    ├── install_guide.md    # MindSpore 安装指南
    ├── readme.md           # MindSpore 文档说明
    ├── features/           # MindSpore 特性文档
    └── models/             # MindSpore 支持的模型
├── appendixes.md           # 附录文档
```

### 核心文档导航

<table>
  <thead>
    <tr>
      <th>内容</th>
      <th>链接</th>
      <th>备注</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">环境安装指导</td>
      <td><a href="pytorch/install_guide.md">install_guide.md</a></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1">快速入门</td>
      <td><a href="quick_start.md">quick_start.md</a></td>
      <td>基于pytorch/mindspore后端的入门指导，从环境安装到预训练拉起</td>
    </tr>
    <tr>
      <td rowspan="3">仓库支持模型清单</td>
      <td><a href="pytorch/models/dense_model.md">dense_models</a></td>
      <td>稠密模型清单</td>
    </tr>
    <tr>
      <td><a href="pytorch/models/moe_model.md">MOE_models</a></td>
      <td>MOE模型清单</td>
    </tr>
    <tr>
      <td><a href="pytorch/models/ssm_model.md">SSM_models</a></td>
      <td>SSM模型清单</td>
    </tr>
    <tr>
      <td rowspan="1">特性清单</td>
      <td rowspan="1"><a href="pytorch/features">features</a></td>
      <td rowspan="1">收集了部分仓库支持的性能优化和显存优化的特性</td>
    </tr>
    <tr>
      <td rowspan="4">模型前处理操作</td>
      <td rowspan="1"><a href="pytorch/solutions/checkpoint/checkpoint_convert.md">checkpoint_convert</a></td>
      <td rowspan="1">支持mcore、hf、lora等各种不同格式权重间的部分转换路径</td>
    </tr>
    <tr>
      <td><a href="pytorch/solutions/checkpoint/checkpoint_convert_v2.md">checkpoint_convert_v2</a></td>
      <td>支持大参数模型mcore、hf等各种不同格式权重间的转换</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/pretrain/pretrain_dataset.md">pretrain_dataset</a></td>
      <td rowspan="2">支持的数据集格式包括<a href="pytorch/solutions/finetune/datasets/alpaca_dataset.md">alpaca</a>,<a href="pytorch/solutions/finetune/datasets/sharegpt_dataset.md">sharegpt</a></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/datasets">finetune_dataset</a></td>
    </tr>
    <tr>
      <td rowspan="2">预训练方案</td>
      <td><a href="pytorch/solutions/pretrain/pretrain.md">pretrain</a></td>
      <td>多样本预训练方法</td>
    </tr>
    <tr>
      <td><a href="pytorch/solutions/pretrain/pretrain_eod.md">pretrain_eod</a></td>
      <td>多样本pack预训练方法</td>
    </tr>
    <tr>
      <td rowspan="5">模型微调方法</td>
      <td rowspan="1"><a href="pytorch/solutions/finetune/instruction_finetune.md">instruction_finetune</a></td>
      <td rowspan="1">模型全参微调方案</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/multi_sample_pack_finetune.md">multi_sample_pack_finetune</a></td>
      <td rowspan="1">多样本Pack微调方案</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/multi-turn_conversation.md">multi-turn_conversation</a></td>
      <td rowspan="1">多轮对话微调方案</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/lora_finetune.md">lora_finetune</a></td>
      <td rowspan="1">模型lora微调方案</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/finetune/qlora_finetune.md">qlora_finetune</a></td>
      <td rowspan="1">模型qlora微调方案</td>
    </tr>
    <tr>
      <td rowspan="3">模型推理方法</td>
      <td rowspan="1"><a href="pytorch/solutions/inference/inference.md">inference</a></td>
      <td rowspan="1">模型推理</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/solutions/inference/chat.md">chat</a></td>
      <td rowspan="1">对话</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="pytorch/features/yarn.md">yarn</a></td>
      <td rowspan="1">使用yarn方案来扩展上下文长度，支持长序列推理</td>
    </tr>
    <tr>
      <td rowspan="3">模型评估</td>
      <td><a href="pytorch/solutions/evaluation/evaluation_guide.md">evaluation</a></td>
      <td rowspan="1">模型评估方案</td>
    </tr>
    <tr>
      <td><a href="pytorch/models/models_evaluation.md">evaluation_baseline</a></td>
      <td rowspan="1">仓库模型评估清单</td>
    </tr>
    <tr>
      <td><a href="pytorch/solutions/evaluation/evaluation_datasets">evaluation_datasets</a></td>
      <td rowspan="1">仓库支持评估数据集</td>
    </tr>
  </tbody>
</table>

## 总结

MindSpeed LLM 项目提供了一个完整的大语言模型训练解决方案，具有以下核心特点：

- **多框架支持**：同时支持 PyTorch（含 mcore 和 fsdp2 两种训练后端）和 MindSpore 训练框架
- **模块化设计**：代码按照功能模块进行组织，便于维护和扩展
- **丰富的模型支持**：涵盖 dense、MoE 、SSM 、Linear等多种模型架构，支持主流开源大模型
- **完整的工具链**：提供从数据预处理、模型训练、评估到在线推理的全流程工具
- **完善的文档体系**：按照训练框架和功能模块组织文档，便于用户快速上手

项目通过清晰的目录结构、详细的文档说明和丰富的示例脚本，帮助开发者高效地进行大语言模型的训练、微调和部署工作。

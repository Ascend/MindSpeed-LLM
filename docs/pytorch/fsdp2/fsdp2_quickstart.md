# MindSpeed LLM FSDP2 后端训练使用指南

本文档以 **微调场景** GPT-OSS 20B 为例，介绍如何使用 MindSpeed LLM 的 FSDP2 后端进行大语言模型训练，涵盖环境准备、配置说明与训练启动全流程。


## 1. 环境安装

环境安装请参考 [安装指导](../install_guide.md)

## 2 目录结构  
拉起微调相关脚本位置如下所示
```bash
MindSpeed-LLM/
├── examples/fsdp2/gpt_oss/
│   ├── tune_gpt_oss_20b_a3b_4K_fsdp2.yaml        # 微调配置
│   └── tune_gpt_oss_20b_a3b_4K_fsdp2_mindspeed.sh # 启动脚本
├── configs/fsdp2/data/dataset_info.json           # 数据集注册表
└── train_fsdp2.py                                 # FSDP2 训练入口
```

## 3. 配置修改

### 3.1 模型路径配置  
```yaml
model:
  model_name_or_path: /path/to/gpt-oss-20b-hf/      # 替换为您的模型本地路径或Hugging Face模型ID
  tokenizer_name_or_path: None                      # 模型与Tokenizer路径不一致时需指定
```

### 3.2 数据集配置  

#### 方式一：内联配置（单数据集快速验证）

```yaml
  dataset:
    file_name: "./my_data.json"                     # 替换为数据文件路径
    formatting: "alpaca"                            # 根据数据格式选择alpaca/sharegpt等
  cutoff_len: 2048                                  # 分词后输入序列的截断长度，超过该长度的序列会被截断
```

#### 方式二：通过 `dataset_info.json` 注册

1. 编辑 `configs/fsdp2/data/dataset_info.json`，添加数据集条目：

```json
{
  "alpaca_full": {
    "file_name": "./train-00000-of-00001.parquet"
  },
  "sharegpt4_zh": {
    "file_name": "./sharegpt_zh.jsonl",
    "formatting": "sharegpt"
  }
}
```

2. 在 YAML 配置中引用：

```yaml
data:
  dataset: alpaca_full, sharegpt4_zh                # 微调数据集：可填写逗号分隔的数据集名称，支持多数据集混合
  template: gpt                                     # 微调构建prompt的模板名称
  cutoff_len: 2048                                  # 分词后输入序列的截断长度，超过该长度的序列会被截断
```

## 4. 分布式训练启动脚本说明

`examples/fsdp2/gpt_oss/tune_gpt_oss_20b_a3b_4K_fsdp2_mindspeed.sh`  

```bash
source examples/fsdp2/env_config.sh                 # 加载NPU环境变量配置

NPUS_PER_NODE=8                                     # 每节点NPU卡数
NNODES=1                                            # 总节点数
MASTER_ADDR=localhost                               # 主节点IP地址
MASTER_PORT=6499                                    # 主节点通信端口

torchrun \
  --nproc_per_node $NPUS_PER_NODE \
  # 每节点启动8个进程
  --nnodes $NNODES \
  # 总共1个节点
  --node_rank 0 \
  # 当前节点序号（多机训练时需调整）
  --master_addr $MASTER_ADDR \
  # 主节点地址
  --master_port $MASTER_PORT \
  # 主节点端口
  train_fsdp2.py examples/fsdp2/gpt_oss/tune_gpt_oss_20b_a3b_4K_fsdp2.yaml  
  # 启动训练入口
```

## 5. 启动训练
在仓库根目录执行：
```bash
bash examples/fsdp2/gpt_oss/tune_gpt_oss_20b_a3b_4K_fsdp2_mindspeed.sh
```
即可成功拉起训练任务。


## 6. 其他配置参数说明
### 6.1 模型配置
```yaml
model:
  model_id: gpt_oss                                 # 模型类型标识，新增模型类型需在 mindspeed_llm/fsdp2/models/model_registry.py 的ModelRegistry类中注册
  model_name_or_path: /path/to/gpt-oss-20b-hf/      # 模型的本地路径（必填项，未指定会抛出异常）
  trust_remote_code: False                          # 是否允许加载Hugging Face上自定义建模文件中的模型，用于适配自定义模型架构
  train_from_scratch: False                         # 是否使用随机权重从头开始训练模型，不加载模型权重。
  tokenizer_name_or_path: None                      # Tokenizer的路径或名称，与model_name_or_path路径不一致时需要指定
```

### 6.2 并行策略
```yaml
parallel:
  fsdp_size: 8                                      # 全分片数据并行(FSDP)大小，将模型参数分片存储到8张卡
  fsdp_modules:                                     # 启用FSDP的模型层结构列表（必填项，不可为空）
    - model.layers.{*}                              # 所有Transformer层启用FSDP分片
    - model.embed_tokens                            # 词嵌入层启用FSDP分片
    - lm_head                                       # 语言模型输出头启用FSDP分片
  tp_size: 1                                        # 张量并行(Tensor Parallel)大小，将模型张量按列/行拆分到多张卡
  ep_size: 1                                        # 专家并行(Expert Parallel)大小，适用于MoE模型，将不同专家拆分到多张卡
  ep_modules:                                       #  启用专家并行的模型层结构，仅适用于MoE模型
    - model.layers.{*}.mlp.experts                  # 所有层的专家模块启用专家并行
  ep_fsdp_size: 1                                   # 专家并行组内的FSDP大小，在专家并行基础上对单个专家参数进行分片
  ep_fsdp_modules:                                  # 专家并行组内启用FSDP的模型层结构
    - model.layers.{*}.mlp.experts                  # 专家模块内部参数进一步分片
  ep_dispatcher: eager                              # MoE专家并行的调度策略：eager(立即分发)/fused(融合计算)/mc2(混合压缩分发)
  recompute: True                                   # 是否启用梯度检查点(激活重计算)，通过牺牲部分计算量节省显存占用
  recompute_modules:                                # 启用激活重计算的模型层结构
    - model.layers.{*}                              # 所有Transformer层启用重计算
  cp_size: 1                                        # 上下文并行(Context Parallel)大小，将输入序列的上下文拆分到多张卡
  cp_type: ulysses                                  # 上下文并行算法类型，目前仅支持ulysses算法
```

### 6.3 训练参数
```yaml
training:
  per_device_train_batch_size: 1                    # 每张卡的训练batch size大小
  gradient_accumulation_steps: 1                    # 梯度累积步数，将多个批次的梯度累积后再执行反向传播/参数更新
  dataloader_num_workers: 1                         # 数据加载子进程数，加速数据预处理
  disable_shuffling: 1                              # 是否禁用训练集洗牌
  seed: 42                                          # 训练开始时设置的随机种子，保证实验可复现性
  dataloader_drop_last: True                        # 当数据集大小不能被batch size整除时，是否丢弃最后一个不完整的批次
  output_dir: ./output                              # 训练结果输出目录，用于保存模型检查点、日志、预测结果等（必填项）
  optimizer: adamw                                  # 优化器类型，目前仅支持AdamW优化器
  lr: 1e-05                                         # AdamW优化器的初始学习率
  weight_decay: 0.01                                # AdamW优化器的权重衰减系数
  adam_beta1: 0.9                                   # AdamW优化器的beta1参数，控制一阶动量的指数衰减率
  adam_beta2: 0.95                                  # AdamW优化器的beta2参数，控制二阶动量的指数衰减率
  adam_epsilon: 1e-08                               # AdamW优化器的epsilon参数，用于数值稳定性
  max_grad_norm: 1.0                                # 梯度裁剪的最大范数，防止梯度爆炸
  lr_scheduler_type: cosine                         # 学习率调度器类型：cosine(余弦退火)/linear(线性衰减)/constant(恒定)
  warmup_ratio: 0.0                                 # 线性预热占总训练步数的比例
  min_lr: 1e-06                                     # cosine调度器的最小学习率，避免学习率过低导致训练停滞
  num_train_epochs: 3.0                             # 总训练轮数，若max_steps>0，该参数会被覆盖
  max_steps: -1                                     # 总训练步数，>0时覆盖num_train_epochs
  save_steps: 500                                   # 每500步保存一次模型检查点
  logging_steps: 1                                  # 每1步记录一次训练日志
```

### 6.4 微调场景数据集配置

```yaml
  dataset: alpaca_full                              # 训练数据集：可填写配置字典或逗号分隔的数据集名称
  template: gpt                                     # 构建 prompt 的模板名称
  cutoff_len: 2048                                  # 分词后输入序列的截断长度，超过该长度的序列会被截断
  max_samples: 100000                               # 调试用，用于截断每个数据集的样本数量，与 streaming 互斥
  overwrite_cache: True                             # 是否覆盖已缓存的预处理后数据集
  preprocessing_num_workers: 1                      # 数据预处理的进程数
```

其中dataset支持两种配置方式，推荐使用 **dataset_info.json 注册方式** 便于多数据集混合训练。

#### 方式一：内联配置（适用于单数据集快速验证）  

```yaml
data:
  dataset:
    file_name: "./my_data.json"                     # 数据文件路径
    formatting: "alpaca"                            # 数据格式模板，支持alpaca/sharegpt等格式
```

#### 方式二：通过 `dataset_info.json` 注册  

1. 编辑 `configs/fsdp2/data/dataset_info.json`，添加数据集条目：
```json
{
  "alpaca_full": {
    "file_name": "./train-00000-of-00001.parquet"
  },
  "sharegpt4_zh": {
    "file_name": "./sharegpt_zh.jsonl",
    "formatting": "sharegpt"
  }
}
```

2. 在 YAML 配置中引用：  

```yaml
data:
  dataset: alpaca_full, sharegpt4_zh                # 微调数据集：可填写逗号分隔的在 dataset_info.json 中配置的数据集名称，支持多数据集混合
```

### 6.5 预训练场景数据集配置  

当前预训练场景数据集配置方式和微调场景数据集配置方式有所不同，在此给出示例说明

```yaml
data:
  dataset: "your origin data path.example: /home/train-00000-of-a09b74b3ef9c3b56.parquet" #直接填写原始数据集路径
  template: gpt                                     # 构建 prompt 的模板名称
  cutoff_len: 4096                                  # 分词后输入序列的截断长度，超过该长度的序列会被截断
  max_samples: 100000                               # 调试用，用于截断每个数据集的样本数量，与 streaming 互斥
  overwrite_cache: True                             # 是否覆盖已缓存的预处理后数据集
  preprocessing_num_workers: 1                      # 数据预处理的进程数
  data_manager_type: mg                             # 数据管理器类型，lf 表示微调数据处理，mg 表示预训练
```

> 🔍 完整参数说明请参考：[FSDP参数说明](./arguments.md)

---
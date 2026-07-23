# 快速入门：Qwen3-8B 模型 FSDP2 后端训练

## 概述

本文档提供了一个简易示例，帮助初次接触 MindSpeed LLM 的开发者快速启动模型训练任务，并使用 FSDP2 后端完成大语言模型的预训练和微调任务。
以下将以 Qwen3-8B 模型为例，指导开发者完成大语言模型的预训练和微调任务，主要步骤包括：

- 环境准备：根据安装指引进行搭建环境
- 权重和数据集准备：从HuggingFace下载Qwen3-8B开源模型权重，并获取Alpaca数据集
- 启动训练任务：在昇腾 NPU 上使用 FSDP2 后端进行模型预训练和微调

> [!NOTE]
>
> MindSpeed LLM 支持 <term>Ascend 950 系列产品</term>、<term>Atlas A3 训练系列产品</term> 和 <term>Atlas A2 训练系列产品</term>，且要求单 NPU 的片上内存为 64GB 及以上，详见[模型支持列表](../models/supported_models.md)。
>
> 当前 Qwen3-8B 的示例脚本中 `NPUS_PER_NODE=8` 表示需要 8 个 NPU，如果实际情况低于此配置，可能遇到OOM（Out of Memory，内存不足）问题。

开发者入门基础：

- 具备基础的 PyTorch 使用经验
- 具备初级的 Python 开发经验
- 对FSDP（Fully Sharded Data Parallel，全分片数据并行）有基本了解

## 环境准备

请单击[MindSpeed快速安装](https://www.hiascend.com/developer/software/mindspeed)根据引导进行环境搭建，详细安装说明请参考[MindSpeed LLM安装指导](install_guide.md)。

## 权重和数据集准备

1. 获取开源模型权重

    创建一个目录存储权重文件。

    ```shell
    mkdir -p ./model_from_hf/qwen3_hf
    cd ./model_from_hf/qwen3_hf
    ```

    通过HuggingFace或ModelScope获取模型权重文件（择一获取）。

    方式一：通过HuggingFace获取

    ```shell
    # wget获取权重文件
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/generation_config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/merges.txt
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00001-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00002-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00003-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00004-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00005-of-00005.safetensors
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model.safetensors.index.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer_config.json
    wget https://huggingface.co/Qwen/Qwen3-8B/resolve/main/vocab.json
    ```

    方式二：通过ModelScope获取（国内推荐）

    ```shell
    # wget获取权重文件（从ModelScope下载）
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/config.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/generation_config.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/merges.txt
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00001-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00002-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00003-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00004-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00005-of-00005.safetensors
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model.safetensors.index.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/tokenizer.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/tokenizer_config.json
    wget https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/vocab.json
    ```

    利用sha256sum计算sha256数值校验权重文件正确性和完整性。

    ```shell
    # 打开文件明细可获取sha256值，https://huggingface.co/Qwen/Qwen3-8B/blob/main/model-00001-of-00005.safetensors 或者 https://www.modelscope.cn/models/Qwen/Qwen3-8B/file/view/master/model-00001-of-00005.safetensors
    sha256sum ./model-00001-of-00005.safetensors
    sha256sum ./model-00002-of-00005.safetensors
    sha256sum ./model-00003-of-00005.safetensors
    sha256sum ./model-00004-of-00005.safetensors
    sha256sum ./model-00005-of-00005.safetensors
    cd ../..
    ```

2. 获取数据集

    通过 HuggingFace 获取 Alpaca 数据集。

    ```shell
    mkdir dataset
    cd dataset/
    # HuggingFace 数据集链接（择一获取）
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    # ModelScope 数据集链接（择一获取）
    wget https://www.modelscope.cn/datasets/angelala00/tatsu-lab-alpaca/resolve/master/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

3. 设置环境变量

    ```shell
    source /usr/local/Ascend/cann/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh
    ```

    以上命令以 root 用户安装后的默认路径为例，请用户根据 set_env.sh 的实际路径进行替换。

## 启动预训练

在这一阶段，我们将修改预训练示例脚本和配置文件，启动模型预训练，具体步骤如下：

1. 编辑预训练启动脚本

    ```shell
    vi examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.sh
    ```

2. 修改并保存分布式参数配置

    配置示例如下：

    ```bash
    source examples/fsdp2/env_config.sh                 # 加载 NPU 环境变量配置

    NPUS_PER_NODE=8             # 使用单节点的8卡NPU
    MASTER_ADDR=localhost       # 单机使用本节点IP地址或者localhost，多机所有节点都配置为主节点IP地址
    MASTER_PORT=6499            # 本节点端口号为6499
    NNODES=1                    # 根据参与节点数量配置，单机为1，多机即多节点
    NODE_RANK=0                 # 单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复，NODE_RANK为0的节点为主节点
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))            # world size

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.yaml
    ```

3. 编辑训练参数配置文件

    ```shell
    vi examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.yaml
    ```

4. 修改并保存训练参数配置

    配置示例如下：

    ```yaml
    model:
      model_name_or_path: ./model_from_hf/qwen3_hf/     # 替换为下载的HuggingFace权重路径
      trust_remote_code: False
      train_from_scratch: False

    data:
      dataset:
        file_name: ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet   # 替换为下载的数据集路径
      template: qwen3
      cutoff_len: 4096
      max_samples: 100000
      overwrite_cache: True
      preprocessing_num_workers: 1
      data_manager_type: mg

    parallel:
      fsdp_size: 8                                       # FSDP 分片数量，必须等于NPUS_PER_NODE * NNODES（即world size）
      fsdp_modules:
        - model.layers.{*}
        - model.embed_tokens
        - lm_head
      tp_size: 1
      recompute: True
      recompute_modules:
        - model.layers.{*}

    training:
      stage: pt                                          # 训练阶段，pt 为预训练
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 1
      dataloader_num_workers: 4
      seed: 42
      dataloader_drop_last: True
      output_dir: ./output                               # 训练保存权重输出目录
      optimizer: adamw
      lr: 1e-05
      max_steps: 2000
      save_steps: 500
      logging_steps: 1
    ```

5. 执行预训练脚本

    ```shell
    bash examples/fsdp2/qwen3/pretrain_qwen3_8b_4k_fsdp2_A2.sh
    ```

    执行后一段时间可见终端显示。

    ```shell
    INFO [2026-06-22 19:28:40] >>  iteration        1/    2000 | consumed samples:          8 | consumed tokens:      32768 | elapsed time per iteration (ms): 10520.60 | learning rate: 9.999994E-06 | global batch size:     8 | lm loss: 1.304194E+00 | grad norm: 8.515 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 20.87 |
    INFO [2026-06-22 19:28:41] >>  iteration        2/    2000 | consumed samples:         16 | consumed tokens:      65536 | elapsed time per iteration (ms): 1879.90 | learning rate: 9.999978E-06 | global batch size:     8 | lm loss: 1.232217E+00 | grad norm: 4.346 | max_memory_allocated(GB): 19.65 | max_memory_reserved(GB): 25.59 |
    INFO [2026-06-22 19:28:43] >>  iteration        3/    2000 | consumed samples:         24 | consumed tokens:      98304 | elapsed time per iteration (ms): 1769.99 | learning rate: 9.999950E-06 | global batch size:     8 | lm loss: 1.134654E+00 | grad norm: 1.550 | max_memory_allocated(GB): 19.65 | max_memory_reserved(GB): 25.59 |
    ```

    进入迭代，说明训练正常进行。

> [!NOTE]
>
> - 多机训练需在多个终端同时启动预训练脚本（每个终端的预训练脚本只有 `NODE_RANK` 参数不同，MASTER_ADDR均为主节点的IP地址，其他参数均相同）。
> - FSDP2 后端会自动将模型参数分片到各 NPU，确保每个 NPU 只存储部分参数，从而支持超大规模模型训练。

## 启动微调

在这一阶段，我们将修改微调示例脚本和配置文件，启动模型微调，具体步骤如下：

1. 编辑微调启动脚本

    ```shell
    vi examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh
    ```

2. 修改并保存分布式参数配置

    配置示例如下：

    ```bash
    source examples/fsdp2/env_config.sh                 # 加载 NPU 环境变量配置

    NPUS_PER_NODE=8             # 使用单节点的8卡NPU
    MASTER_ADDR=localhost       # 单机使用本节点IP地址或者localhost，多机所有节点都配置为主节点IP地址
    MASTER_PORT=6499            # 本节点端口号为6499
    NNODES=1                    # 根据参与节点数量配置，单机为1，多机即多节点
    NODE_RANK=0                 # 单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复，NODE_RANK为0的节点为主节点
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))            # world size

    DISTRIBUTED_ARGS="
        --nproc_per_node $NPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT
    "

    torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml
    ```

3. 编辑微调参数配置文件

    ```shell
    vi examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.yaml
    ```

4. 修改并保存微调参数配置

    配置示例如下：

    ```yaml
    model:
      model_name_or_path: ./model_from_hf/qwen3_hf/     # 替换为下载的HuggingFace权重路径
      trust_remote_code: False
      train_from_scratch: False

    data:
      dataset:
          file_name: "/home/dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet"   # 替换为下载的数据集路径
          formatting: "alpaca"                                                       # 数据集数据格式，默认为alpaca
      template: qwen3
      cutoff_len: 4096
      max_samples: 100000
      overwrite_cache: True
      preprocessing_num_workers: 1

    parallel:
      fsdp_size: 8                                       # FSDP 分片数量，必须等于NPUS_PER_NODE * NNODES（即world size）
      fsdp_modules:
        - model.layers.{*}
        - model.embed_tokens
        - lm_head
      tp_size: 1
      recompute: True
      recompute_modules:
        - model.layers.{*}

    training:
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 1
      dataloader_num_workers: 4
      disable_shuffling: 1
      seed: 42
      dataloader_drop_last: True
      output_dir: ./output                               # 微调结果输出目录
      optimizer: adamw
      lr: 1e-05
      weight_decay: 0.01
      max_grad_norm: 1.0
      lr_scheduler_type: cosine
      warmup_ratio: 0.0
      min_lr: 1e-06
      max_steps: 2000
      save_steps: 500
      logging_steps: 1
    ```

5. 执行微调脚本

    ```shell
    bash examples/fsdp2/qwen3/tune_qwen3_8b_4k_fsdp2_A2.sh
    ```

    执行后一段时间可见终端显示。

    ```shell
    INFO [2026-06-22 19:25:37] >>  iteration        1/    2000 | consumed samples:          8 | consumed tokens:        564 | elapsed time per iteration (ms): 7827.39 | learning rate: 1.666667E-07 | global batch size:     8 | lm loss: 3.316887E+00 | grad norm: 70.959 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 20.90 |
    INFO [2026-06-22 19:25:38] >>  iteration        2/    2000 | consumed samples:         16 | consumed tokens:       1357 | elapsed time per iteration (ms): 1331.74 | learning rate: 3.333333E-07 | global batch size:     8 | lm loss: 2.443476E+00 | grad norm: 41.986 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 22.43 |
    INFO [2026-06-22 19:25:38] >>  iteration        3/    2000 | consumed samples:         24 | consumed tokens:       2113 | elapsed time per iteration (ms): 981.08 | learning rate: 5.000000E-07 | global batch size:     8 | lm loss: 2.669216E+00 | grad norm: 45.335 | max_memory_allocated(GB): 19.07 | max_memory_reserved(GB): 22.43 |
    ```

    进入迭代，说明训练正常进行。

> [!NOTE]
>
> - 多机微调需在多个终端同时启动微调脚本（每个终端的微调脚本只有 `NODE_RANK` 参数不同，MASTER_ADDR均为主节点的IP地址，其他参数均相同）。
> - 微调默认使用 alpaca 数据集格式，如需使用其他数据集，请参考[数据集配置说明](../features/fsdp2/arguments.md)。

脚本中包含训练参数，下表为部分参数解释。

**表 1** 训练脚本参数说明

|参数名|说明|配置示例|
|----|----|----|
|`fsdp_size`|全分片数据并行大小，必须等于 world size（即 `NPUS_PER_NODE * NNODES`）。|正整数，如 `8`、`16`|
|`fsdp_modules`|启用FSDP的模型层结构列表|`["model.layers.{*}", "model.embed_tokens", "lm_head"]`|
|`recompute`|是否启用重计算，通过牺牲部分计算量节省显存|`True` / `False`|
|`recompute_modules`|启用激活重计算的模型层结构|`["model.layers.{*}"]`|
|`data_manager_type`|数据管理器类型，mg表示预训练场景，微调时无需配置|预训练：`mg`，微调：不配置|
|`dataset`|数据集：使用数据集注册表中的名称，也可以使用本地数据集路径|`alpaca_full`、`sharegpt4_zh`|
|`template`|微调构建prompt的模板名称|`qwen3`、`gpt`|
|`cutoff_len`|分词后输入序列的截断长度，超过该长度的序列会被截断|`2048`、`4096`、`16384`|
|`trust_remote_code`|是否允许加载HuggingFace上自定义建模文件中的模型|`True` / `False`|
|`train_from_scratch`|是否使用随机权重从头开始训练模型，不加载模型权重|`True` / `False`|

> [!NOTE]
>
> 🔍 完整参数说明请参见[FSDP2 参数说明](../features/fsdp2/arguments.md)。

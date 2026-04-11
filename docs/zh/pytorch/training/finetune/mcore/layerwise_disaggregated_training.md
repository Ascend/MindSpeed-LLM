# 边云协同分布式安全训练

## 使用方法

由于边云协同分布式训练特性当前仅支持Qwen2.5/Qwen3系列模型，因此本文档以Qwen3-32B模型为例（PP=2，总隐藏层数64层）介绍使能方法，具体步骤如下：

1. 参考[MindSpeed LLM安装指导](../../install_guide.md)，完成环境安装。

    请在训练开始前配置好昇腾NPU套件相关的环境变量，如下所示：

    ```shell
    source /usr/local/Ascend/cann/set_env.sh     # 修改为实际安装的Toolkit包路径
    source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径
    ```

2. 准备好模型权重和微调数据集。

    完整的Qwen3-32B模型文件夹应该包括以下内容：

    ```shell
    .
    ├── README.md                    # 模型说明文档
    ├── config.json                  # 模型结构配置文件
    ├── generation_config.json       # 文本生成时的配置
    ├── merges.txt                   # tokenizer的合并规则文件
    ├── model-00001-of-00017.safetensors  # 模型权重文件第1部分（共17部分）
    ├── model-00002-of-00017.safetensors  # 模型权重文件第2部分
    ├── ...
    ├── model-00016-of-00017.safetensors  # 模型权重文件第16部分
    ├── model-00017-of-00017.safetensors  # 模型权重文件第17部分
    ├── model.safetensors.index.json      # 权重分片索引文件，指示各个权重参数对应的文件
    ├── tokenizer.json              # Hugging Face格式的tokenizer
    ├── tokenizer_config.json       # tokenizer相关配置
    └── vocab.json                  # 模型词表文件
    ```

3. 进行权重转换，将HF权重转换为Megatron-Mcore格式。

    边云协同分布式训练采用U-shape切分，模型首尾层权重需要分别存储。详细配置请参考[Qwen3权重转换脚本](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh)。

    以Qwen3-32B模型在TP8PP2切分为例，需要修改相关路径参数和模型切分配置：

    ```shell
    --target-tensor-parallel-size 8          # TP切分大小
    --target-pipeline-parallel-size 2        # PP切分大小
    --num-layer-list 16,32,16               # U-shape切分：首层16层、隐藏层32层、尾层16层
    --load-dir ./model_from_hf/qwen3_hf/     # 原始HF模型权重路径
    --save-dir ./model_weights/qwen3_mcore/  # Megatron权重保存路径
    ```

    参数说明：

    - `--num-layer-list`：配置非均匀PP切分，传参为各级流水的隐藏层数`L0,...,LPP`，其中L0和LPP表示首尾隐藏层数。以PP=2为例，传参`16,32,16`表示首层16层、中间层32层、尾层16层。

    确认路径无误后运行权重转换脚本：

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

4. 将Megatron-Mcore格式模型转换为VPP格式。

    边云协同分布式训练需要将首尾层权重合并为VPP格式，调用权重转换脚本`convert_ckpt_pp_vpp.py`进行操作：

    ```shell
    python mindspeed_llm/tasks/layerwise_disaggregated_training/convert_ckpt_pp_vpp.py merge \
        --load-dir ./model_weights/qwen3_mcore/ \
        --save-dir-edge ./model_weights/qwen3_vpp_edge/ \
        --save-dir-cloud ./model_weights/qwen3_vpp_cloud/ \
        --merge-stages 0,1 \
        --middle-stages 1
    ```

    各参数解析如下：

    | 参数              | 说明                                       | 必填 |
    | ----------------- | ------------------------------------------ | ---- |
    | `--load-dir`      | Megatron-Mcore格式权重文件加载路径           | 是   |
    | `--save-dir-edge` | 边侧权重文件保存路径                         | 是   |
    | `--save-dir-cloud`| 云侧权重文件保存路径                         | 是   |
    | `--merge-stages`  | 首尾层的PP stage索引，格式为`0,PP`          | 是   |
    | `--middle-stages` | 中间层的PP stage索引，格式为`1,...,PP-1`    | 是   |

5. 进行数据预处理。

    以Alpaca数据集为例执行数据预处理，详细配置请参考[Qwen3数据预处理脚本](../../../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh)：

    ```shell
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet  # 原始数据集路径
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf                # HF的tokenizer路径
    --output-prefix ./finetune_dataset/alpaca                        # 保存路径
    ```

    相关参数设置完毕后，运行数据预处理脚本：

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

6. 启动微调训练。

    配置模型微调脚本，详细配置请参考[Qwen3-32b微调脚本](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh)，需要修改相关路径参数和模型切分配置：

    ```shell
    CKPT_LOAD_DIR="./model_weights/qwen3_vpp_edge/"  # 边侧权重加载路径
    CKPT_LOAD_CLOUD_DIR="./model_weights/qwen3_vpp_cloud/"  # 云侧权重加载路径
    CKPT_SAVE_DIR="./ckpt/qwen3_finetune/"          # 微调完成后的权重保存路径
    DATA_PATH="./finetune_dataset/alpaca"           # 数据集路径
    TOKENIZER_PATH="./model_from_hf/qwen3_hf"       # 词表路径
    TP=8                                             # TP切分大小
    PP=2                                             # PP切分大小
    ```

    在训练脚本中增加以下参数开启边云协同分布式训练特性：

    ```shell
    --layerwise-disaggregated-training              # 开启边云协同分布式安全训练
    --num-layer-list 16,32,16                       # 非均匀PP切分，与权重转换时保持一致
    --num-virtual-stages-per-pipeline-rank 2         # 虚拟Pipeline Stage数，必须配置为2
    ```

    相关参数设置完毕后，运行微调脚本：

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

## 使用约束

### 模型范围

- 支持以下Qwen2.5/Qwen3系列的LLM模型：

    | 模型类型 | 具体模型                      |
    | -------- | ----------------------------- |
    | LLM      | qwen3-32B, qwen2.5-32B, qwen2.5-72B |

- 暂不支持MoE模型。

### 其他约束

- 暂不支持LoRA
- 暂不支持常规VPP并行：`--num-virtual-stages-per-pipeline-rank`传参必须为`2`，使能首尾层共部署。

## 注意事项

- 训练参数的并行配置（如TP/PP）需要与权重转换时的配置保持一致。
- 边云协同分布式训练采用U-shape切分方案，模型首尾层同时部署在边侧，原始样本无需上传云端。
- 跨域协同训练通过流水编排优化和计算通信掩盖，实现边云跨域连接场景的高效训练。

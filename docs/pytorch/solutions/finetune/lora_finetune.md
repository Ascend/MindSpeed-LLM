# LoRA 微调

## 使用场景

LoRA（Low-Rank Adaptation）是一种高效的模型微调方法，广泛应用于预训练的深度学习模型。通过在权重上添加低秩矩阵，LoRA 使得微调过程更为轻量，节省计算资源和存储空间。LoRA 的核心思想是将模型的参数更新分解为低秩的形式。具体步骤如下：

- **分解权重更新**：在传统的微调方法中，直接对模型的权重进行更新。而 LoRA 通过在每一层的权重矩阵中引入两个低秩矩阵 $A$ 和 $B$ 进行替代。即：
$
W' = W + A \cdot B
$

![alt text](../../../../sources/images/lora_finetune/lora_model.png)

   其中，$W'$ 是更新后的权重，$W$ 是原始权重，$A$ 和 $B$ 是需要学习的低秩矩阵。

- **降低参数量**：由于 $A$ 和 $B$ 的秩较低，所需的参数量显著减少，节省了存储和计算成本。

## 使用方法

基于预训练语言模型，当前文档提供了一个简单的单样本格式数据LoRA微调任务示例。下面以Qwen3-8B模型和单台`Atlas 900 A2 POD`（1x8集群）进行LoRA微调。大模型LoRA微调主要包含以下流程：

![alt text](../../../../sources/images/lora_finetune/process_of_lora_tuning.png)

第一步，请参考[安装指导](../../install_guide.md)，完成环境安装。请注意由于Qwen3要求使用`transformers>=4.51.0`，因此Python需使用3.9及以上版本。请在训练开始前配置好昇腾NPU套件相关的环境变量，如下所示：

```shell
source /usr/local/Ascend/cann/set_env.sh     # 修改为实际安装的Toolkit包路径
source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径
```

第二步，准备好模型权重和微调数据集。模型权重下载请参考[Dense模型](../../models/dense_model.md)、[MOE模型](../../models/moe_model.md)和[SSM模型](../../models/ssm_model.md)文档中对应模型的下载链接。以[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/tree/main)模型为例，完整的模型文件夹应该包括以下内容：

```shell
.
├── README.md                    # 模型说明文档
├── config.json                  # 模型结构配置文件
├── generation_config.json       # 文本生成时的配置
├── merges.txt                   # tokenizer的合并规则文件
├── model-00001-of-00005.safetensors  # 模型权重文件第1部分（共5部分）
├── model-00002-of-00005.safetensors  # 模型权重文件第2部分
├── model-00003-of-00005.safetensors  # 模型权重文件第3部分
├── model-00004-of-00005.safetensors  # 模型权重文件第4部分
├── model-00005-of-00005.safetensors  # 模型权重文件第5部分
├── model.safetensors.index.json      # 权重分片索引文件，指示各个权重参数对应的文件
├── tokenizer.json              # Hugging Face格式的tokenizer
├── tokenizer_config.json       # tokenizer相关配置
└── vocab.json                  # 模型词表文件
```

第三步，进行权重转换，即将模型原始的HF权重转换成Megatron权重。LoRA微调脚本可使用普通base Megatron权重进行微调任务，以Qwen3-8B模型在TP1PP2切分为例，详细配置请参考[Qwen3权重转换脚本](../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh)。需要修改相关路径参数和模型切分配置：

```shell
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
......
--target-tensor-parallel-size 1          # TP切分大小
--target-pipeline-parallel-size 2        # PP切分大小
--load-dir ./model_from_hf/qwen3_hf/     # HF权重路径
--save-dir ./model_weights/qwen3_mcore/  # Megatron权重保存路径
```

确认路径无误后运行权重转换脚本：

```shell
bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

第四步，进行数据预处理。接下来将以[Alpaca数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)为例执行数据预处理，详细配置请参考[Qwen3数据预处理脚本](../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh)。需要修改脚本内的路径：

```shell
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径
......
--input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # 原始数据集路径 
--tokenizer-name-or-path ./model_from_hf/qwen3_hf # HF的tokenizer路径
--output-prefix ./finetune_dataset/alpaca  # 保存路径
......
```

数据预处理相关参数说明：

- `handler-name`：指定数据集的处理类，常用的有`AlpacaStyleInstructionHandler`，`SharegptStyleInstructionHandler`，`AlpacaStylePairwiseHandler`等。
- `tokenizer-type`：指定处理数据的tokenizer，常用`PretrainedFromHF`。
- `workers`：处理数据集的并行数。
- `log-interval`：处理进度更新的间隔步数。
- `enable-thinking`：快慢思考模板开关，可设定为`[true,false,none]`，默认值是`none`。开启后，会在数据集的模型回复中添加`<think>`和`</think>`，并参与到loss计算，所有数据被当成慢思考数据；当关闭后，空的CoT标志将被添加到数据集的用户输入中，不参与loss计算，所有数据被当成快思考数据；设置为`none`时适合原始数据集是混合快慢思考数据的场景。**目前只支持Qwen3系列模型**。
- `prompt-type`：用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../../../configs/finetune/templates.json)文件内查看。

相关参数设置完毕后，运行数据预处理脚本：

```shell
bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
```

第五步，配置模型LoRA微调脚本，详细的参数配置请参考[Qwen3-8b LoRA微调脚本](../../../../examples/mcore/qwen3/tune_qwen3_8b_4K_lora_ptd.sh)。脚本中的环境变量配置见[环境变量说明](../../features/environment_variable.md)。注意：训练参数的并行配置，如TP/PP等需要与第三步权重转换时的配置保持一致。

模型LoRA微调可在单机或者多机上运行，以下是单机运行的相关参数配置说明：

```shell
# 单机配置
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1  
NODE_RANK=0  
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

环境变量确认无误后，需要修改相关路径参数和模型切分配置：

```shell
CKPT_LOAD_DIR="your model ckpt path"      # 权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="your model save ckpt path" # LoRA微调完成后的权重保存路径
DATA_PATH="your data path"                # 数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="your tokenizer path"      # 词表路径，填入下载的开源权重词表路径
TP=1                                      # 权重转换时target-tensor-parallel-size的值
PP=2                                      # 权重转换时target-pipeline-parallel-size的值
```

微调脚本相关参数说明：
- `DATA_PATH`：数据集路径。请注意实际数据预处理生成文件末尾会增加`_input_ids_document`等后缀，该参数填写到数据集的前缀即可。例如实际的数据集相对路径是`./finetune_dataset/alpaca/alpaca_packed_input_ids_document.bin`等，那么只需要填`./finetune_dataset/alpaca/alpaca`即可。
- `is-instruction-dataset`：用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。
- `prompt-type`：指定模型模板，使 base 模型在微调后具备更好的对话能力。
- `no-pad-to-seq-lengths`： 支持动态序列长度微调，默认按 8 的倍数进行 padding，可以通过 `--pad-to-multiple-of` 参数修改 padding 的倍数。
- `lora-r：LoRA rank`，表示低秩矩阵的维度。较低的 rank 值模型在训练时会使用更少的参数更新，从而减少计算量和内存消耗。然而，过低的 rank 可能限制模型的表达能力。
- `lora-alpha`：控制 LoRA 权重对原始权重的影响比例, 数值越高则影响越大。一般保持 `α/r` 为 2。
- `lora-fusion`： 是否启用[CCLoRA](../../features/cc_lora.md)算法，该算法通过计算通信掩盖提高性能。
- `lora-target-modules`：选择需要添加 LoRA 的模块。当前可选模块： `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`

第六步，启动LoRA微调脚本。参数配置完毕后，如果是单机运行场景，只需要在一台机器上启动LoRA微调脚本：

```shell
bash examples/mcore/qwen3/tune_qwen3_8b_4K_lora_ptd.sh
```

如果是多机运行，则需要在单机的脚本上修改以下参数：

```shell
# 多机配置 
# 根据分布式集群实际情况配置分布式参数
NPUS_PER_NODE=8  # 每个节点的卡数
MASTER_ADDR="your master node IP"  # 都需要修改为主节点的IP地址（不能为localhost）
MASTER_PORT=6000
NNODES=2  # 集群里的节点数，以实际情况填写
NODE_RANK="current node id"  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
```

最后确保每台机器上的模型路径和数据集路径等无误后，在多个终端上同时启动LoRA微调脚本即可开始训练。

第七步，进行模型验证。完成LoRA微调后，需要进一步验证模型是否具备了预期的输出能力。仓库提供了基础的推理脚本[Qwen3-8B推理脚本](../../../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh)，LoRA推理需要在该脚本的基础上增加LoRA相关参数，便可观察模型在不同生成参数配置下的回复。以Qwen3-8B为例，对应的LoRA推理脚本可命名为`generate_qwen3_8b_lora_ptd.sh`。

在推理脚本基础上修改路径参数并增加LoRA相关参数：

```shell
TOKENIZER_PATH="your tokenizer directory path"   # 词表路径，填入下载的开源权重词表路径
CHECKPOINT="your model directory path"           # 权重加载路径，填入权重转换时保存的权重路径
CHECKPOINT_LORA="your lora model directory path" # LoRA微调完成后的权重保存路径
......
--lora-load ${CHECKPOINT_LORA}  \
--lora-r 16 \
--lora-alpha 32 \
--lora-fusion \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
--prompt-type qwen3 \
......
| tee logs/generate_qwen3_8b_lora_ptd.log # 对应日志文件名称
```

参数说明：
- `lora-load`：加载 LoRA 权重断点继续训练或用于推理。在推理时需与 `--load` 参数配合使用，加载 `CKPT_SAVE_DIR` 路径下的 LoRA 权重。

相关参数设置完毕后，运行推理脚本：

```shell
bash examples/mcore/qwen3/generate_qwen3_8b_lora_ptd.sh
```

预期模型能够正常回答数据集中的问题，回答无乱码或重复。
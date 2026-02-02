# LoRA 微调迁移开发

MindSpeed LLM支持在微调任务上使用LoRA进行低参数训练，使用方法为在基准任务上加上LoRA参数进行使能。

本教程旨在为用户提供模型LoRA微调迁移开发指导。接下来以Qwen3-8B模型和单台`Atlas 900 A2 POD`（1x8集群）为例，逐步说明如何进行LoRA微调脚本开发。

在按照如下步骤操作前，需要先参考[安装指导](../install_guide.md)完成环境安装，并准备好模型权重和微调数据集。模型权重下载请参考[Dense模型](../models/dense_model.md)、[MOE模型](../models/moe_model.md)和[SSM模型](../models/ssm_model.md)文档中对应模型的下载链接。数据集下载请参考[Alpaca风格数据集](../solutions/finetune/datasets/alpaca_dataset.md)、[ShareGPT风格数据集](../solutions/finetune/datasets/sharegpt_dataset.md)和[Pairwise风格数据集](../solutions/finetune/datasets/pairwise_dataset.md)文档中数据集下载章节的内容。

## 1、模型权重转换

LoRA微调脚本可使用普通base Megatron权重进行微调任务，以Qwen3-8B模型在TP1PP2切分为例，详细配置请参考[Qwen3权重转换脚本](../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh)。需要修改相关路径参数和模型切分配置：

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

## 2、数据预处理

以Alpaca数据集为例执行数据预处理，详细配置请参考[Qwen3数据预处理脚本](../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh)。需要修改脚本内的路径：

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
- `prompt-type`：用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../../configs/finetune/templates.json)文件内查看。

相关参数设置完毕后，运行数据预处理脚本：

```shell
bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
```

## 3、LoRA微调脚本开发

仓库提供了基础的微调脚本[Qwen3-8B微调脚本](../../../examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh)，LoRA微调需要在该脚本的基础上增加LoRA相关参数。以Qwen3-8B为例，对应的LoRA微调脚本可命名为`tune_qwen3_8b_4K_lora_ptd.sh`。

并行配置参数如TP/PP等需要与权重转换时的配置保持一致。需要修改相关路径参数和模型切分配置：

```shell
CKPT_LOAD_DIR="your model ckpt path"      # 权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="your model save ckpt path" # LoRA微调完成后的权重保存路径
DATA_PATH="your data path"                # 数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="your tokenizer path"      # 词表路径，填入下载的开源权重词表路径
TP=1                                      # 权重转换时target-tensor-parallel-size的值
PP=2                                      # 权重转换时target-pipeline-parallel-size的值
```

LoRA微调需要在全参微调脚本基础上，在TUNE_ARGS中增加LoRA参数：

```shell
--lora-r 16 \
--lora-alpha 32 \
--lora-fusion \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
......
| tee logs/tune_qwen3_8b_4K_lora_ptd.log # 对应日志文件名称
```

微调脚本相关参数说明：
- `lora-r：LoRA rank`，表示低秩矩阵的维度。较低的 rank 值模型在训练时会使用更少的参数更新，从而减少计算量和内存消耗。然而，过低的 rank 可能限制模型的表达能力。
- `lora-alpha`：控制 LoRA 权重对原始权重的影响比例, 数值越高则影响越大。一般保持 `α/r` 为 2。
- `lora-fusion`： 是否启用[CCLoRA](../features/cc_lora.md)算法，该算法通过计算通信掩盖提高性能。
- `lora-target-modules`：选择需要添加 LoRA 的模块。当前可选模块： `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`

上述参数配置完成后运行LoRA微调脚本：

```shell
bash examples/mcore/qwen3/tune_qwen3_8b_4K_lora_ptd.sh
```

## 4、LoRA推理脚本开发

完成LoRA微调后，需要进一步验证模型是否具备了预期的输出能力。仓库提供了基础的推理脚本[Qwen3-8B推理脚本](../../../examples/mcore/qwen3/generate_qwen3_8b_ptd.sh)，LoRA推理需要在该脚本的基础上增加LoRA相关参数。以Qwen3-8B为例，对应的LoRA推理脚本可命名为`generate_qwen3_8b_lora_ptd.sh`。

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

并行配置参数如TP/PP等需要与权重转换时的配置保持一致。相关参数设置完毕后，运行LoRA推理脚本：

```shell
bash examples/mcore/qwen3/generate_qwen3_8b_lora_ptd.sh
```

预期模型能够正常回答数据集中的问题，回答无乱码或重复。

## 5、LoRA微调权重评估脚本开发

MindSpeed LLM支持大模型在公开基准数据集上进行准确率评估，详细统计信息见[evaluation.md](../models/models_evaluation.md)。仓库提供了基础的权重评估脚本[Qwen3-8B权重评估脚本](../../../examples/mcore/qwen3/evaluate_qwen3_8b_ptd.sh)，LoRA微调权重评估需要在该脚本的基础上增加LoRA相关参数。以Qwen3-8B为例，对应的LoRA微调权重评估脚本可命名为`evaluate_qwen3_8b_lora_ptd.sh`。

在权重评估脚本基础上修改路径参数并增加LoRA相关参数：

```shell
TOKENIZER_PATH="your tokenizer directory path"   # 词表路径，填入下载的开源权重词表路径
CHECKPOINT="your model directory path"           # 权重加载路径，填入权重转换时保存的权重路径
CHECKPOINT_LORA="your lora model directory path" # LoRA微调完成后的权重保存路径
DATA_PATH="your data path"                       # 下载的Benchmark数据集路径，通常使用MMLU
......
--lora-load ${CHECKPOINT_LORA} \
--lora-r 16 \
--lora-alpha 32 \
--lora-fusion \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
......
| tee logs/evaluate_qwen3_8b_lora_ptd.log # 对应日志文件名称
```

并行配置参数如TP/PP等需要与权重转换时的配置保持一致。相关参数设置完毕后，运行LoRA微调权重评估脚本：

```shell
bash examples/mcore/qwen3/evaluate_qwen3_8b_lora_ptd.sh
```

注意：评估效果由训练数据集和训练方法决定，需要用户自行选择合适的数据集或调整LoRA微调脚本参数。

## 6、LoRA权重与Base权重的合并与转换

当前仓库支持将LoRA微调权重与基础模型权重合并，转换为Huggingface格式。在权重转换脚本中添加以下LoRA参数即可：

```shell
--lora-load ${CHECKPOINT_LORA}  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
```

以Qwen3-8B为例，若希望将LoRA权重合并后转换为Hugging Face格式，对应的权重转换脚本可命名为`ckpt_convert_qwen3_mcore2hf_lora.sh`。

```shell
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径

python convert_ckpt.py
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \          # 转为HF权重时切分需要设置为1
    --target-pipeline-parallel-size 1 \        # 转为HF权重时切分需要设置为1
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_weights/qwen3_mcore/ \  # Megatron权重保存路径
    --lora-load ./ckpt/qwen3_lora \            # LoRA微调完成后的权重保存路径
    --save-dir ./model_from_hf/qwen3_hf/       # 转换后的HF权重路径，若填写原始HF模型路径，新权重将存储于./model_from_hf/qwen3_hf/mg2hg/
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --model-type-hf qwen3
```

路径参数设置完毕后，运行权重转换脚本：

```shell
bash examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf_lora.sh
```

注意：
- LoRA参数的值应与微调时的参数设置保持一致，以确保转换后的模型具有相同的性能表现和兼容性。
- 当前权重转换V2版本不支持LoRA参数，且V1和V2不支持混用，若要使用LoRA权重与Base权重的合并与转换功能，第1步的模型权重转换也需要使用V1版本。
- 由于调用peft库合并LoRA权重后，权重数据类型为float16，但是部分模型如qwen系列模型，默认数据类型为bfloat16，合并后的权重转回HF格式会有精度损失问题。可以将原始HF模型的config.json中的数据类型改为float16暂时规避。

Qwen3-8B对应的V1权重转换脚本示例：

```shell
source /usr/local/Ascend/cann/set_env.sh # 修改为实际安装的Toolkit包路径

python convert_ckpt.py
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir ./model_from_hf/qwen3_hf/ \                       # HF权重路径
    --save-dir ./model_weights/qwen3_mcore/ \                    # Megatron权重保存路径
    --tokenizer-model ./model_from_hf/qwen3_hf/tokenizer.json \  # HF权重分词器模型文件路径
    --model-type-hf qwen3 \
    --params-dtype bf16
```

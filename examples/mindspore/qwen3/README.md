# 1 环境配置

MindSpeed-LLM MindSpore后端的安装步骤参考：[基础安装指导](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/mindspore/features/install_guide.md)

# 2 权重转换

## 2.1 权重下载

从[huggingface(以Qwen3-0.6B为例)](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main)下载模型权重和其它配置文件，若需要在开源权重上继续预训练、微调、推理，也请下载网络模型文件。

## 2.2 权重转换
提供脚本将huggingface开源权重转换为mcore权重，用于训练、推理、评估等任务。使用方法如下，请根据实际需要的TP/PP等切分策略和权重路径修改权重转换脚本：

```
cd MindSpeed-LLM
bash examples/mindspore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

运行脚本后，预期会看到类似以下的日志输出，表示权重转换成功：

```
successfully saved checkpoint from iteration 1 to ./model_weights/qwen3_mcore/
INFO:root:Done!
```

* 注：MindSpore 后端转换出的模型权重无法用于 Torch后端训练或推理。

# 3 数据预处理

当前MindSpore后端已完全支持MindSpeed-LLM的多种任务场景下的数据预处理，数据预处理指南参见[数据预处理](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/pytorch/solutions/pretrain/pretrain_dataset.md)。

## 3.1 预训练数据处理

（以alpaca数据集为例）修改`data_convert_qwen3_pretrain.sh`预训练脚本

配置好数据输入/输出路径、tokenizer模型路径即可启动：

```
bash examples/mindspore/qwen3/data_convert_qwen3_pretrain.sh
```

预训练数据集处理结果如下：

```
./dataset/alpaca_text_document.bin
./dataset/alpaca_text_document.idx
```

## 3.2 微调数据处理

（以alpaca数据集为例）修改`data_convert_qwen3_instruction.sh`微调脚本

配置好数据输入/输出路径、tokenizer模型路径即可启动：

```
bash examples/mindspore/qwen3/data_convert_qwen3_instruction.sh
```

微调数据集处理结果如下：

```
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```

# 4 训练

## 4.1 预训练

在`pretrain_qwen3_0point6b_4K_ms.sh`脚本中修改相关参数，并执行脚本

```
cd MindSpeed-LLM
bash examples/mindspore/qwen3/pretrain_qwen3_0point6b_4K_ms.sh
```

| 变量名 | 含义 |
| --- | --- |
| MASTER_ADDR | 多机情况下主节点IP |
| NODE_RANK | 多机下，各机对应节点序号 |
| DATA_PATH | 预处理后的数据路径 |
| TOKENIZER_PATH | Qwen3 0.6b tokenizer目录 |
| CKPT_LOAD_DIR | 初始权重加载，如无初始权重则随机初始化 |
| CKPT_SAVE_DIR | 训练中权重保存位置 |
| TRAIN_ITERS | 训练迭代步数 |

* 注：0.6b模型规模较小，一般单机即可

## 4.2 微调

在`tune_qwen3_0point6b_4K_full_ms.sh`脚本中修改相关参数，并执行脚本

```
cd MindSpeed-LLM
bash examples/mindspore/qwen3/tune_qwen3_0point6b_4K_full_ms.sh
```

| 变量名 | 含义 |
| --- | --- |
| MASTER_ADDR | 多机情况下主节点IP |
| NODE_RANK | 多机下，各机对应节点序号 |
| DATA_PATH | 预处理后的数据路径 |
| TOKENIZER_PATH | Qwen3 0.6b tokenizer目录 |
| CKPT_LOAD_DIR | 初始权重加载，如无初始权重则随机初始化 |
| CKPT_SAVE_DIR | 训练后的权重保存位置 |
| TRAIN_ITERS | 训练迭代步数 |

* 注1：0.6b模型规模较小，一般单机即可
* 注2：`CKPT_LOAD_DIR`选择加载预训练保存后的权重

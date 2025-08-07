# 1 环境配置

MindSpeed-LLM MindSpore后端的安装步骤参考：[基础安装指导](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/mindspore/features/install_guide.md)

# 2 Huggingface模型仓下载

从[huggingface(以Qwen3-0.6B为例)](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main)下载除模型权重外的其他配置文件，目录结构保持`huggingface`原始目录结构

# 3 数据预处理

当前MindSpore后端已完全支持MindSpeed-LLM的多种任务场景下的数据预处理，数据预处理指南参见[数据预处理](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/pytorch/solutions/pretrain/pretrain_dataset.md)。

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

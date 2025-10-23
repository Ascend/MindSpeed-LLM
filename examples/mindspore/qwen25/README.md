## Mindspore后端提供Qwen25系列模型支持
<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
      <th>是否支持</th>
    </tr>
  </thead>
  <tbody>
      <tr>
      <td rowspan="7"><a href="https://huggingface.co/Qwen">Qwen25</a></td>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-7B/tree/main">7B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>✅</td>
      <tr>
      <td> 32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>✅</td>
      <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-32B/tree/main">32B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>✅</td>
      <tr>
      <td> 32K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>✅</td>
      <tr>
    </tr>
  </tbody>
</table>

## MindSpore后端跑通Qwen25模型手把手教程


### 环境配置

MindSpeed-LLM MindSpore后端的安装步骤参考[基础安装指导](../../../docs/mindspore/features/install_guide.md)。



### 权重转换

1. 权重下载  
   以7B模型为例，从[huggingface](https://huggingface.co/Qwen/Qwen2.5-7B/tree/main)下载模型权重和其它配置文件，若需要在开源权重上继续预训练、微调、推理，也请下载网络模型文件。

2. 权重转换  
   提供脚本将huggingface开源权重转换为mcore权重，用于训练、推理、评估等任务。使用方法如下，请根据实际需要的TP/PP等切分策略和权重路径修改权重转换脚本：
    ```sh
    cd MindSpeed-LLM
    bash examples/mindspore/qwen25/ckpt_convert_qwen25_hf2mcore.sh
    ```
   运行脚本后，预期会看到类似以下的日志输出，表示权重转换成功：
    ```log
   successfully saved checkpoint from iteration 1 to ./model_weights/qwen2.5_mcore/
   INFO:root:Done!
    ```
**注意：**
- MindSpore 后端默认在Device侧进行权重转换，在模型较大时存在OOM风险，因此建议用户手动修改`convert_ckpt.py`，在包导入时加入如下代码设置CPU侧执行权重转换：

```python
import mindspore as ms
ms.set_context(device_target="CPU", pynative_synchronize=True)
import torch
torch.configs.set_pyboost(False)
```

- MindSpore 后端转换出的模型权重无法用于 Torch后端训练或推理。

### 数据预处理

当前MindSpore后端，已完全支持MindSpeed-LLM的多种任务场景下的数据预处理

#### 预训练
以Alpaca数据集为例，在进行[数据预处理](../../../docs/pytorch/solutions/pretrain/pretrain_dataset.md)时，只需在预训练数据预处理脚本`data_convert_qwen25_pretrain.sh`中配置好数据输入/输出路径、tokenizer模型路径，并启动即可：
```sh
bash examples/mindspore/qwen25/data_convert_qwen25_pretrain.sh
```
预训练数据集处理结果如下：
```log
./dataset/alpaca_text_document.bin
./dataset/alpaca_text_document.idx
```
预训练时，数据集路径 --data-path 参数传入 ./dataset/alpaca_text_document 即可

#### 微调
以[Alpaca风格微调数据集处理](../../../docs/pytorch/solutions/finetune/datasets/alpaca_dataset.md)为例，只需在预训练数据预处理脚本`data_convert_qwen25_instruction.sh`中配置好数据输入/输出路径、tokenizer模型路径，并启动即可：
```sh
bash examples/mindspore/qwen25/data_convert_qwen25_instruction.sh
```
微调数据集处理结果如下：
```log
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```
微调时，数据集路径输入 ./finetune_dataset/alpaca 即可

### 训练

#### 预训练

预训练使用方法如下
```sh
# 以7b模型为例
cd MindSpeed-LLM
bash examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh
```
用户需要根据实际情况修改脚本中的以下变量

  |变量名  | 含义                                |
  |--------|-----------------------------------|
  | MASTER_ADDR | 多机情况下主节点IP                        |
  | NODE_RANK | 多机下，各机对应节点序号                      |
  | CKPT_SAVE_DIR | 训练中权重保存路径                         |
  | DATA_PATH | 数据预处理后的数据路径                       |
  | TOKENIZER_PATH | qwen25 tokenizer目录                |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，用于初始权重加载，如无初始权重则随机初始化 |

#### 微调
微调和预训练的使用方法类似
```sh
# 以全参微调7b模型为例
cd MindSpeed-LLM
bash examples/mindspore/qwen25/tune_qwen25_7b_4k_full_ms.sh
```
与预训练一样，用户需要根据实际情况修改脚本中的上述变量。

### 推理
推理使用方法如下
```sh
# 以7b模型为例
cd MindSpeed-LLM
bash examples/mindspore/qwen25/generate_qwen25_7b_ms.sh
```
用户需要根据实际情况修改脚本中以下变量

  | 变量名  | 含义                 |
  |--------|--------------------|
  | MASTER_ADDR | 多机情况下主节点IP         |
  | NODE_RANK | 多机下，各机对应节点序号       |
  | CHECKPOINT | 训练保存的权重路径          |
  | TOKENIZER_PATH | qwen25 tokenizer目录 |

### 评估
评估使用方法如下
```sh
# 以7b模型为例
cd MindSpeed-LLM
bash examples/mindspore/qwen25/evaluate_qwen25_7b_ms.sh
```

用户需要根据实际情况修改脚本中以下变量。关于数据集，可参考[评估数据集](../../../docs/pytorch/solutions/evaluation/evaluation_datasets/mmlu_evaluation.md)

  | 变量名  | 含义                    |
  |--------|-----------------------|
  | MASTER_ADDR | 多机情况下主节点IP            |
  | NODE_RANK | 多机下，各机对应节点序号          |
  | TOKENIZER_PATH | qwen25 tokenizer目录    |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，或训练保存的权重路径         |
  |  DATA_PATH | 评估采用的数据集路径，当前推荐使用MMLU |
  | TASK  | 评估采用的数据集，当前推荐使用MMLU   | 

使用MMLU的前3个子集，进行评估的结果如下：
```log
INFO:mindspeed_llm.tasks.evaluation.eval_impl.mmlu_eval:mmlu acc = 321/387=0.8294573643410853
total: 100%|█████████████████████████████████████████████████████| 3/3 [06:16<00:00, 128.12s/it]INFO:main:
             subject   question_n   acc
0   abstract_algebra          100   0.720000
1          astronomy          152   0.927632
2            anatomy          135   0.800000
3              total          387   0.829457
INFO:main:MMLU Running Time:, 376.0990614891052
```

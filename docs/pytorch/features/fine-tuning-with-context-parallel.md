# 长序列微调

## 使用方法

### 数据预处理
数据预处理方法同[**多样本pack微调**](../solutions/finetune/multi_sample_pack_finetune.md)。

### 微调参数

【--is-instruction-dataset】

用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。

【--prompt-type】

用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../../configs/finetune/templates.json)文件内查看。

【--reset-position-ids】

每条数据由不同的样本拼接而成，因此其位置 ID 并不连续。该参数用于为每条拼接的数据重置位置 ID，以确保在处理多个样本时，位置编码保持一致性。

【--context-parallel-size】

设置CP切分的并行数目，配置值要求能够被序列长度整除。

【--attention-mask-type】

设置mask类型，微调开启CP的场景下只能为general。

【--context-parallel-algo】

通过传入指定参数，选择不同的cp算法，具体包含如下几种：

1. [**megatron_cp_algo**](https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/ring-attention-context-parallel.md)
2. [**ulysses_cp_algo**](https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md)
3. [**hybrid_cp_algo**](https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/hybrid-context-parallel.md)

由于在微调场景，`--attention-mask-type`只能设置为`general`，所以理论上样本越短，拼接序列包含的样本数目越多，`--context-parallel-size`设置越大，性能收益越明显。但是要注意 seq-length / context-parallel-size > 8k时可以一定程度上弥补CP带来的通信损失，针对这种场景参考配置如下，参数相关介绍参考上述对应算法的链接。

```shell
    --seq-length 131072
    --context-parallel-size 8
    --context-parallel-algo megatron_cp_algo  # CP较小时(CP<=4），使用ulysses_cp_algo是性能不错的选择
    --attention-mask-type general
```

## 使用效果
|    模型     | 序列长度 | 分布式策略（TP/PP/CP） | gbs |       CP类型       | attention-mask-type | reset-attention-mask |  显存   | 吞吐 TFLOP/s/GPU |
|:---------:|:----:|:---------------:|:---:|:----------------:|:-------------------:|:------------------:|:-----:|:--------------:|
| Llama2-7B | 32k  |      2/1/4      | 16  | megatron_cp_algo |       general       |        True        | 52777 |     102.7      |
| Llama2-7B | 32k  |      2/1/4      | 16  | ulysses_cp_algo  |       general       |        True        | 53681 |     192.3      |


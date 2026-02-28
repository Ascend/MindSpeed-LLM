# 长序列微调

## 使用方法

### 数据预处理

数据预处理方法同[**多样本pack微调**](../solutions/finetune/multi_sample_pack_finetune.md)。

### 微调参数

【--is-instruction-dataset】

用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。

【--prompt-type】

用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../../../configs/finetune/templates.json)文件内查看。

【--reset-position-ids】

每条数据由不同的样本拼接而成，因此其位置 ID 并不连续。该参数用于将数据的position-ids按照eod结尾生成ids，而非连续的ids。模型将在每个EOD之后，将对position-ids从0开始重新编号，从而隔离不同句子间的位置计算，作用于attention中query和key的位置编码。

【--reset-attention-mask】

每条数据由不同的样本拼接而成，因此其attention mask不再是单纯的下三角形状。该参数开启时，会按照EOD计算句子的分隔位置，生成actual-seq-len，传入FA算子中相当于锯齿状的mask计算效果，FA进行TND格式计算。

【--context-parallel-size】

设置CP切分的并行数目，配置值要求能够被序列长度整除。

【--attention-mask-type】

设置`--attention-mask-type`类型：默认是causal，支持causal和general格式。
1. `--attention-mask-type`是general，attention-mask会从数据获取生成。
2. `--attention-mask-type`是causal，attention-mask会在FA前生成压缩固定长度(2048)的mask，性能和显存会比方案1更好，推荐使用。

【--context-parallel-algo】

通过传入指定参数，选择不同的cp算法，具体包含如下几种：

1. [**megatron_cp_algo**](https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/ring-attention-context-parallel.md)
2. [**ulysses_cp_algo**](https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md)
3. [**hybrid_cp_algo**](https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/hybrid-context-parallel.md)

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

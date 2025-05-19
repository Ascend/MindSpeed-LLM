### 长序列微调

## 数据集处理方法
数据集处理方法同[**多样本pack微调**](../solutions/finetune/multi_sample_pack_finetune.md)。


## 启动长序列微调脚本

【--is-instruction-dataset】

用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。

【--prompt-type】

用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[`templates`](../../../configs/finetune/templates.json)文件内查看。

【--reset-position-ids】

每条数据由不同的样本拼接而成，因此其位置 ID 并不连续。该参数用于为每条拼接的数据重置位置 ID，以确保在处理多个样本时，位置编码保持一致性。

【--context-parallel-size】

CP切分的并行数目，要可以被序列长度整除。

【--cp-attention-mask-type】

设置mask类型，微调开启CP的场景下只能为general。

【--adaptive-cp-manually-set-mask-list】

长序列场景完整mask会占用大量显存（约`seq-length * seq_length * 2`），并且影响端到端性能；当前pack模式长序列微调场景下，因mask具有一定规律性，
当`--context-parallel-algo`设置为 `adaptive_cp_algo` 或 `hybrid_adaptive_cp_algo`时，使能`--adaptive-cp-manually-set-mask-list`可以不生成完整的mask，
使能每个rank生成各自独立的mask_list，进而节省显存，提高端到端性能。

【--context-parallel-algo】

通过传入指定参数，选择不同的cp算法，具体包含如下几种：

1. [**megatron_cp_algo**](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/ring-attention-context-parallel.md)
2. [**ulysses_cp_algo**](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/ulysses-context-parallel.md)
3. [**hybrid_cp_algo**](https://gitee.com/ascend/MindSpeed/blob/master/docs/features/hybrid-context-parallel.md)

由于在微调场景，`--cp-attention-mask-type`只能设置为`general`，当训练样本为多条短样本拼接，CP较大时，建议优先选择`adaptive_cp_algo`，
理论上样本越短，拼接序列包含的样本数目越多，`--context-parallel-size`设置越大，性能收益越明显，但是要注意 seq-length / context-parallel-size > 8k时可以
尽可能弥补CP带来的通信损失，针对该种场景参考配置如下，参数相关介绍参考上述对应算法的链接。CP较小时（一般<=4），`ulysses_cp_algo`是性能不错的选择。

```bash
    --seq-length 131072
    --context-parallel-size 8
    --context-parallel-algo adaptive_cp_algo
    --cp-attention-mask-type general
    --adaptive-cp-manually-set-mask-list
    --adaptive-cp-dynamic-attn-mask
    --adaptive-cp-only-reschedule
```

## 使用效果
|    模型     | 序列长度 | 分布式策略（TP/PP/CP） | gbs |       CP类型       | cp-attention-mask-type | reset-position-ids |  显存   | 吞吐 TFLOP/s/GPU |
|:---------:|:----:|:---------------:|:---:|:----------------:|:----------------------:|:------------------:|:-----:|:--------------:|
| Llama2-7B | 32k  |      2/1/4      | 16  | megatron_cp_algo |        general         |        True        | 52777 |     102.7      |
| Llama2-7B | 32k  |      2/1/4      | 16  | ulysses_cp_algo  |        general         |        True        | 53681 |     192.3      |
| Llama2-7B | 32k  |      2/1/4      | 16  | adaptive_cp_algo |        general         |        True        | 52461 |     139.1      |

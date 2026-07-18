# Long-Sequence Fine-Tuning

## How to Use

### Data Preprocessing

The data preprocessing method is the same as that used for [**Multi-Sample Pack Fine-Tuning**](../../training/finetune/mcore/multi_sample_pack_finetune.md).

### Fine-Tuning Parameters

`--is-instruction-dataset`

Specifies that fine-tuning uses an instruction dataset to ensure that the model is trained on the specified instruction data.

`--prompt-type`

Specifies the model template. It enables the base model to become more conversational after fine-tuning. You can view the available `prompt-type` options in the [`templates`](../../../../../configs/finetune/templates.json) file.

`--reset-position-ids`

Each data item is formed by concatenating different samples. Therefore, its position IDs are not contiguous. This parameter generates position IDs based on end-of-document (EOD) markers instead of contiguous IDs. After each EOD, the model renumbers the position IDs from 0. Therefore, positional computation is isolated across different sentences, which affects the position encoding of the query and key in attention.

`--reset-attention-mask`

Each data item is formed by concatenating different samples. Therefore, the attention mask is no longer a simple lower-triangular shape. When you enable this parameter, the system calculates sentence boundaries according to EOD and generates `actual-seq-len`. It then passes the result to the FA operator, which produces a jagged mask pattern before performing TND-format computation.

`--context-parallel-size`

Sets the number of parallel partitions for CP sharding. The configured value must evenly divide the sequence length.

`--attention-mask-type`

The default value is `causal`, and the `causal` and `general` formats are supported.

1. When `--attention-mask-type` is `general`, the attention mask is generated from the data.
2. When `--attention-mask-type` is `causal`, the attention mask is generated as a compressed fixed-length mask of 2048 before FA. This approach delivers better performance and uses less memory than option 1. Therefore, it is recommended.

`--context-parallel-algo`

Use the specified parameter to select a CP algorithm. The available algorithms are as follows:

1. [**megatron_cp_algo**](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/ring-attention-context-parallel.md)
2. [**ulysses_cp_algo**](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/ulysses-context-parallel.md)
3. [**hybrid_cp_algo**](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/hybrid-context-parallel.md)

```shell
    --seq-length 131072
    --context-parallel-size 8
    --context-parallel-algo megatron_cp_algo  # When CP is small (CP <= 4), using ulysses_cp_algo is a good performance choice.
    --attention-mask-type general
```

## Results

| Model | Sequence Length | Distributed Strategy (TP/PP/CP) | gbs | CP Type | attention-mask-type | reset-attention-mask | Memory Usage | Throughput TFLOP/s/GPU |
|:-----:|:---------------:|:-------------------------------:|:---:|:-------:|:-------------------:|:--------------------:|:------------:|:----------------------:|
| Llama-2-7B | 32k | 2/1/4 | 16 | megatron_cp_algo | general | True | 52777 | 102.7 |
| Llama-2-7B | 32k | 2/1/4 | 16 | ulysses_cp_algo | general | True | 53681 | 192.3 |

# KVAllGather Long-Sequence Parallelism

## Feature Overview

For sparse flash attention, lightning indexer, and lightning indexer loss, perform an All-Gather communication on the partitioned key and value tensors before computation to obtain the complete key and value tensors.

For a detailed introduction, see [**kvallgather_cp_algo**](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/kvallgather-context-parallel.md).

## Usage

| Key Parameter | Description |
| --- | --- |
| `--context-parallel-size [int]` | Number of context-parallel ranks to enable. The default is 1. Configure it based on your requirements. |
| `--context-parallel-algo` <b>kvallgather_cp_algo</b> | Long-sequence parallelism algorithm option. Set it to `kvallgather_cp_algo` to enable KVAllGather long-sequence parallelism. |
| `--seq-length [int]` | Input sequence length. |

## Notes

1. This feature currently supports only the sparse flash attention fused operator, the lightning indexer fused operator, and the lightning indexer loss fused operator.
2. This feature currently supports only `attention-mask-type` set to `causal`.
3. This feature supports only fixed-length padding training scenarios. It uses a load-balanced sequence partitioning method, and `--seq-length` must be divisible by `2 * context-parallel-size`.

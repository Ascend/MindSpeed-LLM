# Ring Attention for Long-Sequence Parallelism

## Problem Analysis

Across generative AI and research models, long-sequence training is becoming increasingly important. In generative AI, tasks such as conversational AI, long-document summarization, and video generation all require reasoning over long contexts in both spatial and temporal dimensions. Similarly, chapter-level and book-level summarization, which can span tens of thousands or even hundreds of thousands of Chinese characters, is also important for conversational AI and summarization tasks. Existing parallelization methods such as data, tensor, and pipeline parallelism cannot partition along the sequence dimension. As the sequence dimension (`S`) grows, training memory overhead increases on the order of $O(S^2)$. Therefore, long-sequence scenarios require targeted optimizations to meet training demands.

## Solution

Ring Attention supports long-sequence parallelism to address scaling issues in the sequence dimension. For details, refer to the original paper:
> Ring Attention with Blockwise Transformers for Near-Infinite Context (<https://arxiv.org/pdf/2310.01889>)

### Approach

Ring Attention draws on the blockwise softmax principle to perform blockwise attention computation without needing the full matrix for the entire sequence. Therefore, the authors propose executing self-attention and feed-forward network computations in blocks, distributing the sequence dimension across multiple devices. Specifically, the method builds a ring communication structure for attention computation blocks among processes, and each process holds a sharded local QKV block. After local attention is computed, KV blocks are sent backward and fetched forward around the process-device ring, enabling attention and feed-forward computations block by block. At the same time, local attention computation and KV-block communication can overlap, eliminating additional communication overhead. In addition, this approach does not require any data concatenation during attention computation. Therefore, the supported sequence length can theoretically grow without bound.

## Use Cases

When training GPT-like models, once data enters MoE layers, the actual sequence length exceeds 8K.

Unlike the Ulysses approach, this method does not require `head_size` to be divisible by `cp_size`.

It is compatible with FlashAttention, which is enabled by default.

To overlap computation and communication, in theory you need to ensure that the sequence length assigned to each compute block satisfies `c >= F/B`. Here, `F` is the FLOPS of each device and `B` is the bandwidth between devices. See the original paper for the detailed derivation. In practice, ensure that each compute block receives a sufficiently large sequence length to achieve good overlap.

## Usage

| Key Parameter | Description |
| --- | --- |
| --context-parallel-size [int] | Sets the number of context-parallel ranks. The default is 1. Configure it based on your requirements. |
| --seq-length [int] | Input sequence length. |
| --use-cp-send-recv-overlap | Recommended. Enables send-receive overlap. |
| --attention-mask-type [general/causal] | Optional. Sets the mask computation type. The default is causal (triangular) mask computation. Setting `general` enables full computation. |
| --context-parallel-algo megatron_cp_algo | Long-sequence parallelism algorithm option. The default is `ulysses_cp_algo`. Set it to `megatron_cp_algo` to enable Ring Attention. |

## Effects

By splitting the input sequence across multiple compute devices in parallel, memory consumption on a single device is reduced. Compared with not enabling sequence parallelism, single-step latency increases, while computation efficiency improves compared with recomputation.

## Notes

1. When Context Parallel is enabled, you must also enable FlashAttention. Otherwise, this feature is not supported.
2. When training GPT-like models, you are advised to set `attention-mask-type` to `causal`.
3. With an 8K sequence length, computation time becomes shorter. Therefore, the send and receive time after CP partitioning may exceed the computation time, which degrades performance. Therefore, you are advised to configure `seq-length / context-parallel-size > 8K` for the best results. The specific formula is `S/(T*alpha) >= 1/(W*beta)`, where `S = seq-length / context-parallel-size`, `T` is the theoretical compute capability of the chip, `alpha` is computation efficiency, `W` is the theoretical communication bandwidth, and `beta` is bandwidth utilization.

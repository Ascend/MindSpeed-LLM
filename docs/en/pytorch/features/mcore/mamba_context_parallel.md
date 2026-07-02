# Mamba-CP

## Background

Mamba is introduced to address the quadratic complexity of Transformer sequence lengths and has become an important architecture for training on long sequences. As sequence length grows, activations place increasing pressure on memory usage. Therefore, context parallelism (CP) remains urgently needed to reduce the memory pressure caused by ultra-long sequences. However, external open-source Mamba frameworks still do not support CP.

## Problem

The recursive state space model (SSM) computation steps in Mamba are temporally dependent. Traditional CP must wait for the previous CP rank to complete its computation and pass the result to the next CP rank before it can execute the next step. This creates idle waiting. Therefore, we designed a parallel Mamba-CP scheme that allows all ranks to perform state transfer computation concurrently. Compared with traditional CP, it delivers substantial performance gains.

See Figure 5 in the Mamba-2 paper for the traditional CP design.

## Solution

For the state transfer portion with temporal dependencies, perform AllGather on `local_decay` and `local_state` across each CP rank so that all CP ranks can perform the state transfer computation concurrently. We also overlap computation with communication during the forward AllGather and backward ReduceScatter operations.

## Use Cases

1. CP is orthogonal to TP and SP. You can enable CP on top of TP to reduce memory usage further.
2. TP has an `n_groups` divisibility restriction, but CP does not.
3. In memory-constrained scenarios, CP saves more memory than recomputation.

## How to Use

| Key Parameter | Description |
| --- | --- |
| --context-parallel-algo mamba_cp_algo | Option for the long-sequence parallelism algorithm. The default value is `ulysses_cp_algo`. Set it to `mamba_cp_algo` to enable Mamba-CP. |
| --context-parallel-size [int] | Number of ranks to use when CP is enabled. The default value is 1. Configure it based on your requirements. |

## Results

Memory-saving methods include recomputation and additional CP. Recomputation is common, but it adds about 30 percent to runtime. In long-sequence, memory-constrained scenarios, you usually need to combine features to maximize performance without exceeding hardware memory. The following table compares the memory savings from enabling CP and from recomputation under the same memory footprint.

Memory optimization and performance before and after enabling CP

| Sequence Length | Parallel Configuration | Memory Usage | Memory Optimization | Performance | Performance Change |
| --------------- | ---------------------- | ------------ | ------------------- | ----------- | ------------------ |
| 32K             | TP4CP1                 | 56129 MB     | N/A                 | 3761.1 ms   | N/A                |
| 32K             | TP4CP2                 | 32613 MB     | 42%                 | 3862.3 ms   | -2.69%             |

Comparison of memory reduction and performance between CP and recomputation

| Sequence Length | Parallel Configuration      | Memory Usage         | Performance | Speedup |
| --------------- | --------------------------- | -------------------- | ----------- | ------- |
| 32K             | TP4CP1 + full recomputation | Same memory as 30 GB | 4728.8 ms   | N/A     |
| 32K             | TP4CP2                      | Same memory as 30 GB | 3862.3 ms   | +22.43% |

## Notes

1. When you need to save memory in a Mamba-CP scenario, enable CP first and then enable recomputation.

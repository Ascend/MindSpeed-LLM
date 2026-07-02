# Communication over Computation for Computation-Communication Parallelism

## Problem Analysis

During LLM training, the forward and backward passes of `ColumnParallelLinear` and `RowParallelLinear` both contain adjacent computation-communication pairs with sequential dependencies. The computation uses `Matmul`, while the communication uses `AllReduce` when sequence parallelism is disabled, or `AllGather` and `ReduceScatter` when sequence parallelism is enabled. Because these computation-communication pairs have sequential dependencies, meaning that the input of the next stage is the output of the previous stage, they often run serially. In that case, both the computation stream and the communication stream wait idle, and execution efficiency does not reach its maximum.

## Solution

Split the computation and communication tasks into finer-grained subtasks and overlap them through pipelining.

### Implementation Approach

#### Python Script Implementation

Further split the tensors into two, four, or eight parts, and use Python scripts to overlap computation and communication for each subtensor. Therefore, this improves utilization of the computation and communication streams.

#### Fused Operator Implementation

Based on the MTE remote memory access capability, implement finer-grained computation and communication subtasks inside the operator through a fused large-kernel implementation to achieve pipeline overlap.

## Use Cases

This feature is mainly used in training scenarios. It applies when the `Attention` module and the `MLP` module execute serially and the computation-communication tasks have sequential dependencies and are adjacent in position.

When you use the Python script implementation, the `m` dimension of the left matrix in `Matmul` must be a multiple of the split count, which can be 2, 4, or 8, and it is not suitable for cases where the computation and communication segments have very different durations. Note that when the script-side implementation uses a large number of matrix splits and split counts, it can easily become host bound and fail to produce the expected benefit. It supports the three communication scenarios `ALL_REDUCE`, `ALL_GATHER`, and `REDUCE_SCATTER`, and it allows you to flexibly choose communication first or computation first.

The following computation-communication fused operators are already supported.

1. `MATMUL_ALL_REDUCE`, which performs computation first and communication second, and its deterministic computation variant.
2. `MATMUL_REDUCE_SCATTER`, which performs computation first and communication second, and its deterministic computation variant.
3. `ALL_GATHER_MATMUL` and `ALL_GATHER_MATMUL_V2`s, which perform communication first and computation second. The `V2` interface supports access to the `ALL_GATHER` intermediate result.
4. Quantization scenarios. `MATMUL_ALL_REDUCE` supports W8A16 pseudo-quantization in the FP16 format, with granularities that include per tensor, per channel, and per group.

## How to Use

There are currently two ways to enable computation-communication parallelism. You can use either the Python script option or the fused operator option.

Choose one of the following two scenarios as needed.

Set `--use-ascend-coc` to enable computation-communication parallelism. Use the following options to configure it.

### 1. Enabling Computation-Communication Parallelism through a Python Script

```shell
--use-ascend-coc
--coc-parallel-num 2 # or 4, or 8
```

### 2. Enabling Computation-Communication Parallelism through Fused Operators

Note: You must install ATB before you can use the computation-communication parallel fused operators.

ATB installation:

- Install the binary package. After you install the CANN-NNAL package, run `source /usr/local/Ascend/nnal/atb/set_env.sh`.

```shell
--use-ascend-coc
--coc-fused-kernel # Note: Currently, only the TP=8 scenario is supported.
```

When you use both `coc-parallel-num > 1` and `coc-fused-kernel`, the `coc-fused-kernel` parameter takes priority and overrides `coc-parallel-num > 1`.

## Notes

This feature is not yet compatible with `--use-ascend-mc2`.

This feature is not yet adapted for MoE models.

HDK supports versions later than 2024 RC2. CANN supports versions later than 2024 RC4.

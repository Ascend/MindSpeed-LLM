FSDP2 是 PyTorch 分布式并行的下一代范式，旨在解决 FSDP1（`FlatParameter` 包装器模式）在灵活性与组合性上的痛点。它不再通过 Python 类包装模型，而是通过 **`torch.distributed.fsdp.fully_shard`** API 对模型进行原位（In-place）的并行化处理。

---

## 1. FSDP2 vs FSDP1：从 FlatParameter 到 Per-Parameter Sharding

与 FSDP1 将多个参数打平（Flatten）为一个巨大的 `FlatParameter` 不同，FSDP2 采用了 **Per-Parameter Sharding（逐参数分片）** 的策略。

* **FSDP1 (Legacy)**：将层内的所有参数拉平拼接，切分这个巨大的 1D 向量。这破坏了原始模型的参数结构，导致对参数的某些操作（如自定义初始化、特定层微调）变得复杂。
* **FSDP2 (New)**：保持模型原始的参数结构不变。每个参数（`nn.Parameter`）被单独切分并管理。这种设计使得 FSDP2 具有极高的组合性（Composability），可以轻松与 Tensor Parallel (TP) 或 Checkpointing 结合。

## 2. 核心工作原理

在 **DistributedDataParallel (DDP)** 训练中，每个 Rank（GPU）都拥有一个完整的模型副本，并处理一个独立的数据 Batch，最后通过 **All-Reduce** 在所有 Rank 间同步梯度。

与 DDP 相比，**FSDP (Fully Sharded Data Parallel)** 通过对模型参数、梯度和优化器状态进行**切片 (Sharding)**，显著降低了 GPU 显存占用。这使得在单卡显存受限的情况下训练超大模型成为可能。

#### FSDP 参数生命周期 (Workflow)

如下图所示，FSDP 将 DDP 的 All-Reduce 操作分解为 Reduce-Scatter 和 All-Gather：

1. **Fully Sharded (静止态)**：在 Forward 和 Backward 计算之外，参数是完全切分的（每张卡只存 1/N）。
2. **All-Gather (准备态)**：在 Forward 和 Backward 开始前，切分的参数被广播聚合为完整的参数（Unsharded Parameters）。
3. **Compute (计算态)**：使用完整参数进行计算。
4. **Reduce-Scatter (同步态)**：在 Backward 内部，计算出的完整梯度（Local Unsharded Gradients）被立即归约并切分（Reduce-Scatter）为切片梯度。
5. **Update (更新态)**：优化器使用切片梯度更新切片参数，因此优化器状态也是切分的。
   ![](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026013110007727/38333772/847451cca7674365aa19d0b33da03ca2.png)

## 3. 核心引擎：DTensor (Distributed Tensor)

FSDP2 的底层基石是 **DTensor (`torch.distributed.tensor.DTensor`)**。

* **逻辑视图与物理视图分离**：
* **逻辑上**：参数看起来仍然是一个完整的 Tensor（例如 `[4096, 4096]`），保持了与单卡训练一致的编程体验。
* **物理上**：参数实际上被切分并分布在 `DeviceMesh` 定义的设备组中（例如每张卡只持有 `[512, 4096]` 的 `Local Tensor`）。
* **DeviceMesh**：FSDP2 依赖 `DeviceMesh` 来描述设备的拓扑结构。这使得它天然支持多维并行（例如 2D FSDP，或 FSDP + TP），只需定义不同的 Mesh 维度即可。

## 4. 混合精度 (Mixed Precision)

FSDP2 通过 `MixedPrecisionPolicy` 提供了灵活的精度控制，严格区分了**存储精度**、**计算精度**和**通信精度**。

* **Param Dtype (计算)**：在 Forward/Backward 计算前，FSDP2 会自动将参数 Cast 为低精度（如 `bfloat16`）。
* **Reduce Dtype (通信)**：在梯度同步（Reduce-Scatter）阶段，为了保证数值稳定性，通常将梯度 Cast 为高精度（如 `float32`）进行累加。
* **Buffer Dtype**：独立控制 Buffer（如 BatchNorm 统计量）的精度，防止溢出。

```python
# FSDP2 混合精度流程
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
# Forward: Parameters (FP32 storage) -> Cast to BF16 -> Compute
# Backward: Gradients (BF16) -> Cast to FP32 -> AllReduce
```

## 5. 通信与计算重叠 (Forward/Backward with Prefetching)

为了极致的训练效率，FSDP2 实现了高度优化的**通信计算重叠（Overlap）**机制，即 **Prefetching（预取）**。
![FSDP Implicit](https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026013110007727/38335790/7993d5bc90874265b2284f8a7ef87bef.png)

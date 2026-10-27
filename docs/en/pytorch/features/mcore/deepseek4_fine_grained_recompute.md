# DeepSeek V4 Fine-Grained Recompute 

## Overview

DeepSeek V4 fine-grained recompute reduces NPU memory usage by recomputing some intermediate results in CSA and MHC during the backward pass, which reduces the number of activations stored during training. This feature adds a certain amount of recomputation overhead. Therefore, you are advised to enable it based on the model configuration and memory conditions.

CSA recompute and MHC recompute are independent of each other. You can enable them separately or together.

## How to Use

| Key Parameter | Description |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| --recompute-csa-attention | Enables DeepSeek V4 CSA fine-grained recompute. It covers the supported Q projection, Indexer, Sparse Attention, and output projection computations. When combined with `--recompute-norm`, it recomputes the Q normalization of the target layers. |
| --mhc-recompute | Enables fine-grained recompute of the fused NPU MHC pre/post. You must also configure `--enable-mhc` and `--use-fused-mhc`. |

To enable only CSA fine-grained recompute:

```shell
--recompute-csa-attention
```

To enable only MHC fine-grained recompute:

```shell
--enable-mhc \
--use-fused-mhc \
--mhc-recompute
```

To enable both types of fine-grained recompute:

```shell
--recompute-csa-attention \
--enable-mhc \
--use-fused-mhc \
--mhc-recompute
```

## Notes

1. This feature currently applies only to DeepSeek V4 MCore training scenarios, and it is disabled by default.
2. `--recompute-csa-attention` and `--mhc-recompute` have no parameter dependency between them. When you enable both, they coordinate the release timing of intermediate activations.
3. `--mhc-recompute` takes effect only for the fused NPU MHC implementation. Therefore, you must enable `--enable-mhc` and `--use-fused-mhc` at the same time.
4. Fine-grained recompute takes effect only during training with gradient computation enabled. This recompute is not enabled for the MTP layer at present.
5. Once enabled, this feature trades additional computation overhead for memory savings. The actual benefit depends on the model size, sequence length, and parallelism configuration.

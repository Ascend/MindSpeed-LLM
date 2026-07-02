# High-Dimensional Tensor Parallelism

Refer to [High-Dimensional Tensor Parallelism](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/tensor-parallel-2d.md) for an introduction to this feature.

## Use Cases

Use this feature when the cluster scale is large and the TP domain is large, for example when training Llama-3-405B on A3 with `TP=16`.

## Usage

### Basic Parameters

Add `--tp-2d` to the training script argument list to enable 2D tensor parallelism. Set the split sizes on the x and y axes with `--tp-x N1` and `--tp-y N2`, respectively. Ensure that `tp = N1 * N2` and that `N1 > 1` and `N2 > 1`.
For example:

```bash
    --tensor-model-parallel-size 16 \
    --tp-2d \
    --tp-x 8 \
    --tp-y 2 \
```

### Other Optimization Parameters

These parameters help hide communication overhead for the 2D tensor parallelism feature and take effect only when `tp-2d` is enabled:

 - `--enable-overlap-ag-with-matmul`: Hides all-gather communication behind the linear layer forward pass to improve performance.
 - `--enable-overlap-matmul-with-rs`: Hides matmul computation behind reduce-scatter communication during the linear layer forward pass to improve performance.
 - `--coc-fused-kernel`: Enables a fused compute and communication operator during the linear layer forward pass. This fuses matmul with all-gather and reduce-scatter at the operator level for further acceleration. This feature is incompatible with the previous two features and depends on the ATB acceleration library.
 - `--enable-backward-overlap-ag-with-matmul`: Hides all-gather communication behind matmul when computing gradients in the linear layer backward pass to improve performance. This feature depends on the ATB acceleration library.

**Note:** Only one of the three forward-pass optimization parameters, `--enable-overlap-ag-with-matmul`, `--enable-overlap-matmul-with-rs`, and `--coc-fused-kernel`, can be enabled at the same time.

## Usage Constraints

 - This feature is incompatible with `--sequence-parallel` and `--use-fused-rmsnorm`. You must disable those features before you use this feature.
 - This feature does not yet support MoE models or related features.
 - This feature is recommended for ultra-large dense models and scenarios with a large TP domain, such as Llama-3-405B with `TP=16`. Smaller models and smaller TP settings may reduce performance. Adjust the configuration according to the actual environment.
 - When training Llama-3-405B with `TP=16`, use 2D tensor parallelism and set `tp-x=8` and `tp-y=2`. In other scenarios, you need to tune `tp-x` and `tp-y` based on differences in compute efficiency and communication group partitioning. Some configurations do not guarantee better performance.
 - The fused operator depends on CANN 8.0.1.B020 or later. Install CANN-NNAL and initialize the additional integration step. The fused-operator scenario currently supports only `micro-batch-size=1`.

# Async Activation Offload

## Background and Challenges

As model parameter counts grow and sequence lengths increase, memory demand during training rises sharply. Current approaches to optimizing activation memory mainly rely on activation recomputation and sequence parallelism. These techniques have the following bottlenecks:

- Activation recomputation saves memory by discarding activation values from the forward pass and recomputing them during the backward pass. This introduces a large amount of redundant computation.
- Sequence parallelism can reduce memory use by distributing the computation for a single sequence across multiple devices. However, frequent cross-device communication can be difficult to hide effectively.

To address these challenges, you can use the asynchronous activation offload strategy.

## Solution

- Memory optimization: Offload activation tensors from the device side to the host side, which significantly reduces peak memory usage.
- Asynchronous execution: Use multiple streams to make offload (device-to-host, D2H) and load (host-to-device, H2D) operations asynchronous. Therefore, the copy overhead can overlap with computation.
- Early prefetch: During the backward pass, use the prefetch mechanism to load the tensors needed next in advance and hide load latency.

## Usage

This feature supports organizing tensor lifecycles by block, which lets you manage activation values for different model blocks flexibly. The following example shows how to use it.

```python
with async_save_on_cpu(
    h2d_stream=h2d_stream,
    d2h_stream=d2h_stream,
    block_idx=block_idx,
    depth=depth,
    custom_check_fn=your_check_fn
):
    # The forward pass for one block of the model. This is only an example.
    output = layer(input)
```

### Parameter Details

- `h2d_stream` / `d2h_stream`: The H2D and D2H streams. It is recommended that you create dedicated global streams for H2D and D2H tasks so they run asynchronously with the computation stream.
- `block_idx`: The index of the current block in the model.
- `depth`: The total number of layers in the model.
- `custom_check_fn`: A custom validation function. Only activation values that return `True` after validation are offloaded. In practice, you should select parts with heavy computation and a small activation footprint, and combine them with a recomputation strategy. For activation values with a large activation footprint and short compute time, use recomputation. For activation values with a small activation footprint and long compute time, use offload. Otherwise, the overhead of H2D and D2H becomes too large and is difficult to hide with computation.

## Use Cases and Results

- Long-sequence scenarios: The computation cost of self-attention grows quadratically with sequence length. With this approach, you can offload the activation values from the forward pass of self-attention and skip recomputing self-attention during the recomputation stage. In typical scenarios, end-to-end performance improves by more than 20 percent.
- FSDP2 scenarios: Under the FSDP2 distributed strategy, model parameters are partitioned and gathered. For shorter sequence lengths, computation time cannot mask communication time. You can use this approach to offload the activation values at the recomputation entry point. After saving memory, you can increase the micro-batch size or sequence length to raise the compute-to-communication ratio. In typical scenarios, end-to-end performance improves by more than 60 percent.

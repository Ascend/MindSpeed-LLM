# Performance Tuning

Performance tuning is an important part of training. Proper tuning can significantly improve model training efficiency and reduce resource consumption. In this document, performance refers to the time required to complete one end-to-end training step for a given model and input data on a machine (GPU, NPU, or other platform). To account for differences in training data size and number of epochs across models, performance here is defined as the time taken to complete training on a single batch. The term "end-to-end" refers to the process of completing a single training step of an AI model — in other words, the performance measurement and optimization discussed in this document are both approached from the perspective of the model.

For a single batch, the time is mainly composed of the following parts:

Total single-batch time = Data loading time + Forward/backward time + Optimizer time + Post-processing time + Communication time + Scheduling time

Each component is described below:

Data loading time: Time the model spends loading the data it needs (such as images, videos, and text), including reading data from hardware storage devices to the CPU, preprocessing (encoding, decoding, etc.) in the CPU, and transferring CPU data to the device. For models that need to be split across multiple devices, data loading also includes the time to broadcast from the data loading device to other devices.

Forward/backward time: Time taken by the forward and backward passes of the deep learning model, including forward data computation and backward gradient computation.

Optimizer time: Time taken to update model parameters.

Post-processing time: Time after optimizer updates, including data post-processing and some necessary synchronization operations, usually specific to the model.

Communication time: Communication time between devices in a single node and between nodes in a multi-node setup. Due to the mechanisms of PyTorch, when communication and computation can overlap, this represents the communication time that is not hidden by computation.

Scheduling time: Time required for the model to go from CPU instructions to invoking kernel operations on the NPU side.

## Performance Data Collection

During training, we need to collect performance data to help analyze model performance issues and identify bottlenecks. MindSpeed LLM supports collecting profiling data based on Ascend chips to provide insights into model operation. For usage guidance, refer to [Profiling Data Collection](../tools/profiling.md).

## Performance Analysis Workflow

After collecting performance data, you can use [MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md) to visualize and analyze the data to locate performance bottlenecks.

MindStudio Insight is a performance analysis tool provided by Ascend that supports multi-dimensional analysis of profiling data, including:

- **Operator time analysis**: Identify operators with long execution times and locate computation bottlenecks.
- **Communication time analysis**: Analyze the time ratio of communication to computation and optimize communication strategies.
- **Memory analysis**: View memory usage and identify memory bottlenecks.
- **Pipeline analysis**: Analyze bubble ratios in pipeline parallelism.

## Performance Tuning Methods

MindSpeed-LLM provides a variety of performance tuning features. You can select appropriate strategies based on your scenario — see [Training Schemes and Features](../features/README.md) for details.

Common performance tuning features include:

- **Long Sequence Parallelism**: Reduces per-device computation by splitting the sequence dimension, supporting Ascend Ring Attention, Ulysses, etc. See [Ring Attention for Long-Sequence Parallelism](../features/mcore/ring-attention-context-parallel.md) and [Ulysses Context Parallel](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/ulysses-context-parallel.md).
- **Asynchronous Activation Offloading**: Offloads activation values to the host side, using asynchronous mechanisms to hide copy overhead behind computation and reduce peak memory usage. See [Async Activation Offload](../features/mcore/async_activation_offload.md).

Additionally, for sequence length training not covered in the examples directory, refer to [Long Sequence Out-of-the-Box Tuning Guide](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E8%B0%83%E4%BC%98%E6%8C%87%E5%8D%97%2F%E9%95%BF%E5%BA%8F%E5%88%97%E5%BC%80%E7%AE%B1%E8%B0%83%E4%BC%98%E6%8C%87%E5%8D%97.md).

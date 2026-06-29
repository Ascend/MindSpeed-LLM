# 性能调优

性能优化是训练的重要环节，合理的性能调优可以显著提升模型训练效率，降低资源消耗。性能在本文中，指机器（GPU、NPU或其他平台）在指定模型和输入数据的背景下，完成一次端到端训练所需要花费的时间，考虑到不同模型的训练数据量和训练轮次（epoch）差异，此处定义的性能是在完成一个batch训练所需花费的时间。而这里的端到端，我们通常是指完成一个人工智能模型单步训练的过程，也就是说，本文所讨论的性能的衡量和性能的优化，都是站在模型角度上。

对于一个batch而言，时间主要由以下部分构成：

单batch总时间 = 数据加载时间 + 模型前反向时间 + 优化器时间 + 模型后处理时间 + 通信时间 + 调度时间

各组成成分介绍如下：

- 数据加载时间：模型在加载自身所需要的数据（如图片、视频和文本等）的时间，包括将数据从硬件存储设备读取到CPU（Central Processing Unit）中、CPU中数据的预处理（编解码等操作）、CPU数据放到Device上的时间。对于一些需要切分在若干张卡上的模型，数据加载还包括从数据加载卡广播到其他卡上的时间。

- 模型前反向时间：深度学习模型的前向过程和反向过程的时间，即Forward和Backward过程，包含前向的数据计算和反向的数据微分求导的时间。

- 优化器时间：模型参数更新时间。

- 模型后处理时间：优化器更新后的时间，包括数据的后处理或者一些必要的同步操作，通常取决于模型特有操作。

- 通信时间：单节点时卡之间和多节点时节点之间的通信时间。由于PyTorch的特殊机制，在通信和计算可以并行的情况下，表示未被计算掩盖的通信时间。

- 调度时间：模型从CPU的指令到调用NPU侧的核（Kernel）所需要的时间。

## 性能数据采集

在训练过程中，我们需要采集模型的性能数据帮助我们分析模型的性能问题，找到性能瓶颈。MindSpeed LLM支持基于昇腾芯片采集profiling数据，以提供对模型运行情况的分析，使用指导请参考[性能数据采集工具](../tools/profiling.md)。

## 性能分析流程

采集到性能数据后，可通过[MindStudio Insight](https://www.hiascend.com/document/detail/zh/mindstudio/2600/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)对性能数据进行可视化分析，定位性能瓶颈。

MindStudio Insight是昇腾提供的性能分析工具，支持对profiling采集的数据进行多维度分析，包括：

- 算子耗时分析：识别耗时较长的算子，定位计算瓶颈。
- 通信耗时分析：分析通信与计算的时间占比，优化通信策略。
- 内存分析：查看显存使用情况，识别显存瓶颈。
- 流水线分析：分析流水线并行的空泡比例。

## 性能调优方法

MindSpeed LLM提供了多种性能调优特性，可根据实际场景选择合适的策略，具体可参考[训练方案与特性说明](../features/README.md)的说明。

常见的性能调优特性包括：

- 长序列并行：通过切分序列维度降低单卡计算量，支持Ascend Ring Attention、Ulysses等，详见[Ascend Ring Attention 长序列并行](../features/mcore/ring-attention-context-parallel.md)和[Ulysses 长序列并行](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/ulysses-context-parallel.md)。
- 异步激活值卸载：将激活值卸载至Host侧，利用异步机制使拷贝被计算掩盖，降低峰值显存，详见[异步激活值卸载](../features/mcore/async_activation_offload.md)。

# 虚拟流水线并行

## 问题分析

PipeDream流水线并行切分粒度过大，运行过程中仍然有许多空泡(bubble)，计算资源利用率仍有提高空间。

## 解决方案

将计算进一步细分，减少空泡。

### 解决思路:

在设备数量不变的情况下，分出更多的流水线阶段，以更多的通信量，换取空泡比率降低。

![Alt text](../../../sources/images/virtual_pipeline_parallel/virtual-pipeline.png)

[原文链接](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf)

为了方便理解，举一个例子：假设模型层数为16，张量并行大小为1，流水线并行大小为4，虚拟流水线并行大小为2。模型会被分为 4 * 2 = 8 个阶段，每个阶段 16 / 8 = 2 个层。

    Device 0: [1, 2] [9, 10]
    Device 1: [3, 4] [11, 12]
    Device 2: [5, 6] [13, 14]
    Device 3: [7, 8] [15, 16]

前向的顺序为 device 0 -> device 1 -> device 2 -> device 3 -> device 0 -> device 1 -> device 2 -> device 3

## 使用场景

想要进一步减小空泡比率，提升性能

## 使用方法

设置`--num-layers-per-virtual-pipeline-stage N`。表示每个阶段的层数。要求模型的总层数 L % N == 0。

## 使用效果

空泡比率进一步减小

## 注意事项

Megatron虚拟流水并行VPP影响权重切分方式，保存、加载权重时需保证VPP配置一致，才能正常加载；


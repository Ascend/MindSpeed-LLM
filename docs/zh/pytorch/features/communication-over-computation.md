# 计算通信并行 CoC (Communication Over Computation)

## 问题分析

大模型训练过程中，其ColumnParallelLinear和RowParallelLinear部分的前向/反向均存在相互毗邻、顺序依赖的计算通信组合，计算为Matmul，而通信则为AllReduce（不开启序列并行）或AllGather和ReduceScatter（开启序列并行）。这些计算通信的组合因为存在顺序依赖（即后一个的输入是前一个输出），常常被串行执行，但这时候计算和通信流都存在一定的空闲等待时间，该过程的执行效率没有被最大化。

## 解决方案

通过将计算和通信任务分别拆分成更细粒度的子任务来实现相互的流水掩盖。

### 解决思路

#### Python脚本侧实现
将张量进行进一步切分（2/4/8份），通过Python脚本的方式实现每个子tensor之间计算和通信的并行，从而增大计算和通信流的利用率；

#### 融合算子实现
基于MTE远端内存访问能力，以融合大Kernel方式在算子实现的内部将计算和通信任务分别拆分成更细粒度的子任务来实现相互的流水掩盖；

## 使用场景
该特性目前主要用于训练场景，当Attention模块和MLP模块串行执行且计算通信存在顺序依赖与位置毗邻关系时适用。

使用Python脚本侧实现时，对Matmul左矩阵的m轴有一定要求，必须是切分数（2/4/8）的倍数，且不适用于计算与通信片段耗时相差较大的情况。需要注意的是，脚本侧实现在切分矩阵、切分数量较大时，容易出现host bound问题，从而不能得到预期的收益。支持ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER三个通信场景，支持灵活设置先通信或先计算。

对于计算通信融合算子，目前已支持：
1. MATMUL_ALL_REDUCE融合算子（先计算后通信）及其确定性计算；
2. MATMUL_REDUCE_SCATTER融合算子（先计算后通信）及其确定性计算；
3. ALL_GATHER_MATMUL, ALL_GATHER_MATMUL_V2融合算子（先通信后计算）（V2版本接口支持ALL_GATHER中间结果获取）；
4. 量化场景：MATMUL_ALL_REDUCE融合算子支持fp16格式的w8A16伪量化，粒度包含per tensor / per channel / per group；

## 使用方法

当前计算通信并行有两种实现方法：python脚本使能、融合算子使能，两者选其一即可。

请根据需要选择下列两种场景中的一个进行使用。

设置--use-ascend-coc使能计算通信并行功能，使用方式通过如下变量进行设置：

### 1. 使用通过Python脚本使能的计算通信并行特性

```shell
--use-ascend-coc 
--coc-parallel-num 2 # 或者4，或者8
```

### 2. 使用通过融合算子使能的计算通信并行特性
注意：计算通信并行融合算子需要安装ATB后才能使用！

ATB安装方法：

- 二进制包安装：安装CANN-NNAL包之后, source /usr/local/Ascend/nnal/atb/set_env.sh
```shell
--use-ascend-coc
--coc-fused-kernel # 注意：当前只支持TP=8的场景！
```

同时使用coc-parallel-num > 1和coc-fused-kernel参数时，coc-fused-kernel参数优先级更高，会覆盖coc-parallel-num > 1

## 注意事项
暂不兼容 --use-ascend-mc2 特性。

当前暂未适配MoE模型。

HDK支持2024年RC2之后版本，CANN支持2024年RC4之后版本。

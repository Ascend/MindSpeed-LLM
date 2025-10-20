# Ring Attention长序列并行

## 问题分析

从生成性AI到科研模型，长序列训练正在变得非常重要。 在生成性AI领域，会话式AI、长文档摘要和视频生成等任务都需要在空间和时间层面对长上下文进行推理。 同样，章节和书籍级别的摘要（数万甚至数十万字）在会话式AI和摘要任务中也非常重要。现有的数据、张量和流水线等并行方法无法在序列维度进行切分。当序列维度(S)增长时，训练内存开销会以 $O$($S^2$) 的速度增长。因此需要针对长序列场景进行特定的优化解决长序列场景的训练需求。

## 解决方案

支持Ring Attention长序列并行方案，以此解决序列维度扩展问题。具体细节参见原文：
> Ring Attention with Blockwise Transformers for Near-Infinite Context (https://arxiv.org/pdf/2310.01889)

### 解决思路:

Ring Attention借鉴了分块Softmax原理，在不需要获取整个序列的完整矩阵情况下进行分块attention计算。因此作者提出以分块方式执行自注意力和前馈网络计算，跨多个设备分布序列维度。具体地，该方法在进程之间构建注意力计算块的环状通信结构（Ring），每个进程具有一个切分后的本地QKV块。在计算完本地的attention后，通过向后发送和向前获取KV块，遍历进程设备环，以逐块的方式进行注意力和前馈网络计算。同时，本地的attention计算和KV块的通信理想情况下可以互相掩盖，从而消除了额外引入的通信开销。另外该方案在计算attention的过程中全程不需要数据拼接，支持的序列长度理论上可以无限拓展。

## 使用场景

当使用GPT类模型进行训练，同时数据进MoE层时实际序列长度8K以上。

不同于Ulysses方案，该方案不需要确保head_size被cp_size整除。

可兼容FlashAttention，目前已默认开启FlashAttention。

如果想要使得计算和通信可以互相掩盖，理论上需要确保每个计算块分到的序列长度$c \geq F/B$。其中F是每个device的FLOPS，B是每个device间的带宽。具体推导过程参见原文。在实践中，需要确保每个计算块分到的序列长度足够大，才能较好掩盖。


## 使用方法

| 重要参数                                      | 参数说明                                                                                                                                                         |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --context-parallel-size [int]             | 开启CP对应的数量，默认为1，根据用户需求配置。                                                                                                                                     |
| --seq-length [int]                        | 输入序列的长度。                                                                                                                                                     |
| --use-cp-send-recv-overlap                | 建议开启，开启后支持send receive overlap功能。                                                                                                                            |
| --attention-mask-type [general/causal] | 可选，设置Mask计算类型，默认是causal（倒三角）Mask计算，设置general代表全量计算。                                                                                                          |
| --context-parallel-algo megatron_cp_algo  | 长序列并行算法选项，默认项为`ulysses_cp_algo`，当设置为`megatron_cp_algo`时开启Ring Attention。                                                                                     |

## 使用效果

利用多个计算设备对输入序列进行并行切分，降低单设备的内存消耗，相比不开启序列并行单步耗时增加，相比重计算计算效率提升。


## 注意事项：

1. 开启Context Parallel时需要同时开启Flash Attention特性，否则特性不支持。
2. 在使用GPT类模型进行训练的场景下，建议attention-mask-type设置为causal。
3. 在8k的序列长度情况下，由于计算的时间缩短，cp功能分割之后的send receive的时间反而会长于计算时间，造成性能的下降，所以建议配置seq-length / context-parallel-size> 8k以获取最佳效果。具体公式参考：S/(T\*alpha) >= 1/(W\*beta)，其中，S=seq-length / context-parallel-size， T表示芯片的理论算力，alpha表示计算效率，W表示理论通信带宽，beta表示带宽利用率。

# kvallgather 长序列并行

## 特性介绍

针对 sparse flash attention、lightning indexer，lightning indexer loss ,在计算前对已切分的key和value执行allgather通信操作获得完整的key和value。

详细介绍请参考[**kvallgather_cp_algo**](https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/kvallgather-context-parallel.md)。

## 使用方法

| 重要参数                                               | 参数说明                                                    |
|----------------------------------------------------|---------------------------------------------------------|
| --context-parallel-size [int]                      | 开启CP对应的数量，默认为1，根据用户需求配置。                                |
| --context-parallel-algo <b>kvallgather_cp_algo</b> | 长序列并行算法选项，设置为`kvallgather_cp_algo`, 开启KVAllGather长序列并行。 |
| --seq-length [int]                                 | 输入序列的长度。    

## 注意事项

1. 当前仅适配了sparse flash attention，lightning indexer融合算子，lightning indexer loss融合算子。
2. 当前仅支持`attention-mask-type`为`causal`。
3. 仅支持定长padding训练场景，采用负载均衡的序列切分方式，`--seq-length`要求能被 2 * context-parallel-size整除。 


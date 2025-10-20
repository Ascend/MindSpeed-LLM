# 高维张量并行
特性介绍参考[高维张量并行](https://gitcode.com/ascend/MindSpeed/blob/master/docs/features/tensor-parallel-2d.md)
## 使用场景
集群规模较大，且设置TP域很大时，例如A3训练llama3-405B，TP=16时；
## 使用方法
### 基础参数添加
在训练脚本的参数列表中加入 --tp-2d，开启2D张量并行，--tp-x N1和--tp-y N2分别设置其x轴、y轴的切分大小，其中需满足tp = N1 * N2(N1 > 1, N2 > 1)。
例如：
```
    --tensor-model-parallel-size 16 \
    --tp-2d \
    --tp-x 8 \
    --tp-y 2 \
```
### 其他优化参数
用于辅助高维张量并行特性进行通信隐藏，需要开启tp-2d时生效：

 - --enable-overlap-ag-with-matmul: 在linear层forward计算时，开启all-gather通信和matmul进行隐藏，以便加速
 - --enable-overlap-matmul-with-rs: 在linear层forward计算时，开启matmul计算和reduce-scatter通信进行隐藏，以便加速
 - --coc-fused-kernel: 在linear层forward计算时，开启计算通信融合算子，将matmul计算与all-gather、reduce-scatter都进行算子级融合，实现进一步加速（该特性不与前两个特性兼容，依赖ATB加速库）
 - --enable-backward-overlap-ag-with-matmul: 在linear层backward计算梯度时，开启all-gather通信和matmul进行隐藏，以便加速（该特性依赖ATB加速库）
**注意** 上述3个forward计算优化参数--enable-overlap-ag-with-matmul、--enable-overlap-matmul-with-rs、--coc-fused-kernel只能同时开启1个。


## 使用约束
 - 该特性不与--sequence-parallel、--use-fused-rmsnorm特性相兼容，使用该特性需关闭；
 - 该特性暂不支持MoE类模型及其相关特性；
 - 该特性推荐场景为超大稠密模型、TP域较大场景，例如llama3-405B TP=16，较小模型、较小的TP域设置会引起性能下降，请根据实际情况调整配置；
 - 在llama3-405B TP=16进行模型训练时，建议开启2D张量并行，tp-x=8,tp-y=2。其他场景由于计算效率和通信组的划分差异，需要根据tp-x和tp-y实际调优情况进行配置，部分配置不能保证效率提升；
 - 融合算子依赖CANN 8.0.1.B020及以上版本，安装CANN-NNAL并初始化添加环节；融合算子场景当前仅支持micro-batch-size=1;

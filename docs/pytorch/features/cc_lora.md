# CCLoRA

## 问题分析

LoRA微调算法通过更新附加在冻结预训练模型权重上的低秩矩阵，实现对模型的高效微调。过程中串行执行冻结预训练权重及低秩权重的前向和反向，在分布式场景，会造成冗余通信耗时。实际预训练分支与更新分支没有依赖关系，可通过异步方式实现通算优化；此外，通过挖掘数学等价的计算方式，可覆盖各种场景，实现极致性能优化。

## 解决方案

1. 将单一流水优化为通信计算双流水线：
![Alt text](../../../sources/images/cc_lora/cc_lora.png)

2. 数学等价方式，合并通信：

   在SP&Row场景，B是普通线性层，各卡参数一致，反向对A求导可以进行如下变换：
   $$
   𝑔𝑟𝑎𝑑_𝑎=all\_𝑔ather (𝑔𝑟𝑎𝑑_𝑦∗ B ) = all\_𝑔ather(𝑔𝑟𝑎𝑑_𝑦)∗ B \\
   $$
   因为在x的梯度计算中，可以利用MC2计算时获取all\_gather(𝑔𝑟𝑎𝑑_𝑦)：
   $$
   grad_x = all\_𝑔ather (𝑔𝑟𝑎𝑑_𝑦) * X \\
   $$
   所以变换后，可以省略以下通信
   $$
   𝑔𝑟𝑎𝑑_𝑦∗ B的all\_gather
   $$
   
3. 数学等价方式，优化scale逻辑

$$
  输入：x\in\mathbb{R}^{B\times S\times H}, \quad 输出： Y = Wx + \lambda BAx = \begin{cases}
  Y=Wx+B(\lambda A)x, & \text{若 }BS<H \\
  Y=(W+B\lambda A)x, & \text{若 }BS\geq H &
  \end{cases}
$$


## 使用方法

RC2以上版本，LoRA微调场景，算法与PP、VPP、分布式优化器等场景兼容

通过设置--lora-fusion开启CCLoRA的加速，在开启了TP或SP场景，加入--use-fused-mlp启用[Fused_MLP](fused_mlp.md)算法，针对LoRA的MLP模块进一步进行计算通信掩盖提高性能。

注意:CCLoRA 与 --overlap-param-gather 特性冲突 , 不能同时使用。

## 使用效果

以下验证硬件信息为Atlas 900 A2 PODc，集群规模1x8

|       模型        | NPU  |  TP  |  PP  | SP   | 基线吞吐 | CCLoRA+吞吐 | 性能提升 |
| :---------------: | :--: | :--: | :--: | ---- | :------: | :---------: | :------: |
|  Llama2-7B-动态   |  8   |  8   |  1   | √    |   9.72   |    13.64    |  40.3%   |
|  Llama2-7B-pack   |  8   |  8   |  1   | √    |   5.97   |    6.65     |  11.5%   |
| Mixtral-7*8B-动态 |  8   |  1   |  4   | ×    |  13.97   |    20.55    |  47.1%   |
|  Llama2-70B-动态  |  8   |  1   |  4   | ×    |  13.50   |    14.75    |   9.3%   |
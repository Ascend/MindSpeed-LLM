# Seq1F1B 流水线并行

## 背景与挑战

训练大型语言模型（LLM）在很大程度上依赖于分布式训练策略，其中流水线并行性（PP）起着至关重要的作用。随着训练序列扩展到32K甚至128K令牌，当前的PP方法面临着严重的瓶颈，包括大量的流水线气泡和高显存占用，极大地阻碍了训练吞吐量和模型可扩展性。[Seq1F1B](https://arxiv.org/abs/2406.03488 'Seq1F1B Paper')一种序列级一前一后（1F1B）PP方法，专为在长序列上训练LLM而设计，具有高训练吞吐量和显存效率。它使用一种计算策略来适当地划分序列，显著减少了流水线气泡和显存占用。

## 原理

![alt text](../../../sources/images/mindspore/seq1f1b/seq1f1b_img.png)

**预热阶段-累积前向**：该阶段只进行前向计算。使用序列划分的方法，将原始序列划分成$s$个子序列，依次读入并进行前向计算。因此，Seq1F1B中每个子序列对应的前向计算时间和动态显存降低至1F1B的$1/s$。

**稳定阶段-交替前向和后向**：该阶段交替进行前向后向计算。Seq1F1B中每个子序列对应的前向和后向计算时间约为1F1B的$1/s$。当流水线并行阶段数量为$p$时，Device1在进行第一次反向之前，累积的前向数量峰值为$s+p-1$。

**冷却阶段-结束后向**：该阶段进行剩余的后向计算，结束后同步更新优化器和模型参数。

**总体来看**：Seq1F1B的收益如下

- 每个子序列长度相等的情况下，Device1的动态显存峰值约为1F1B的 $\frac{s+p-1}{sp} = \frac{1}{s}+\frac{1}{p}-\frac{1}{sp}$, 但KV缓存会增加一些额外的显存占用。

- 每个子序列前后向计算时间相同的情况下，整体空泡降约为1F1B的 $\frac{1}{s}$

因此，$s$和$p$越大，Seq1F1B的显存收益越明显。此外，我们实现了支持重计算（Recompute）特性的Seq1F1B，可以兼容现有的全量重计算和选择重计算特性，进一步降低动态显存占用。

## 使用方法

### 数据预处理
目前Seq1F1B支持多样本Pack模式的预训练和微调场景，数据处理阶段需要进行如下修改，具体细节参考[大模型分布式预训练pack模式说明文档](../../pytorch/solutions/pretrain/pretrain_eod.md)和[多样本Pack模式微调说明文档](../../pytorch/solutions/finetune/multi_sample_pack_finetune.md)：

- 多样本Pack模式预训练场景：数据预处理阶段额外添加`--append-eod`参数开启pack模式数据预处理，在每个输入序列的末尾添加一个特殊的标记来表示输入序列的结束。

- 多样本Pack模式微调场景：数据预处理阶段加入`--pack`将数据转为Pack格式，使用`--seq-length`指定Pack数据集每条数据的长度，使用`--append-eod`在每个输入序列的末尾添加一个特殊的标记来表示输入序列的结束。


### 参数详解
在 msrun 启动bash 脚本中增加如下参数来使用Seq1F1B:

- 使用参数 `--enable-seq1f1b` 使能Seq1F1B流水线并行特性。

- 使用参数 `--seq1f1b-splits [int]` 控制seq1f1b中序列拆分后子序列的数量$s$，默认使用4。

- 使用参数 `--seq1f1b-balance-method [string]` 平衡子序列的方法，可选项为['average', 'uniform_comp']，默认使用'average'。'average'方法是根据序列长度进行均匀切分，平衡每个子序列的token数目；'uniform_comp'方法是根据序列的计算量进行切分，平衡每个子序列的FLOPs。在多样本pack场景中推荐选择'average'，在单样本长序列训练场景中使用推荐使用'uniform_comp'。

- 使用参数 `--reset-attention-mask` 进行多样本pack模式预训练/微调，此时`不能使用参数--no-pad-to-seq-lengths`。

### 建议搭配特性

- 使用Seq1F1B时，建议使用`gradient-accumulation-fusion`特性，即不添加`--no-gradient-accumulation-fusion`参数。由于序列切分会导致算子下发次数增加$s$倍，因此在使用Seq1F1B时，融合算子的性能提升将更加显著。

- 在显存紧缺的场景中，建议Seq1F1B搭配重计算特性。可参考以下参数配置开启重计算，更多重计算特性说明及配置可参考[重计算相关文档](../../pytorch/features/recompute_relative.md)。

```python
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers ${N} \
```

## 使用效果分析及验证

### 性能及显存理论分析

假设每个子序列的计算时间和显存占用近似相同，可得到下表分析结果：

| PP\评价指标        | 空泡 | 动态显存 | 
| ----------| ------------ | ----------------- |
| 1F1B      | $(p-1)(F+B)$      |      $pM$          |
| Seq1F1B   | $(p-1)\frac{(F+B)}{s}$    |  $\frac{s+p-1}{s}M$+kv_cache  |

其中，$p$为流水线并行阶段数量，$s$为序列切分数量，$F$和$B$为Model Chunk进行一次前向和后向的时间，$M$为Model Chunk进行一次前向过程所累积的动态显存。

### 性能及显存实验验证

实验中seq_len=16k, $s$=4, 且均开启同样的重计算。1F1B<sup>R</sup>和Seq1F1B<sup>R</sup> 表示开启重计算特性后的1F1B和Seq1F1B。

#### Qwen3-4b 预训练

| Qwen3-4b (PP=8,TP=1)  | 性能 ms/iter | Device1显存 MB |
| ----------| ------------ | -------------| 
| 1F1B<sup>R</sup>      |  11711 (× 1.00)     |  46214  |
| Seq1F1B<sup>R</sup>  |  10257 (× 1.14)  |  26942  |

#### DeepSeek v3(裁剪版) 微调

| DeepSeek v3 单机8卡 (PP=4,TP=2)  | 性能 ms/iter | Device1显存 MB |
| ----------| ------------ | ----------------- |
| 1F1B<sup>R</sup>       | 16532 (× 1.00)      |   31122    |
| Seq1F1B<sup>R</sup>   | 15002 (× 1.10)   |   24539    |

## 注意

- 当前Seq1F1B仅适配了Qwen3/DeepSeek3模型，其他模型待验证。
- 当前Seq1F1B仅支持--micro-batch-size为1。
- 当前Seq1F1B仅支持多样本pack模式的预训练/微调场景。
- 由于序列切分会带来算子下发次数为原来的$s$倍，实验发现，seq1f1b在子序列长度不小于4k时能明显看到性能收益。
- 目前在Seq1F1B场景中，下方的特性未支持或未验证

<table>
  <thead>
    <tr>
      <th>特性</th>
      <th>未支持/未验证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> --micro-batch-size > 1</td>
      <td> 未支持 </td>
    </tr>
    <tr>
      <td> --mla-up-proj-tp-overlap </td>
      <td> 未支持 </td>
    </tr>
    <tr>
      <td> --mla-zero-memory </td>
      <td> 未支持 </td>
    </tr>
    <tr>
      <td> --context-parallel-size > 1</td>
      <td> 未支持 </td>
    </tr>
    <tr>
      <td> --mla-mm-split </td>
      <td> 未验证 </td>
    </tr>
    <tr>
      <td> --recompute-mla-up-proj </td>
      <td> 未验证 </td>
    </tr>
    <tr>
      <td> --mla-fa-divide-qk </td>
      <td> 未验证 </td>
    </tr>
  </tbody>
</table>

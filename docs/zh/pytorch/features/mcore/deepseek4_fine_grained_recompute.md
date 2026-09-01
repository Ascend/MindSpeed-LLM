# DeepSeek V4细粒度重计算

## 特性介绍

DeepSeek V4细粒度重计算通过在反向传播时重新计算CSA和MHC中的部分中间结果，减少训练过程中保存的激活值，从而降低NPU显存占用。该特性会增加一定的重计算开销，建议根据模型配置和显存情况选择开启。

CSA重计算与MHC重计算相互独立，可以单独开启，也可以同时开启。

## 使用方法

| 重要参数 | 参数说明 |
|----------|----------|
| `--recompute-csa-attention` | 开启DeepSeek V4 CSA细粒度重计算。覆盖受支持的Q投影、Indexer、Sparse Attention以及输出投影等计算；与`--recompute-norm`组合时重计算目标层的Q归一化。 |
| `--mhc-recompute` | 开启融合NPU MHC pre/post细粒度重计算。需要同时配置`--enable-mhc`和`--use-fused-mhc`。 |

仅开启CSA细粒度重计算：

```shell
--recompute-csa-attention
```

仅开启MHC细粒度重计算：

```shell
--enable-mhc \
--use-fused-mhc \
--mhc-recompute
```

同时开启两种细粒度重计算：

```shell
--recompute-csa-attention \
--enable-mhc \
--use-fused-mhc \
--mhc-recompute
```

## 注意事项

1. 当前特性仅用于DeepSeek V4的Mcore训练场景，默认关闭。
2. `--recompute-csa-attention`与`--mhc-recompute`没有参数依赖关系；同时开启时会协调中间激活值的释放时机。
3. `--mhc-recompute`仅对融合NPU MHC实现生效，因此需要同时开启`--enable-mhc`和`--use-fused-mhc`。
4. 细粒度重计算仅在训练且开启梯度计算时生效，MTP层当前不启用该重计算。
5. 开启后会以额外计算开销换取显存节省，实际收益与模型规模、序列长度和并行配置有关。

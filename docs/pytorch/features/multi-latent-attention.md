# Multi Latent Attention

## 使用场景

### 问题描述

DeepSeek系列模型创造性地提出多头潜在注意力：Multi-head Latent Attention(简称MLA），替代传统多头注意力(Multi Head Attention)。
具体而言，MLA利用低秩键值联合压缩(low-rank key-value joint compression)来降低推理时的KV Cache开销，并且模型效果不输于传统的MHA。

从MHA到MLA演化如图所示：
![](../../../sources/images/mla/image_01.png)

### 特性介绍

【--multi-latent-attention】

用于开启MLA特性，脚本里面使能该参数会将attention模块替换为MLA结构。

【--mla-fa-without-pad】

![](../../../sources/images/mla/image_04.png)

如上图所示：未开启该特性时，当query、key的维度和value的维度不匹配时，会将value维度padding到与query、key相同的维度，然后进入FA进行计算。开启该特性后，在进入FA之前不会进行pad处理，消减pad操作，减少额外显存占用，提升训练性能。

![](../../../sources/images/mla/image_02.png)

**建议CANN版本为8.2.RC1及更高版本使用，如遇到上图FA计算时shape不匹配报错，建议更新CANN包。**

【--mla-mm-split】

在对压缩后的q_compressed和kv_compressed进行升维，q_compressed经过升维后会转成q_no_pe和q_pos_emb，kv_compressed会转成k_no_pe和value，所以在进行升维操作时可以有两种方式，如下图所示：

![image](../../../sources/images/mla/image_03.png)

- 开启--mla-mm-split后，与q_compressed相乘的矩阵会被初始化为两个(linear_qk_nope,linear_qk_rope)，q_compressed与这两个矩阵相乘直接得到q_no_pe和q_pos_emb；与kv_compressed相乘的矩阵会被初始化为两个(linear_kv_nope,linear_v)，kv_compressed与这两个矩阵相乘得到k_no_pe和value，会消减两次split操作，从而避免产生非连续tensor，优化转连续开销，但是由于对一个大矩阵乘拆分了两次，会降低矩阵乘的效率，同时TP通信多场景可能带来更多通信开销。
- 不使用--mla-mm-split，与q_compressed相乘的矩阵会被初始化一个(linear_kv_up_proj)，q_compressed与这一个大矩阵相乘得到结果然后split成q_no_pe和q_pos_emb；与kv_compressed相乘的矩阵会被初始化为一个(linear_kv_up_proj),kv_compressed与这个一个大矩阵相乘后split成k_no_pe和value。与开启--mla-mm-split相比，关闭该特性会提高矩阵计算效率，但是可能会有转连续的开销。

**推荐在无TP场景或者TP通讯量较少场景使用。**

## 使用约束

【--multi-latent-attention】

如果使用MLA特性，需要在shell脚本里面指定支持MLA的spec(目前仓上支持MLA的spec有[deepseek_spec](mindspeed_llm/tasks/models/spec/deepseek_spec.py) 、[minicpm_spec](mindspeed_llm/tasks/models/spec/minicpm_spec.py))，同时shell里面添加--multi-latent-attention特性。

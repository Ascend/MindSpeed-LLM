# 权重转换

## 权重转换背景

随着模型规模从亿级向万亿级跃迁，TB级别参数模型在实际部署与迁移过程中对系统资源提出了极高的要求，单一设备无法容纳完整模型参数。MindSpeed-LLM使用了一种支持按需加载并具备内存高效性的权重转换方案，以解决大参数规模模型在转换阶段易崩溃的问题，为超大模型的高效训练与应用提供基础技术支持。


- [权重下载](#1-权重下载)

  从Huggingface等网站下载开源模型权重，支持命令行和网页下载。
- [权重转换](#2-权重转换)
  - [Huggingface权重转换到Megatron-Mcore](#21-huggingface权重转换到megatron-mcore格式)

    将Huggingface模型权重转换为Megatron-Mcore格式，支持多种并行切分。

  - [Megatron-Mcore权重转换到Huggingface格式](#22-megatron-mcore权重转换到huggingface格式)

    将Megatron-Mcore模型权重转换为Huggingface格式，适用于不同框架间的模型迁移。

  - [【调试功能】Huggingface权重减层转换到Megatron-Mcore格式](#23-调试功能huggingface权重减层转换到megatron-mcore格式)

    支持将Huggingface模型权重减层转换为Megatron-Mcore格式，支持多种并行切分。


- [大参数权重转换特性清单](#权重转换特性清单)

## 权重转换使用

权重转换旨在解决不同深度学习框架和训练策略下模型权重的兼容性问题，支持在多个模型和训练配置之间进行高效的权重互转。核心功能包括：

**权重互转**：能够在 HuggingFace、Megatron-LM主流框架之间，实现任意并行切分策略的权重格式互转。

**训练并行策略权重转换**：支持多种训练并行策略之间的权重转换，包括 张量并行(TP)、流水线并行(PP)、专家并行(EP)、专家张量并行(ETP) 和 虚拟流水并行(VPP) 等。无论是针对不同并行策略的训练，还是需要在不同策略之间切换的场景，都能实现灵活的权重转换，以适应各种训练和推理需求。



## 1. 权重下载

从Huggingface等网站下载开源模型权重

预训练权重链接在 [稠密模型](../../models/dense_model.md)和[MoE模型](../../models/moe_model.md) 章节列表的`参数`列链接中可以获取；更多社区资源可以在`模型`列链接中获取，如`Chat/Instruct`权重等。

### 下载方式

#### 方法一：网页直接下载
通过浏览器访问链接，手动下载所有权重文件。

#### 方法二：命令行下载
将权重保存到 `MindSpeed-LLM/model_from_hf` 目录，示例：


```shell
#!/bin/bash
mkdir ./model_from_hf/llama-2-7b-hf/
cd ./model_from_hf/llama-2-7b-hf/
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/generation_config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00002-of-00002.bin
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model.bin.index.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/special_tokens_map.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer.model
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/tokenizer_config.json
cd ../../
```

### 常见问题
如果下载过程中遇到问题，请参考：

HuggingFace 官方文档：https://huggingface.co/docs/hub/models-downloading

ModelScope 下载指南：https://modelscope.cn/docs/models/download

网络问题解决方案：可尝试使用国内镜像源或代理

### 注意事项
确保有足够的磁盘空间存放模型权重

检查文件完整性，下载后验证文件大小和MD5值

部分模型可能需要登录或申请权限才能下载

## 2. 权重转换

### 2.1 Huggingface权重转换到Megatron-Mcore格式

权重转换实现了 HuggingFace 权重到 Megatron-Mcore 格式的转换，支持多种并行策略（如张量并行、流水并行等），确保转换后可以在 MindSpeed-LLM 框架下继续训练和推理。

**注意**：

在做权重转换前，请先确认训练时的参数配置，根据您的训练配置修改仓上的权重转换脚本（因为这些配置会改变权重的结构，如果与训练的参数不一致的话，会导致训练无法加载权重），当前权重转换方案支持的全量参数配置见下表：
<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>需要和训练配置一致</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--load-model-type</td>
      <td>源模型类型，可选项为'hf'或'mg'，默认为 'hf'</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--save-model-type</td>
      <td>转换后模型类型，可选项为'hf'或'mg'，默认为 'mg'</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--load-dir</td>
      <td>源模型路径</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--save-dir</td>
      <td>转换后模型权重的储存路径</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--model-type-hf</td>
      <td>huggingface模型类别，默认为qwen3，对于已经支持的模型，脚本内已经配置好，使用者无需更改</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--target-tensor-parallel-size</td>
      <td>TP，指定张量并行数量，默认为 1</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--target-pipeline-parallel-size</td>
      <td>PP，指定流水线数量，默认为 1</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--target-expert-parallel-size</td>
      <td>EP，指定专家并行数量，默认为1</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--expert-tensor-parallel-size</td>
      <td>ETP，指定专家张量并行，默认为 1</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layers-per-virtual-pipeline-stage</td>
      <td>VPP划分，指定VPP的每个Stage层数，默认为None</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layer-list</td>
      <td>动态PP划分，通过列表指定每个PP Stage的层数，默认为None。使用时，列表以英文逗号隔开，列表的总和为模型总层数，并且列表的长度等于PP。e.g. 总层数为14层，指定参数--num-layer-list 3,4,4,3 \，--target-pipeline-parallel-size 4 \</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--noop-layers</td>
      <td>自定义空层操作，指定在模型某层增加空层，转换后层数为原huggingface模型层数+空层数，默认为None</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--moe-grouped-gemm</td>
      <td> MoE分组矩阵乘法优化</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--moe-tp-extend-ep</td>
      <td>TP拓展EP，开启后，专家层TP组切分专家参数</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--mla-mm-split</td>
      <td>开启后对压缩后的q_compressed和kv_compressed进行升维</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--mtp-num-layers</td>
      <td>MTP层的层数，默认为0</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--schedules-method</td>
      <td>DualPipeV流水排布，默认为None，可选项为'dualpipev'</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

场景限制：

1、模型的层数必须能被PP切分数量整除，否则需要增加空层(--noop-layer)或者使用动态PP(--num-layer-list)

2、VPP(--num-layers-per-virtual-pipeline-stage)和动态PP划分(--num-layer-list)只能二选一


下面提供一个Qwen3-235b模型的hf-mg权重转换脚本仅供参考：

```shell
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 32 \
    --num-layers-per-virtual-pipeline-stage 8 \
    --noop-layers 94,95 \
    --load-dir ./model_from_hf/qwen3_moe_hf/ \
    --save-dir ./model_weights/qwen3_moe_mcore/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe
```


【启动脚本】

MindSpeed LLM提供预制好的模型权重转换脚本，以下为Huggingface到Megatron-Mcore的权重转换脚本命名风格及启动方法，可按模型类别进行查找：

```shell
# 命名及启动：
# bash examples/mcore/model_name/ckpt_convert_xxx_hf2mcore.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_hf2mcore.sh
```


### 2.2 Megatron-Mcore权重转换到Huggingface格式

权重转换实现了 Megatron-Mcore 权重到 HuggingFace 格式的转换，支持多种并行策略（如张量并行、流水并行等）。转换过程中，模型的权重会被适配为 HuggingFace 的标准格式，确保可以在 HuggingFace 环境下继续进行训练和推理。

**注意**：

1、转到Huggingface权重**无需设置--target-tensor-parallel-size 、--target-pipeline-parallel-size、--target-expert-parallel-size、--expert-tensor-parallel-size、--num-layers-per-virtual-pipeline-stage** ，因为Huggingface权重不涉及并行切分。

2、转换成功后的权重保存目录下仅包含模型权重文件，不会生成config.json模型配置文件和tokenizer.model、vocab.json等词表文件。

3、如果Megatron-Mcore权重配置了空层，在Megatron-Mcore权重转换到Huggingface格式时，也需要在命令行加上相同的空层配置。

下面提供一个Qwen3-235b模型的mg-hf权重转换脚本仅供参考：

```shell
python convert_ckpt_v2.py \
    --load-model-type mg \
    --save-model-type hf \
    --noop-layers 94,95 \
    --load-dir ./model_weights/qwen3_moe_mcore/ \
    --save-dir ./model_from_hf/qwen3_moe_hf/ \
    --moe-grouped-gemm \
    --model-type-hf qwen3-moe
```


【启动脚本】

MindSpeed LLM提供预制好的模型权重转换脚本，以下为Megatron-Mcore到Huggingface的权重转换脚本命名风格及启动方法，可按模型类别进行查找：

```shell
# 命名及启动：
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_mcore2hf.sh
```


### 2.3 【调试功能】Huggingface权重减层转换到Megatron-Mcore格式

本框架支持Huggingface权重转换到Megatron-Mcore格式时**减层调试**，并且**无需更改模型的配置文件**，通过以下命令行参数进行减层配置。

【--num-layers】

指定的减层模型层数，不能大于原始模型层数，并且该层数**不包含MTP层**。默认值为None，**非减层情况下通过配置文件传入，无需指定该参数**。

如配置空操作层，num-layers的值应为真实层数，不包含MTP层，也不包括`--noop-layers`层数。

如果需要配合训练脚本进行减层调试，请注意此参数需要**和训练脚本保持一致**。

【--first-k-dense-replace】

指定的减层模型中moe层前的dense层数，不能大于原始模型的dense层数，默认值为None，**非减层情况下通过配置文件传入，无需指定该参数**。

如果需要配合训练脚本进行减层调试，请注意此参数需要**和训练脚本保持一致**。

【--mtp-num-layers】

MTP层的层数。默认值为 0，支持减层时配置MTP层，不能大于原始模型的MTP层数。

如需要配置MTP层，可通过命令行设置，如 `--mtp-num-layers 1 `。

如果需要配合训练脚本进行减层调试，请注意此参数需要**和训练脚本保持一致**。

## 社区贡献

欢迎社区贡献！如您在使用 MindSpeed LLM 权重转换过程中，有任何改进建议或发现了任何问题（包括但不限于功能问题、合规问题），请在Gitcode提交issue，我们将及时审视并解决。

感谢所有为项目做出贡献的社区成员！🎉

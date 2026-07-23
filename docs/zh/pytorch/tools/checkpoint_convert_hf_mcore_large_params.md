# 权重转换

## 权重转换背景

随着模型规模从亿级向万亿级跃迁，TB级别参数模型在实际部署与迁移过程中对系统资源提出了极高的要求，单一设备无法容纳完整模型参数。MindSpeed LLM使用了一种支持按需加载并具备内存高效性的权重转换方案，以解决大参数规模模型在转换阶段易崩溃的问题，为超大模型的高效训练与应用提供基础技术支持。

- [权重下载](#权重下载)

  从HuggingFace等网站下载开源模型权重，支持命令行和网页下载。

- [权重转换使用](#权重转换使用)
  - [HuggingFace权重转换至Megatron-Mcore](#huggingface权重转换至megatron-mcore格式)

    将HuggingFace模型权重转换为Megatron-Mcore格式，支持多种并行切分。

  - [Megatron-Mcore权重转换至HuggingFace格式](#megatron-mcore权重转换至huggingface格式)

    将Megatron-Mcore模型权重转换为HuggingFace格式，适用于不同框架间的模型迁移。

  - [【调试功能】HuggingFace权重减层转换至Megatron-Mcore格式](#调试功能huggingface权重减层转换至megatron-mcore格式)

    支持将HuggingFace模型权重减层转换为Megatron-Mcore格式，支持多种并行切分。

## 权重转换介绍

权重转换旨在解决不同深度学习框架和训练策略下模型权重的兼容性问题，支持在多个模型和训练配置之间进行高效的权重互转。核心功能包括：

**权重互转**：能够在HuggingFace、Megatron-LM主流框架之间实现任意并行切分策略的权重格式互转。

**训练并行策略权重转换**：支持多种训练并行策略之间的权重转换，包括张量并行（TP）、流水并行（PP）、专家并行（EP）、专家张量并行（ETP）和 虚拟流水并行（VPP）等。无论是针对不同并行策略的训练，还是需要在不同策略之间切换的场景，都能实现灵活的权重转换，以适应各种训练和推理需求。

## 权重下载

从HuggingFace等网站下载开源模型权重，训练权重的下载链接可在[模型支持列表](../models/supported_models.md) 的**下载链接**列中获取。

### 下载方式

#### 方法一：网页直接下载

通过浏览器访问链接，手动下载所有权重文件。

#### 方法二：命令行下载

将下载的权重保存至`MindSpeed-LLM/model_from_hf`目录，例如：

```shell
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

- HuggingFace官方文档：<https://huggingface.co/docs/hub/models-downloading>

- ModelScope下载指南：<https://modelscope.cn/docs/models/download>

> [!NOTE]
>
> 如果出现下载速度太慢或者无法访问下载的情况，请配置可用的代理或国内镜像源重试。

### 注意事项

- 确保有足够的磁盘空间存放模型权重

- 检查文件完整性，下载后验证文件大小和MD5值

- 部分模型可能需要登录或申请相应权限才能下载

## 权重转换使用

### HuggingFace权重转换至Megatron-Mcore格式

权重转换实现了HuggingFace权重到Megatron-Mcore格式的转换，支持多种并行策略（如张量并行、流水并行等），确保转换后可以在MindSpeed LLM框架下继续训练和推理。

> [!NOTE]
>
> 在做权重转换前，请先确认训练时的参数配置，根据您的训练配置修改仓上的权重转换脚本（因为这些配置会改变权重的结构，如果与训练的参数不一致的话，会导致训练无法加载权重），需要确认的训练配置见[表 1](#table1)。

**表 1**  训练配置参考 <a id="table1"></a>

| 参数  | 说明  | 默认值 |需要和训练配置一致 |
|-------|----------------|--------|--------|
| `--load-model-type` | 源模型类型，可选项为'hf'或'mg' | 'hf' | ❌ |
| `--save-model-type` | 转换后模型类型，可选项为'hf'或'mg' | 'mg'  | ❌ |
| `--load-dir` | 源模型路径 | None | ❌ |
| `--save-dir` | 转换后模型权重的储存路径 | None | ❌ |
| `--hf-cfg-dir` | 源HuggingFace权重配置文件目录，可选参数<br>在mg2hf转换过程中，将必要配置文件拷贝至权重保存目录，生成开箱即用的hf格式权重。 | None | ❌ |
| `--model-type-hf` | HuggingFace模型类别<br>对于已经支持的模型，脚本内已经配置好，使用者无需更改。 | 'qwen3' | ❌ |
| `--target-tensor-parallel-size` | TP，指定张量并行数量 | 1 | ✅ |
| `--target-pipeline-parallel-size` | PP，指定流水线数量 | 1 | ✅ |
| `--target-expert-parallel-size` | EP，指定专家并行数量 | 1 | ✅ |
| `--expert-tensor-parallel-size` | ETP，指定专家张量并行，当前仅支持开启后ETP=1 | None，在实际转换过程中等于TP的大小 | ✅ |
| `--num-layers-per-virtual-pipeline-stage` | VPP划分，指定VPP的每个Stage层数 | None | ✅ |
| `--num-layer-list` | 动态PP划分，通过列表指定每个PP Stage的层数<br>使用时，列表以英文逗号隔开，列表的总和为模型总层数，并且列表的长度等于PP。例如，总层数为14层，指定参数--num-layer-list 3,4,4,3 \，--target-pipeline-parallel-size 4 \。 | None | ✅ |
| `--noop-layers` | 自定义空层操作，指定在模型某层增加空层，转换后层数为原HuggingFace模型层数加上空层数 | None | ✅ |
| `--moe-grouped-gemm` | MoE分组矩阵乘法优化 | None | ✅ |
| `--moe-tp-extend-ep` | TP拓展EP，开启后，专家层TP组切分专家参数 | None | ✅ |
| `--mla-mm-split` | 开启后对压缩后的q_compressed和kv_compressed进行升维 | None | ✅ |
| `--mtp-num-layers` | MTP层的层数 | 0 | ✅ |
| `--schedules-method` | DualPipeV流水排布，可选项为'dualpipev' | None | ✅ |

#### 使用约束

- 模型的层数必须能被PP切分数量整除，否则需要增加空层`--noop-layers`或者使用动态PP`--num-layer-list`。

- VPP`--num-layers-per-virtual-pipeline-stage`和动态PP划分`--num-layer-list`只能二选一。

#### 使用示例

以下是Qwen3-235b模型的hf-mg权重转换脚本，仅供参考：

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

#### 启动脚本

MindSpeed LLM提供预置的模型权重转换脚本。以下列出HuggingFace至Megatron-Mcore权重转换脚本的命名风格及启动方法，可按模型类别进行查找：

```shell
# 脚本命名
# bash examples/mcore/model_name/ckpt_convert_xxx_hf2mcore.sh

# 启动方法
bash examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_hf2mcore.sh
```

> [!NOTE]
>
> 需在权重转换脚本中配置并行参数、权重词表路径、权重加载路径（包括词表等配置文件）以及权重保存的路径。

### Megatron-Mcore权重转换至HuggingFace格式

权重转换实现了Megatron-Mcore权重至HuggingFace格式的转换，支持多种并行策略（如张量并行、流水并行等）。转换过程中，模型的权重会被适配为HuggingFace的标准格式，确保可以在HuggingFace权重格式下继续进行训练和推理。

#### 使用约束

- 由于HuggingFace权重不涉及并行切分，转至HuggingFace权重时**无需设置--target-tensor-parallel-size、--target-pipeline-parallel-size、--target-expert-parallel-size、--num-layers-per-virtual-pipeline-stage**。

- 转换成功后的权重保存目录下仅包含模型权重文件，不会生成config.json模型配置文件和tokenizer.model、vocab.json等词表文件。可以使用`--hf-cfg-dir`参数指向原始HuggingFace模型的配置文件路径，自动将配置文件拷贝至mg2hf转出的权重保存目录。

- 如果Megatron-Mcore权重配置了空层`--noop-layers`，在将Megatron-Mcore权重转换至HuggingFace格式时，需要在命令行中添加**相同的空层配置**。

- 若原始Megatron-Mcore权重的专家张量并行度（ETP）为 1，则在执行mcore2hf转换脚本时，必须添加 **--expert-tensor-parallel-size 1** 参数。

#### 使用示例

以下是Qwen3-235b模型的mg-hf权重转换脚本，仅供参考：

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

#### 启动脚本

MindSpeed LLM提供预置的模型权重转换脚本。以下列出Megatron-Mcore至HuggingFace权重转换脚本的命名风格及启动方法，可按模型类别进行查找：

```shell
# 脚本命名
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh

# 启动方法
bash examples/mcore/qwen3_moe/ckpt_convert_qwen3_moe_235b_mcore2hf.sh
```

> [!NOTE]
>
> 需在权重转换脚本中配置并行参数、权重词表路径、权重加载路径（包括词表等配置文件）以及权重保存的路径。

### 【调试功能】HuggingFace权重减层转换至Megatron-Mcore格式

本框架支持HuggingFace权重转换到Megatron-Mcore格式时**减层调试**，并且**无需更改模型的配置文件**，通过以下命令行参数进行减层配置。

【--num-layers】

指定的减层模型层数，不能大于原始模型层数，并且该层数**不包含MTP层**。默认值为None，**非减层情况下通过配置文件传入，无需指定该参数**。

如配置空层`--noop-layers`，num-layers的值应为真实层数，不包含MTP层，也不包括`--noop-layers`层数。

如果需要配合训练脚本进行减层调试，请注意在不添加空层配置`--noop-layers`的情况下，此参数需要**和训练脚本保持一致**；在加了空层`--noop-layers`的情况下，权重转换的层数`--num-layers`为真实层数，不包含空层，而训练脚本的层数`--num-layers`为包含空层`--noop-layers`的总层数。

【--first-k-dense-replace】

指定的减层模型中MoE层前的Dense层数，不能大于原始模型的Dense层数，默认值为None，**非减层情况下通过配置文件传入，无需指定该参数**。

如果需要配合训练脚本进行减层调试，请注意此参数需要**和训练脚本保持一致**。

【--mtp-num-layers】

MTP层的层数。默认值为0，支持减层时配置MTP层，不能大于原始模型的MTP层数。

如需要配置MTP层，可通过命令行设置，如`--mtp-num-layers 1`。

如果需要配合训练脚本进行减层调试，请注意此参数需要**和训练脚本保持一致**。

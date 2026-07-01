# 权重转换

## 权重转换背景

随着大规模预训练模型的广泛应用，不同的训练框架和硬件平台之间的适配性问题日益凸显。专有训练框架（如MindSpeed LLM）通常采用定制的并行化策略（例如Tensor Parallelism、Pipeline Parallelism），以应对大规模模型训练中的内存和计算瓶颈。随着训练需求和硬件的变化，模型参数的切分策略也需相应调整。然而，跨框架的权重转换往往面临格式不兼容和切分策略差异等挑战。权重转换旨在促进大规模预训练模型在不同训练框架间的无缝迁移与评估，解决框架间权重格式不兼容及切分策略差异等问题，从而增强模型迁移的灵活性和可扩展性，以支持更广泛的应用场景和业务需求。

- [权重下载](#权重下载)

  从HuggingFace等网站下载开源模型权重，支持命令行和网页下载。

- [权重转换使用](#权重转换使用)
  - [HuggingFace权重转换至Megatron-Mcore格式](#huggingface权重转换至megatron-mcore格式)

    将HuggingFace模型权重转换为Megatron-Mcore格式，支持多种并行切分。

  - [Megatron-Mcore权重转换至HuggingFace格式](#megatron-mcore权重转换至huggingface格式)

    将Megatron-Mcore模型权重转换为HuggingFace格式，适用于不同框架间的模型迁移。

  - [LoRA权重转换](#lora权重转换)

    - [Megatron-Mcore格式权重合并](#megatron-mcore格式权重合并)

      支持将Mcore格式的LoRA微调权重与Base模型权重合并，转换为Megatron或HuggingFace格式。

    - [LoRA权重转换至HuggingFace格式](#lora权重转换至huggingface格式)

      支持将LoRA微调权重单独转为HuggingFace格式。

- [权重转换特性清单](#权重转换特性清单)

## 权重转换介绍

权重转换旨在解决不同深度学习框架和训练策略下模型权重的兼容性问题，支持在多个模型和训练配置之间进行高效的权重互转。其核心功能包括：

- **权重互转**：支持100+种模型的权重互转，能够在HuggingFace、Megatron-LM主流框架之间实现任意并行切分策略的权重格式互转。在转换过程中，用户需要通过指定参数`--use-mcore-models`将权重转换为Megatron-Mcore格式。

- **训练并行策略权重转换**：支持多种训练并行策略之间的权重转换，包括张量并行、流水线并行、专家并行、流水并行动态划分和虚拟流水并行等。无论是针对不同并行策略的训练，还是需要在不同策略之间切换的场景，都能实现灵活的权重转换，以适应各种训练和推理需求。

- **LoRA权重合并与转换**：支持将LoRA权重与Base权重合并，简化了模型推理过程中的加载步骤。合并后的模型可直接用于推理，显著提升了推理效率，减少了不必要的计算资源消耗。同时支持将LoRA微调权重单独转为HuggingFace格式，以支持客户下游任务需求。

## 权重下载

从HuggingFace等网站下载开源模型权重，训练权重的下载链接可在[模型支持列表](../models/supported_models.md)的**下载链接**列中获取。

权重可以通过网页直接下载，也可以通过命令行下载，并将下载的权重保存至"MindSpeed-LLM/model_from_hf"目录，例如：

```shell
#!/bin/bash
mkdir -p ./model_from_hf/llama-2-7b-hf/
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

## 权重转换使用

### HuggingFace权重转换至Megatron-Mcore格式

权重转换实现了HuggingFace权重到Megatron-Mcore格式的转换，支持多种并行策略（如张量并行、流水并行等），确保转换后可以在MindSpeed LLM框架下继续训练和推理。

> [!NOTE]
>
> 在做权重转换前，请先确认训练时的参数配置，根据您的训练配置修改仓上的权重转换脚本（因为这些配置会改变权重的结构，如果与训练的参数不一致的话，会导致训练无法加载权重），需要确认的训练配置见[表 1](#table1)。

**表 1**  训练配置参考 <a id="table1"></a>

| 参数  | 说明  | 默认值 | 是否必选 |
|-------|----------------|--------|--------|
| `--target-tensor-parallel-size` | TP切分数量 | 1 | 是 |
| `--target-pipeline-parallel-size` | PP切分数量  | 1 | 是 |
| `--num-layer-list` | 动态PP划分，通过列表指定每个PP Stage的层数 | None | 否 |
| `--num-layers-per-virtual-pipeline-stage` | VPP划分，指定VPP的每个Stage层数 | None | 否 |
| `--target-expert-parallel-size` | 专家并行，指定专家并行卡数  | 1 | 否 |
| `--noop-layers` | 自定义空层操作，指定在模型某层增加空层，转换后层数为原HuggingFace模型层数加上空层数 | None | 否 |
| `--use-mcore-models` | 转换为Megatron-Mcore权重 | None | 是 |
| `--model-type-hf` | HuggingFace模型类别 | llama2 | 否 |
| `--tokenizer-model` | 需要指明到具体的分词器模型文件<br>包括tokenizer.model、tokenizer.json、qwen.tiktoken、None等，具体取决于HuggingFace中词表文件的格式形式 | None | 是 |
| `--params-dtype` | 指定权重转换后的权重精度模式<br>如果源格式文件为bf16，则需要对应设置为bf16，否则影响推理或评估结果  | fp16 | 是 |

#### 使用约束

- 模型的层数必须能被PP切分数量整除，否则需要增加空层`--noop-layers`或者使用动态PP`--num-layer-list`。

- VPP和动态PP划分只能二选一。

- 请参见[model_cfg.json](../../../../configs/checkpoint/model_cfg.json)中的"model_mappings"部分，查看目前支持的模型。

#### 使用示例

以下是Llama2-7b模型的hf-mg权重转换脚本，仅供参考：

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --num-layer-list 8,8,8,8 \
    --model-type-hf llama2 \
    --use-mcore-models \
    --load-dir ./model_from_hf/llama-2-7b-hf/ \
    --save-dir ./model_weights/llama-2-7b-mcore/ \
    --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model
```

#### 启动脚本

基于MindSpeed LLM，以下是HuggingFace至Megatron-Mcore权重转换脚本命名风格及启动方法。

```shell
# 脚本命名
# bash examples/mcore/model_name/ckpt_convert_xxx_hf2mcore.sh

# 启动方法
bash examples/mcore/llama2/ckpt_convert_llama2_hf2mcore.sh
```

> [!NOTE]
>
> 需在权重转换脚本中配置并行参数、权重词表路径、权重加载路径（包括词表等配置文件）以及权重保存的路径。

### Megatron-Mcore权重转换至HuggingFace格式

权重转换实现了Megatron-Mcore权重至HuggingFace格式的转换，支持多种并行策略（如张量并行、流水并行等）。转换过程中，模型的权重会被适配为HuggingFace的标准格式，确保可以在HuggingFace环境下继续进行训练和推理。

#### 使用约束

- 由于HuggingFace权重不涉及并行切分，转换至HuggingFace权重时必须设置`--target-tensor-parallel-size = 1`和`--target-pipeline-parallel-size = 1`。
- 转换成功后的权重保存目录中仅包含模型权重文件，不会生成config.json模型配置文件和tokenizer.model、vocab.json等词表文件。
- `--save-dir`必须配置为原始HuggingFace模型路径，且该路径下需包含完整的HuggingFace模型文件，包括权重和配置文件。
- 如果Megatron-Mcore权重配置了空层`--noop-layers`，在将Megatron-Mcore权重转换至HuggingFace格式时，需要在命令行中添加**相同的空层配置**，并加参数`--load-checkpoint-loosely`。

#### 使用示例

以下是Llama2-7b模型的mg-hf权重转换脚本，仅供参考：

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --model-type-hf llama2 \
    --use-mcore-models \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/
```

> [!NOTE]
>
> - 配置`--save-dir`参数时，需设置原始HuggingFace模型路径，新权重将保存至./model_from_hf/llama-2-7b-hf/mg2hf。
> - 更多参数详情请参见[表 1](#table1)。

#### 启动脚本

基于MindSpeed LLM，以下是Megatron-Mcore至HuggingFace的权重转换脚本命名风格及启动方法。

```shell
# 脚本命名
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh

# 启动方法
bash examples/mcore/llama2/ckpt_convert_llama2_mcore2hf.sh
```

> [!NOTE]
>
> 需在权重转换脚本中配置并行参数、权重词表路径、权重加载路径（包括词表等配置文件）以及权重保存的路径。

### LoRA权重转换

当前MindSpeed LLM支持以下两种LoRA权重转换方法:

- 将LoRA微调权重与基础模型权重合并，转换为Megatron或HuggingFace格式。

- 将LoRA微调权重单独转为HuggingFace格式，在LoRA微调脚本中加入参数`--lora-ckpt-filter`仅保存LoRA权重。

#### Megatron-Mcore格式权重合并

在权重转换命令中，加入如下参数可将训练的LoRA权重与权重转换出的Base权重进行融合。

```shell
--lora-load ./ckpt/llama-2-7b-lora  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
```

**表 2**  权重参数说明

| 参数 | 说明 | 是否必选 |
|------|--------|------|
| `--lora-load` | 加载LoRA微调后生成的权重 | 否 |
| `--lora-r` | LoRA中的秩（rank），它决定了低秩矩阵的大小 | 否 |
| `--lora-alpha` | 定义了LoRA适应的学习率缩放因子，该参数将影响低秩矩阵的更新速度 | 否 |
| `--lora-target-modules` | 定义LoRA目标模块，该参数为一个由空格分隔的字符串列表，且不具有默认值<br>每个字符串对应需要进行LoRA微调的层名称，且只能在上述四种预定义的参数配置中选择。用户可根据具体需求调整该参数。 | 否 |

##### 合并后转换为Megatron-Mcore权重

下面提供Megatron-Mcore格式的Llama2-7b模型的LoRA权重与Base权重合并，并转为Megatron-Mcore格式的示例脚本，仅供参考：

```shell
python convert_ckpt.py \
    --model-type GPT \
    --use-mcore-models \
    --load-model-type mg \
    --save-model-type mg \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_weights/llama-2-7b-lora2mcore
```

以下是权重转换脚本启动方法。

```shell
# 启动方法（以llama2为例）
bash examples/mcore/llama2/ckpt_convert_llama2_mg2mg_lora.sh
```

##### 合并后转换为HuggingFace权重

下面提供Megatron-Mcore格式的Llama2-7b模型的LoRA权重与Base权重合并，并转为HuggingFace格式的示例脚本，仅供参考：

```shell
python convert_ckpt.py \
    --model-type GPT \
    --use-mcore-models \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/
```

> [!NOTE]
>
> 配置`--save-dir`参数时，需设置原始HuggingFace模型路径，新权重将保存至./model_from_hf/llama-2-7b-hf/mg2hf。

以下是权重转换脚本启动方法。

```shell
# 启动方法（以llama2为例）
bash examples/mcore/llama2/ckpt_convert_llama2_mcore2hf_lora.sh
```

##### 注意

- LoRA参数值需与LoRA微调时的参数保持一致，LoRA权重的切分方式需与Base权重的切分方式保持一致。

- 由于调用peft库合并LoRA权重后，权重数据类型为float16，但是部分模型如Qwen系列模型，默认数据类型为bfloat16，合并后的权重转回HuggingFace格式会有精度损失问题。可以将原始HuggingFace模型的config.json中的数据类型改为float16暂时规避。

- MoE模型暂不支持开启`--moe-grouped-gemm`特性后的LoRA权重转换。

#### LoRA权重转换至HuggingFace格式

通过启用参数`--save-lora-to-hf`，支持将LoRA微调后的LoRA权重转换至HuggingFace格式。

##### 使用示例

以下是Llama2-7b模型的LoRA权重转至HuggingFace格式的示例脚本，仅供参考：

```shell
python convert_ckpt.py \
    --model-type GPT \
    --use-mcore-models \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./ckpt/llama2_lora_filter \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-checkpoint-loosely \
    --save-lora-to-hf \
    --save-dir ./model_from_hf/llama-2-7b-hf/
```

> [!NOTE]
>
> 配置`--save-dir`参数时，需设置原始HuggingFace模型路径，新权重将保存至./model_from_hf/llama-2-7b-hf/mg2hf。

**表 3**  权重参数说明

| 参数 | 说明 | 是否必选 |
|------|--------|------|
| `--save-lora-to-hf` | LoRA转HuggingFace时，设置该参数以指定仅转换LoRA权重 | 否 |
| `--load-checkpoint-loosely` | 允许松弛加载，转换LoRA权重时，需要设置该参数 | 否 |

##### 使用约束

- 原始权重仅为LoRA权重，不包含Base权重，需要在LoRA微调脚本中加入参数`--lora-ckpt-filter`仅保存LoRA权重。

- `--save-lora-to-hf`和`--moe-grouped-gemm`两个参数不能同时使用，在LoRA微调时，脚本中不能加入`--moe-grouped-gemm参数`。

- `--save-lora-to-hf`和`--load-hf-from-config`两个参数不能同时使用。

- LoRA权重转换仅支持Mcore格式；仅支持fc_type为gate_up_down的模型，其余待适配；当前仅支持llama2、mixtral。

##### 启动脚本

基于MindSpeed-LLM，以下是LoRA至HuggingFace的权重转换脚本命名风格及启动方法。

```shell
# 脚本命名
# bash examples/mcore/model_name/ckpt_convert_xxx_lora2hf.sh

# 启动方法
bash examples/mcore/llama2/ckpt_convert_llama2_lora2hf.sh
```

> [!NOTE]
>
> 需在权重转换脚本中配置并行参数、权重词表路径、权重加载路径（包括词表等配置文件）以及权重保存的路径。

## 权重转换特性清单

MindSpeed LLM支持HuggingFace和Megatron-Mcore间的权重格式互转，具体功能列表如下:

<table>
  <thead>
    <tr>
      <th>源格式</th>
      <th>目标格式</th>
      <th>支持特性</th>
      <th>特性入参</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7">HuggingFace </td>
      <td rowspan="7">Megatron-Mcore</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--target-expert-parallel-size</td>
    </tr>
    <tr>
      <td>专家张量并行</td>
      <td>--expert-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>自定义空操作层</td>
      <td>--noop-layers</td>
    </tr>
    <tr>
      <td rowspan="18">Megatron-Mcore </td>
      <td rowspan="8">HuggingFace</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--target-expert-parallel-size</td>
    </tr>
    <tr>
      <td>专家张量并行</td>
      <td>--expert-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>LoRA训练模块</td>
      <td>--lora-target-modules</td>
    </tr>
    <tr>
      <td>LoRA权重</td>
      <td>--lora-load</td>
    </tr>
    <tr>
      <td>LoRA r</td>
      <td>--lora-r</td>
    </tr>
    <tr>
      <td>LoRA alpha</td>
      <td>--lora-alpha</td>
    </tr>
    <tr>
      <td rowspan="10">Megatron-Mcore</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--target-expert-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--num-layer-list</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td>LoRA训练模块</td>
      <td>--lora-target-modules</td>
    </tr>
    <tr>
      <td>LoRA权重</td>
      <td>--lora-load</td>
    </tr>
    <tr>
      <td>LoRA r</td>
      <td>--lora-r</td>
    </tr>
    <tr>
      <td>LoRA alpha</td>
      <td>--lora-alpha</td>
    </tr>
    <tr>
      <td>自定义空操作层</td>
      <td>--noop-layers</td>
    </tr>
  </tbody>
</table>

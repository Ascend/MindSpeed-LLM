# 权重转换

## 权重转换背景

随着大规模预训练模型的广泛应用，不同的训练框架和硬件平台之间的适配性问题逐渐显现。专有训练框架如MindSpeed-LLM通常采用定制的并行化策略（例如Tensor Parallelism、Pipeline Parallelism）以应对大规模模型训练中的内存和计算瓶颈。随着训练需求和硬件的变化，模型参数的切分策略也需进行相应的调整。然而，跨框架的权重转换往往面临格式不兼容和切分策略不同等挑战。权重转换旨在促进大规模预训练模型在不同训练框架之间的无缝迁移与评估，解决框架间权重格式不兼容及切分策略差异等问题，从而增强模型迁移的灵活性和可扩展性,支持更广泛的应用场景和业务需求。


- [权重下载](#1-权重下载)

  从Huggingface等网站下载开源模型权重，支持命令行和网页下载。
- [权重转换](#2-权重转换)
  - [Huggingface权重转换到Megatron-LM格式](#21-huggingface权重转换到megatron-lm格式)

    将Huggingface模型权重转换为Megatron-LM格式，支持多种并行切分。

  - [Megatron-LM权重转换到Huggingface格式](#22-megatron-lm权重转换到huggingface格式)

    将Megatron-LM模型权重转换为Huggingface格式，适用于不同框架间的模型迁移。

  - [Megatron-LM格式权重互转](#23-megatron-lm格式权重互转)

    支持Megatron-Legacy与Megatron-Mcore格式之间的相互转换。

  - [Lora权重转换](#24-lora权重转换)

    - [legacy格式权重合并](#241-megatron-legacy格式权重合并)

      支持将legacy格式的Lora微调权重与基础模型权重合并，转换为Megatron或Huggingface格式；

    - [mcore格式权重合并](#242-megatron-mcore格式权重合并)

      支持将mcore格式的Lora微调权重与基础模型权重合并，转换为Megatron或Huggingface格式；    
    
    - [lora权重转换为Huggingface格式](#243-lora权重转换为huggingface权重)
      
      支持将Lora微调权重单独转为Huggingface格式。

  - [优化器权重转换](#25-优化器权重转换)

    优化器权重转换支持多种并行切分，确保训练过程中优化器状态的迁移。

- [权重转换特性清单](#权重转换特性清单)

## 权重转换使用

权重转换旨在解决不同深度学习框架和训练策略下模型权重的兼容性问题，支持在多个模型和训练配置之间进行高效的权重互转。核心功能包括：

**权重互转**：支持100+种模型的权重互转，能够在 Hugging Face、Megatron-LM主流框架之间，实现任意并行切分策略的权重格式互转。Megatron-LM具体包括两种格式：Megatron-Legacy和Megatron-Mcore。在转换过程中，用户可以通过指定参数 --use-mcore-models 来将权重转换为 Megatron-Mcore 格式；若未指定该参数，则默认转换为 Megatron-Legacy 格式。

**训练并行策略权重转换**：支持多种训练并行策略之间的权重转换，包括 张量并行、流水线并行、专家并行、流水并行动态划分 和 虚拟流水并行 等。无论是针对不同并行策略的训练，还是需要在不同策略之间切换的场景，都能实现灵活的权重转换，以适应各种训练和推理需求。

**Lora权重合并与转换**：支持将 Lora 权重与 Base 权重合并，简化了模型推理过程中的加载步骤。合并后的模型可直接用于推理，显著提升了推理效率，减少了不必要的计算资源消耗。支持将Lora微调权重单独转为Huggingface格式，以支持客户下游任务。

**优化器权重转换**：支持多种并行切分策略，确保优化器状态在不同并行策略间的迁移与兼容，便于在不同训练环境下进行优化器状态恢复。

## 1. 权重下载

从Huggingface等网站下载开源模型权重

预训练权重链接在 [稠密模型](../models/dense_model.md)和[MoE模型](../models/moe_model.md) 章节列表的`参数`列链接中可以获取；更多社区资源可以在`模型`列链接中获取，如`Chat/Instruct`权重等。

权重可以基于网页直接下载，也可以基于命令行下载，保存到MindSpeed-LLM/model_from_hf目录，比如：


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

## 2. 权重转换

### 2.1 Huggingface权重转换到Megatron-LM格式

权重转换实现了 HuggingFace 权重到 Megatron-LM 格式的转换，支持多种并行策略（如张量并行、流水并行等），确保转换后可以在 MindSpeed-LLM 框架下继续训练和推理。下面提供一个Llama2-7b模型的hf-mg权重转换脚本仅供参考：

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
<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>可选/必选</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--target-tensor-parallel-size</td>
      <td>TP 切分数量，默认为 1</td>
      <td>必选</td>
    </tr>
    <tr>
      <td>--target-pipeline-parallel-size</td>
      <td>PP 切分数量，默认为 1</td>
      <td>必选</td>
    </tr>
    <tr>
      <td>--num-layer-list</td>
      <td>动态PP划分，通过列表指定每个PP Stage的层数，默认为None</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--num-layers-per-virtual-pipeline-stage</td>
      <td>VPP划分，指定VPP的每个Stage层数，默认为None</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--target-expert-model-parallel-size</td>
      <td>专家并行，指定专家并行卡数，默认为1</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--noop-layers</td>
      <td>自定义空层操作，指定在模型某层增加空层，转换后层数为原huggingface模型层数+空层数，默认为None</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--use-mcore-models</td>
      <td>转换为Megatron-Mcore权重，若不指定，则默认转换为Megatron-Legacy权重</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--model-type-hf</td>
      <td>huggingface模型类别，默认为llama2</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--tokenizer-model</td>
      <td>需要指明到具体的分词器模型文件，如 tokenizer.model、tokenizer.json、qwen.tiktoken、None等，具体取决于huggingface中词表文件的格式形式</td>
      <td>必选</td>
    </tr>
    <tr>
      <td>--params-dtype</td>
      <td>指定权重转换后的权重精度模式，默认为fp16，如果源格式文件为bf16，则需要对应设置为bf16，影响推理或评估结果</td>
      <td>必选</td>
    </tr>
  </tbody>
</table>


**注意**：

1、VPP和动态PP划分只能二选一

2、目前支持的模型见 [model_cfg.json](https://gitee.com/ascend/MindSpeed-LLM/blob/master/configs/checkpoint/model_cfg.json)中“model_mappings”下包含的模型。


【启动脚本】

MindSpeed-LLM Huggingface到Megatron-Legacy权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/legacy/model_name/ckpt_convert_xxx_hf2legacy.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/legacy/llama2/ckpt_convert_llama2_hf2legacy.sh
```

MindSpeed-LLM Huggingface到Megatron-Mcore权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/mcore/model_name/ckpt_convert_xxx_hf2mcore.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/mcore/llama2/ckpt_convert_llama2_hf2mcore.sh
```

### 2.2 Megatron-LM权重转换到Huggingface格式

权重转换实现了 Megatron-LM 权重到 HuggingFace 格式的转换，支持多种并行策略（如张量并行、流水并行等）。转换过程中，模型的权重会被适配为 HuggingFace 的标准格式，确保可以在 HuggingFace 环境下继续进行训练和推理。下面提供一个Llama2-7b模型的mg-hf权重转换脚本仅供参考：

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
    --save-dir ./model_from_hf/llama-2-7b-hf/  # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hf/
```

参数意义参考2.1

**注意：** 转到Huggingface权重必须设置--target-tensor-parallel-size = 1、--target-pipeline-parallel-size = 1。

【启动脚本】

MindSpeed-LLM Megatron-Legacy到Huggingface的权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/legacy/model_name/ckpt_convert_xxx_legacy2hf.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/legacy/llama2/ckpt_convert_llama2_legacy2hf.sh
```

MindSpeed-LLM Megatron-Mcore到Huggingface的权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/mcore/model_name/ckpt_convert_xxx_mcore2hf.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/mcore/llama2/ckpt_convert_llama2_mcore2hf.sh
```

### 2.3 Megatron-LM格式权重互转

Megatron-LM格式权重互转功能支持**Megatron-Legacy**与**Megatron-Mcore**之间的相互转换，以适应不同的训练和推理需求。下面提供Llama2-7b模型的Megatron-LM格式权重互转脚本仅供参考：

```shell
# legacy转legacy
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --save-dir ./model_weights/llama-2-7b-legacy_tp2pp2/

# legacy转mcore
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --load-from-legacy \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --save-dir ./model_weights/llama-2-7b-mcore_tp2pp2/

# mcore转mocre
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --save-dir ./model_weights/llama-2-7b-mcore_tp2pp2/

# mcore转legacy
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --use-mcore-models \
    --save-to-legacy \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --save-dir ./model_weights/llama-2-7b-legacy_tp2pp2/
```


<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>可选/必选</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--load-from-legacy</td>
      <td>legacy转mcore时设置此参数以指定导入权重格式为legacy</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--save-to-legacy</td>
      <td>mcore转legacy时设置此参数以指定保存权重格式为legacy</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--noop-layers</td>
      <td>自定义空层操作，权重转换当前只支持mcore-mcore、hf-mcore、优化器权重转换，在mcore-mcore时，空层参数不能更改，只能更改并行方式，同时需要设置参数--load-checkpoint-loosely</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--load-checkpoint-loosely</td>
      <td>允许松弛加载，在带有空层的mcore-mcore权重转换时，需要设置此参数</td>
      <td>可选</td>
    </tr>
  </tbody>
</table>


其余参数意义参考2.1

注：上述权重legacy和mcore互转为高阶功能，MindSpeed-LLM基于llama2提供基础能力，并进行版本迭代看护.对于其他模型的支持，用户需根据实际需求自行进行修改与适配。

### 2.4 lora权重转换

当前仓库支持以下两种lora权重转换方法:

(1) 将Lora微调权重与基础模型权重合并，转换为Megatron或Huggingface格式 ; 

(2) 将Lora微调权重单独转为Huggingface格式，在lora微调脚本中加入参数--lora-ckpt-filter仅保存lora权重。

#### 2.4.1 Megatron-Legacy格式权重合并

在权重转换命令中，加入如下参数可以将训练的Megatron-Legacy格式的lora权重与base进行合并。

```bash
--lora-load ./ckpt/llama-2-7b-lora  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
```

<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>可选/必选</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--lora-load</td>
      <td>加载 lora 微调后生成的权重</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--lora-r</td>
      <td>LoRA中的秩（rank），它决定了低秩矩阵的大小</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--lora-alpha</td>
      <td>定义了LoRA适应的学习率缩放因子。这个参数影响了低秩矩阵的更新速度</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--lora-target-modules</td>
      <td>该参数定义了LoRA目标模块，为一个由空格分隔的字符串列表，且不具有默认值。每个字符串对应需要进行LoRA微调的层名称，且只能在上述四种预定义的参数配置中选择。用户可根据具体需求调整该参数，但需注意，legacy模型和mcore模型在该参数的取值上存在差异。</td>
      <td>可选</td>
    </tr>
  </tbody>
</table>

【合并后转换为Megatron-Legacy权重】

下面提供Megatron-Legacy格式的Llama2-7b模型的Lora权重与base权重合并，并转为Megatron-Legacy格式的示例脚本，仅供参考：

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_weights/llama-2-7b-lora2legacy
```


转换脚本命名风格及启动方法为：

```shell
# 命令启动方式以 legacy 下的模型 llama2 为例
bash examples/legacy/llama2/ckpt_convert_llama2_legacy2legacy_lora.sh
```

【合并后转换为Huggingface权重】

下面提供Megatron-Legacy格式的Llama2-7b模型的Lora权重与base权重合并，并转为Huggingface格式的示例脚本，仅供参考：

```shell
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/llama-2-7b-legacy/ \
    --lora-load ./ckpt/llama-2-7b-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hg/
```

转换脚本命名风格及启动方法为：
```shell
#命令启动方式以 legacy 下的模型 llama2 为例
bash examples/legacy/llama2/ckpt_convert_llama2_legacy2hf_lora.sh
```

#### 2.4.2 Megatron-Mcore格式权重合并

在上述权重转换命令中，加入如下参数可以将训练的lora权重与权重转换出的base权重进行融合。

```bash
--lora-load ./ckpt/llama-2-7b-lora  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
```

【合并后转换为Megatron-Mcore权重】

下面提供Megatron-Mcore格式的Llama2-7b模型的Lora权重与base权重合并，并转为Megatron-Mcore格式的示例脚本，仅供参考：

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

<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>可选/必选</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--lora-target-modules</td>
      <td>该参数定义了LoRA目标模块，为一个由空格分隔的字符串列表，且不具有默认值。每个字符串对应需要进行LoRA微调的层名称，且只能在上述四种预定义的参数配置中选择。用户可根据具体需求调整该参数，但需注意，legacy模型和mcore模型在该参数的取值上存在差异。</td>
      <td>可选</td>
    </tr>
  </tbody>
</table>

其余参数含义同2.4.1

转换脚本命名风格及启动方法为：

```shell
#命令启动方式以 mcore 下的模型 llama2 为例
bash examples/mcore/llama2/ckpt_convert_llama2_mg2mg_lora.sh
```

【合并后转换为Huggingface权重】

下面提供Megatron-Mcore格式的Llama2-7b模型的Lora权重与base权重合并，并转为Huggingface格式的示例脚本，仅供参考：

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
    --save-dir ./model_from_hf/llama-2-7b-hf/    # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hg/
```

转换脚本命名风格及启动方法为：
```shell
#命令启动方式以 mcore 下的模型 llama2 为例
bash examples/mcore/llama2/ckpt_convert_llama2_mcore2hf_lora.sh
```

**注意：** 

lora参数值需与lora微调时的参数保持一致,且lora权重的切分方式需与base权重的切分方式保持一致。

由于调用peft库合并lora权重后，权重数据类型为float16，但是部分模型如qwen系列模型，默认数据类型为bfloat16，合并后的权重转回hf格式会有精度损失问题。可以将原始HF模型的config.json中的数据类型改为float16暂时规避。

#### 2.4.3 Lora权重转换为Huggingface权重

通过使能参数--save-lora-to-hf,支持将Lora微调后的lora权重转换为Huggingface格式，下面提供Llama2-7b模型的Lora权重转为Huggingface格式的示例脚本，仅供参考：

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
    --save-dir ./model_from_hf/llama-2-7b-hf/  # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama-2-7b-hf/mg2hf/
```

<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>可选/必选</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--save-lora-to-hf</td>
      <td>lora转hf时设置此参数以指定仅转换lora权重</td>
      <td>可选</td>
    </tr>
    <tr>
      <td>--load-checkpoint-loosely</td>
      <td>允许松弛加载，转换lora权重时，需要设置此参数</td>
      <td>可选</td>
    </tr>
  </tbody>
</table>

**注意：** 

原始权重仅为lora权重，不包含base权重，需要在lora微调脚本中加入参数--lora-ckpt-filter仅保存lora权重；

--save-lora-to-hf和--moe-grouped-gemm两个参数不能同时使用,在lora微调时,脚本中不能加入--moe-grouped-gemm参数;

--save-lora-to-hf和--load-hf-from-config两个参数不能同时使用；

lora权重转换仅支持mcore格式，legacy暂未支持；仅支持fc_type为gate_up_down的模型，其余待适配；当前仅支持llama2、mixtral。

【启动脚本】

MindSpeed-LLM lora到Huggingface的权重转换脚本命名风格及启动方法为：

```shell
# 命名及启动：
# bash examples/mcore/model_name/ckpt_convert_xxx_lora2hf.sh
# 需要配置并行参数以及权重词表加载保存等路径

bash examples/mcore/llama2/ckpt_convert_llama2_lora2hf.sh
```

### 2.5 优化器权重转换

在权重转换脚本中指定`--load-model-type`参数为`optim` , 则为优化器权重转换。

使用方法：

1.准备预训练权重

优化器状态为预训练保存得到，并且需要在预训练脚本中加入参数`--use-distributed-optimizer` 表示使用分布式优化器，并且删除参数`--no-save-optim ` 使训练生成的每个权重文件夹都包括`model_optim_rng.pt`和`distrib_optim.pt` 模型权重文件和优化器状态文件。

2.`mg-mg`权重转换

优化器权重需要先做一次`mg-mg`的权重转换，并指定所需的切分方式，脚本参考2.3中`mcore-mcore`脚本：

```shell
    python convert_ckpt.py \
        --model-type GPT \
        --load-model-type mg \
        --save-model-type mg \
        --target-tensor-parallel-size 4 \
        --target-pipeline-parallel-size 2 \
        --load-dir ./ckpt/llama2-7b-tp2pp4 \
        --save-dir ./ckpt/llama2-7b-tp4pp2-optim \
        --use-mcore-models
```

在此步骤完成后，`--save-dir` 中应该会生成 `model_optim_rng.pt` 格式的权重文件。

3.权重转换优化器:

完成上述步骤后，可以执行优化器权重转换。此时，指定 `--load-model-type optim` 参数来加载优化器权重，并进行转换，下面提供Llama2-7b模型的优化器权重转换的示例脚本，仅供参考：

**注意：** ：并行配置如：TP、PP、EP、VPP、num-layer-list、noop-layers等参数需要与mcore-mcore权重转换脚本相同。


```
python convert_ckpt.py
--model-type GPT
--load-model-type optim
--load-dir ./ckpt/llama2-7b-tp2pp4
--target-tensor-parallel-size 4
--target-pipeline-parallel-size 2
--save-dir ./ckpt/llama2-7b-tp4pp2-optim
--use-mcore-models
--model-type-hf llama2
```


<table>
  <thead>
    <tr>
      <th>参数</th>
      <th>说明</th>
      <th>可选/必选</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>--save-dir</td>
      <td>权重保存路径，需要与`mg-mg`转换时的保存路径一致</td>
      <td>必选</td>
    </tr>
    <tr>
      <td>--load-model-type</td>
      <td>指定加载模型的方式。对于优化器权重转换，必须设置参数值为`optim`</td>
      <td>必选</td>
    </tr>
    <tr>
      <td>--moe-grouped-gemm</td>
      <td>对于moe模型，如果在预训练脚本和mcore-mcore权重转换脚本中加入此参数`--moe-grouped-gemm` ，则在优化器权重转换脚本中，也需要加入该参数。</td>
      <td>可选</td>
    </tr>
  </tbody>
</table>

转换脚本命名风格及启动方法为：

```shell
# 命令启动方式以 mcore 下的模型llama2为例子
bash examples/mcore/llama2/ckpt_convert_llama2_optim.sh
```

**注意**：

优化器权重转换当前仅支持模型：llama2-7b、deepseek2-lite

llama2-7b支持TP、PP、EP、VPP、DPP、noop-layers；

deepseek2-lite支持PP、EP、DPP、noop-layers。


### 权重转换特性清单

MindSpeed-LLM 支持 Huggingface、Megatron-Legacy 以及 Megatron-Core 之间的权重格式互转，具体功能列表如下:

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
      <td rowspan="10">HuggingFace </td>
      <td rowspan="4">Megatron-Legacy</td>
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
      <td rowspan="6">Megatron-Core</td>
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
      <td>--target-expert-model-parallel-size</td>
    </tr>
    <tr>
      <td>自定义空操作层</td>
      <td>--noop-layers</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="16">Megatron-Legacy </td>
      <td rowspan="6">Huggingface</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
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
      <td rowspan="4">Megatron-Core</td>
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
      <td rowspan="6">Megatron-Legacy</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
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
  </tbody>
  <tbody>
    <tr>
      <td rowspan="26">Megatron-Core </td>
      <td rowspan="6">Huggingface</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
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
      <td rowspan="4">Megatron-Legacy</td>
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
      <td rowspan="10">Megatron-Core</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--target-expert-model-parallel-size</td>
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
    <tr>
      <td rowspan="6">distributed-optimizer</td>
      <td>张量并行</td>
      <td>--target-tensor-parallel-size</td>
    </tr>
    <tr>
      <td>流水并行</td>
      <td>--target-pipeline-parallel-size</td>
    </tr>
    <tr>
      <td>专家并行</td>
      <td>--target-expert-model-parallel-size</td>
    </tr>
    <tr>
      <td>虚拟流水并行</td>
      <td>--num-layers-per-virtual-pipeline-stage</td>
    </tr>
    <tr>
      <td>流水并行动态划分</td>
      <td>--target-expert-model-parallel-size</td>
    </tr>
    <tr>
      <td>自定义空操作层</td>
      <td>--noop-layers</td>
    </tr>
  </tbody>
</table>
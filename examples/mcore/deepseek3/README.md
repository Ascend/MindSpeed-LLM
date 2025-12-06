# DeepSeek-V3模型训练

若需使用DeepSeek-V3训练，先根据预训练/微调任务查看使用指导文档示例，配置<a href="../../mcore/deepseek3/">deepseekv3训练脚本</a>，然后执行训练过程。

## 1. 预训练使用指导

<a href="../../../docs/pytorch/solutions/pretrain/pretrain.md">预训练指导文档</a>

预训练过程脚本使用顺序为：
```shell
a. data_convert_deepseek3_pretrain.sh

b. pretrain_deepseek3_671b_4k_xxx.sh (根据集群选择A2或者A3脚本)
```

## 2. 微调训练使用指导

<a href="../../../docs/pytorch/solutions/finetune/instruction_finetune.md">微调训练指导文档</a>

全参微调过程脚本使用顺序为：

```shell
a. data_convert_deepseek3_instruction.sh

b. ckpt_convert_deepseek3_hf2mcore.sh

注意：此脚本是示例，需根据c.tune_deepseek3_671b_4k_full_xxx.sh脚本设置权重转换参数。参数说明查看以下的"DeepSeek-V3权重转换"章节

c. tune_deepseek3_671b_4k_full_xxx.sh (根据集群选择A2或者A3脚本)

d. ckpt_convert_deepseek3_mcore2hf.sh（可选）
```

Lora/QLora微调过程脚本使用顺序为：

```shell
a. data_convert_deepseek3_instruction.sh

b. ckpt_convert_deepseek3_hf2mcore.sh

注意：此脚本是示例，需根据c.tune_deepseek3_671b_4k_xlora_xxx.sh脚本设置权重转换参数。参数说明查看以下的"DeepSeek-V3权重转换"章节

c. tune_deepseek3_671b_4k_xlora_xxx.sh(根据集群选择A2或者A3脚本)

d. ckpt_convert_deepseek3_merge_lora2hf.sh （可选）
```

# DeepSeek-V3权重转换

特别注意：在做权重转换前，请先确认训练时的参数配置，要根据训练脚本pretrain_xxx.sh/tune_xxx.sh参数对ckpt_convert_xxx.sh进行自定义配置。

## 1. hf2mg/mg2hf权重转换说明

### 1.1 转换脚本配置和执行

(1) huggingface转megatron

- 支持将[huggingface权重](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main)转换为分布式megatron mcore权重，用于微调、推理、评估等任务。要求原权重做反量化后获得bf16数据格式，反量化方法请参考MindIE官方提供的[代码](https://modelers.cn/models/MindIE/deepseekv3/blob/main/NPU_inference/fp8_cast_bf16.py)。
- DeepSeek-V3模型目录下的<a href="../../mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh">ckpt_convert_deepseek3_hf2mcore.sh</a>脚本，设置与训练脚本相同配置，再执行转换：
```shell
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh
```

(2) megatron转huggingface

- 支持将训练好的分布式megatron mcore格式的权重转换回huggingface格式。
- DeepSeek-V3模型目录下的<a href="../../mcore/deepseek3/ckpt_convert_deepseek3_mcore2hf.sh">ckpt_convert_deepseek3_mcore2hf.sh</a>脚本，设置与训练脚本相同配置，再执行转换：

```shell
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_mcore2hf.sh
```

(3) LoRA/QLoRA转huggingface

- 支持将训练好的LoRA/QLoRA格式的权重转huggingface格式。
- DeepSeek-V3模型目录下的<a href="../../mcore/deepseek3/ckpt_convert_deepseek3_merge_lora2hf.sh">ckpt_convert_deepseek3_merge_lora2hf.sh</a>脚本，设置与训练脚本相同配置，再执行转换：
```shell
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_merge_lora2hf.sh
```

### 1.2 相关参数说明

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
      <td>--target-tensor-parallel-size, --source-tensor-parallel-size</td>
      <td>张量并行度，默认值为1。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--target-pipeline-parallel-size, --source-pipeline-parallel-size</td>
      <td>流水线并行度，默认值为1。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--target-expert-parallel-size, --source-expert-parallel-size</td>
      <td>专家并行度，默认值为1。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layers-per-virtual-pipeline-stage</td>
      <td>虚拟流水线并行，默认值为None, 注意参数--num-layers-per-virtual-pipeline-stage 和 --num-layer-list 不能同时使用。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--moe-grouped-gemm</td>
      <td>当每个专家组有多个专家时，可以使用Grouped GEMM功能来提高利用率和性能。
注意，QLoRA和LoRA权重转换不支持该参数。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--load-dir</td>
      <td>已经反量化为bf16数据格式的huggingface权重。</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--save-dir</td>
      <td>转换后的megatron格式权重的存储路径。</td>
      <td>❌</td>
    </tr>
    <tr>
      <td>--mtp-num-layers</td>
      <td>MTP层的层数。如不需要MTP层，可设置为0。最大可设置为1。默认值为0。
MTP层权重默认存储在最后一个pp stage。
注意，QLoRA和LoRA权重转换不支持MTP。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layers</td>
      <td>模型层数，该层数不包含MTP层。默认值为61。
如配置空操作层，num-layers的值应为总层数（不包含MTP层）加上空操作层【--noop-layers】层数。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--first-k-dense-replace</td>
      <td>moe层前的dense层数，最大可设置为3。默认值为3。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--num-layer-list</td>
      <td>指定每个pp的层数，相加要等于num-layers。当前仅支持 num-layers = 61 时使用此参数。与--noop-layers互斥，二者选其一使用。默认值为None。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--noop-layers</td>
      <td>自定义空操作层。与--num-layer-list互斥，二者选其一使用。默认值为None。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--moe-tp-extend-ep</td>
      <td>TP拓展EP，专家层TP组不切分专家参数，切分专家数量。默认值为False。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--mla-mm-split</td>
      <td>在MLA中，将2个up-proj matmul操作拆分成4个。默认值为False。
注意，QLoRA和LoRA权重转换不支持该参数。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--schedules-method</td>
      <td>流水线并行方法，可选dualpipev。默认值为None。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--qlora-nf4</td>
      <td>指定是否开启QLoRA权重的量化转换功能，默认为False。</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>--save-lora-to-hf</td>
      <td>加入此参数将单独的不含base权重的LoRA权重转为huggingface格式，与--moe-grouped-gemm不兼容；

在LoRA微调时,脚本中不能加入--moe-grouped-gemm参数，可以在微调脚本中加入--lora-ckpt-filter仅保存LoRA权重。</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## 2 LoRA权重转换

### 2.1 LoRA 权重包含 base 权重

如果 LoRA 权重包含了 base 权重，并且需要将其合并到一起转为huggingface格式：

示例：

```
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-lora \   
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --lora-r 8 \
    --lora-alpha 16
```

--load-dir：填写LoRA权重路径，该权重包括base权重和LoRA权重

--lora-r：LoRA矩阵的秩，需要与LoRA微调时配置相同

--lora-alpha：缩放因子，缩放低秩矩阵的贡献，需要与LoRA微调时配置相同

【适用场景】在LoRA微调时没有加参数'--lora-ckpt-filter'，则保存的权重包括base权重和LoRA权重

### 2.2 LoRA 权重与 base 权重分开加载

如果需要将 base 权重和独立的 LoRA 权重合并转为huggingface格式，可以分别指定两个路径进行加载：

示例：
```
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-mcore \
    --lora-load ./ckpt/filter_lora \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --lora-r 8 \
    --lora-alpha 16
    # --num-layer-list, --noop-layers, --num-layers-per-virtual-pipeline-stage等参数根据任务需要进行配置
```

--load-dir：指定base权重路径

--lora-load：指定LoRA权重路径，注意该权重仅为LoRA权重，在LoRA微调中加入'--lora-ckpt-filter'，只保存LoRA权重

--lora-r、--lora-alpha：与LoRA微调时配置相同

【适用场景】在LoRA微调时加参数'--lora-ckpt-filter'，保存的权重只包含LoRA权重，需要将Lora和HF权重合并

### 2.3 只将LoRA权重转为huggingface格式

如果需要将单独的LoRA权重转为huggingface格式：

```
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 4 \
    --load-dir ./ckpt/lora_v3_filter \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --save-lora-to-hf \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
```

--load-dir：指定LoRA权重路径，注意该权重仅为LoRA权重，在LoRA微调中加入'--lora-ckpt-filter'，只保存LoRA权重

--lora-target-modules：定义了LoRA目标模块，字符串列表，由空格隔开，无默认值。每一个字符串是需要进行LoRA微调的层的名称。

--save-lora-to-hf：指定此参数,仅将LoRA权重转为huggingface格式,注意该权重仅为LoRA权重，在LoRA微调中加入'--lora-ckpt-filter'，只保存LoRA权重

【适用场景】在LoRA微调时加参数'--lora-ckpt-filter'，则保存的权重只包含LoRA权重，仅将Lora权重转为HF格式

## 3 QLoRA 权重转换

### 3.1 QLoRA 权重包含 base 权重

如果 QLoRA 权重包含了 base 权重，并且需要将其合并到一起转为huggingface格式：

在微调脚本中加入'--qlora-save-dequantize',保存时将权重反量化。

【适用场景】在QLoRA微调时没有加参数'--lora-ckpt-filter'，则保存的权重包括base权重和QLoRA权重

合并脚本同`2.1 LoRA 权重包含 base 权重`

### 3.2 LoRA 权重与 base 权重分开加载

如果需要将 base 权重和独立的 QLoRA 权重合并转为huggingface格式，可以分别指定两个路径进行加载：

示例：
```
python examples/mcore/deepseek3/convert_ckpt_deepseek3_mcore2hf.py \
    --source-tensor-parallel-size 1 \
    --source-pipeline-parallel-size 4 \
    --source-expert-parallel-size 8 \
    --load-dir ./model_weights/deepseek3-mcore \
    --lora-load ./ckpt/filter_lora \
    --save-dir ./model_from_hf/deepseek3-hf \
    --num-layers 61 \
    --first-k-dense-replace 3 \
    --num-layer-list 16,15,15,15 \
    --lora-r 8 \
    --lora-alpha 16
    # --num-layer-list, --noop-layers, --num-layers-per-virtual-pipeline-stage等参数根据任务需要进行配置
```

--load-dir：指定base权重路径，由于QLoRA微调加载的权重是量化过的，所以不能直接作为base权重，需要重新转出一份不加参数'--qlora-nf4'的mcore权重作为合并时的base权重

--lora-load：指定QLoRA权重路径，注意该权重仅为QLoRA权重，在微调脚本中加入'--qlora-save-dequantize',保存时将权重反量化，并加入'--lora-ckpt-filter'，只保存QLoRA权重

--lora-r、--lora-alpha：与LoRA微调时配置相同

【适用场景】在QLoRA微调时加参数'--lora-ckpt-filter'，保存的权重只包含LoRA权重，需要将Lora和HF权重合并

### 3.3 只将QLoRA权重转为huggingface格式

如果需要将单独的QLoRA权重转为huggingface格式，在微调脚本中加入'--qlora-save-dequantize',保存时将权重反量化，并加入'--lora-ckpt-filter'，只保存QLoRA权重。

转换脚本同`2.3 只将LoRA权重转为huggingface格式`

【适用场景】在QLoRA微调时加参数'--lora-ckpt-filter'，则保存的权重只包含LoRA权重，仅将Lora权重转为HF格式

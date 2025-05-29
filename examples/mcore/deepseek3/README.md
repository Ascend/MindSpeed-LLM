# DeepSeek-V3权重转换

## 1 使用说明

支持将已经反量化为bf16数据格式的[huggingface权重](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/tree/main)转换为mcore权重，用于微调、推理、评估等任务。反量化方法请参考DeepSeek官方提供的[代码](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/inference/fp8_cast_bf16.py)。

支持将训练好的megatron mcore格式的权重转换回huggingface格式。

### 1.1 启动脚本

使用DeepSeek-V3模型目录下的<a href="../../mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh">huggingface转megatron脚本</a>、<a href="../../mcore/deepseek3/ckpt_convert_deepseek3_mcore2hf.sh">megatron转huggingface脚本</a>和<a href="../../mcore/deepseek3/ckpt_convert_deepseek3_merge_lora2hf.sh">lora转huggingface脚本</a>

#### huggingface转megatron
```bash
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_hf2mcore.sh
```

#### megatron转huggingface

```bash
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_mcore2hf.sh
```

#### lora/qlora转huggingface

```bash
bash examples/mcore/deepseek3/ckpt_convert_deepseek3_merge_lora2hf.sh
```

### 1.2 相关参数说明

【--moe-grouped-gemm】

当每个专家组有多个专家时，可以使用Grouped GEMM功能来提高利用率和性能。
注意，qlora和lora权重转换不支持该参数。

【--target-tensor-parallel-size】

张量并行度，默认值为1。

【--target-pipeline-parallel-size】

流水线并行度，默认值为1。

【--target-expert-parallel-size】

专家并行度，默认值为1。

【--num-layers-per-virtual-pipeline-stage】

虚拟流水线并行，默认值为None, 注意参数--num-layers-per-virtual-pipeline-stage 和 --num-layer-list 不能同时使用。

【--load-dir】

已经反量化为bf16数据格式的huggingface权重。

【--save-dir】

转换后的megatron格式权重的存储路径。

【--mtp-num-layers】

MTP层的层数。如不需要MTP层，可设置为0。最大可设置为1。默认值为0。
MTP层权重默认存储在最后一个pp stage。
注意，qlora和lora权重转换不支持MTP。

【--num-layers】

模型层数，该层数**不包含MTP层**。默认值为61。
如配置空操作层，num-layers的值应为总层数（不包含MTP层）加上空操作层【--noop-layers】层数。

【--first-k-dense-replace】

moe层前的dense层数，最大可设置为3。默认值为3。

【--num-layer-list】

指定每个pp的层数，相加要等于num-layers。当前仅支持 num-layers = 61 时使用此参数
与--noop-layers互斥，二者选其一使用。默认值为None。

【--noop-layers】

自定义空操作层。与--num-layer-list互斥，二者选其一使用。默认值为None。

【--moe-tp-extend-ep】

TP拓展EP，专家层TP组不切分专家参数，切分专家数量。默认值为False。

【--mla-mm-split】

在MLA中，将2个up-proj matmul操作拆分成4个。默认值为False。
注意，qlora和lora权重转换不支持该参数。

【--schedules-method】

流水线并行方法，可选dualpipev。默认值为None。

【--qlora-nf4】

指定是否开启QLoRA权重量化转换，默认为False.

【--save-lora-to-hf】

加入此参数将单独的不含base权重的lora权重转为huggingface格式，与--moe-grouped-gemm不兼容；

在lora微调时,脚本中不能加入--moe-grouped-gemm参数，可以在微调脚本中加入--lora-ckpt-filter仅保存lora权重。

## 2 lora权重转换

### 2.1 lora 权重包含 base 权重

如果 lora 权重包含了 base 权重，并且需要将其合并到一起转为huggingface格式：

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
    --lora-alpha 16 \
```

【--load-dir】填写lora权重路径，该权重包括base权重和lora权重

【--lora-r】lora矩阵的秩，需要与lora微调时配置相同

【--lora-alpha】缩放因子，缩放低秩矩阵的贡献，需要与lora微调时配置相同

【适用场景】在lora微调时没有加参数'--lora-ckpt-filter'，则保存的权重包括base权重和lora权重

### 2.2 lora 权重与 base 权重分开加载

如果需要将 base 权重和独立的 lora 权重合并转为huggingface格式，可以分别指定两个路径进行加载：

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
    --lora-alpha 16 \
    # --num-layer-list, --noop-layers, --num-layers-per-virtual-pipeline-stage等参数根据任务需要进行配置
```

【--load-dir】指定base权重路径

【--lora-load】指定lora权重路径，注意该权重仅为lora权重，在lora微调中加入'--lora-ckpt-filter'，只保存lora权重

【--lora-r】、【--lora-alpha】与lora微调时配置相同

### 2.3 只将lora权重转为huggingface格式

如果需要将单独的lora权重转为huggingface格式：

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
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
```

【--load-dir】指定lora权重路径，注意该权重仅为lora权重，在lora微调中加入'--lora-ckpt-filter'，只保存lora权重

【--lora-target-modules】定义了Lora目标模块，字符串列表，由空格隔开，无默认值。每一个字符串是需要进行LoRA微调的层的名称。

【--save-lora-to-hf】指定此参数,仅将lora权重转为huggingface格式,注意该权重仅为lora权重，在lora微调中加入'--lora-ckpt-filter'，只保存lora权重

## 3 qlora 权重转换

### 3.1 qlora 权重包含 base 权重

如果 qlora 权重包含了 base 权重，并且需要将其合并到一起转为huggingface格式：

在微调脚本中加入--qlora-save-dequantize,保存时将权重反量化。

【适用场景】在lora微调时没有加参数'--lora-ckpt-filter'，则保存的权重包括base权重和qlora权重

合并脚本同`2.1 lora 权重包含 base 权重`

### 3.2 lora 权重与 base 权重分开加载

如果需要将 base 权重和独立的 qlora 权重合并转为huggingface格式，可以分别指定两个路径进行加载：

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
    --lora-alpha 16 \
    # --num-layer-list, --noop-layers, --num-layers-per-virtual-pipeline-stage等参数根据任务需要进行配置
```

【--load-dir】指定base权重路径，由于qlora微调加载的权重是量化过的，所以不能直接作为base权重，需要重新转出一份不加参数'--qlora-nf4'的mcore权重作为合并时的base权重

【--lora-load】指定qlora权重路径，注意该权重仅为qlora权重，在微调脚本中加入'-qlora-save-dequantize',保存时将权重反量化，并加入'--lora-ckpt-filter'，只保存qlora权重

【--lora-r】、【--lora-alpha】与lora微调时配置相同

### 3.3 只将qlora权重转为huggingface格式

如果需要将单独的qlora权重转为huggingface格式，在微调脚本中加入'-qlora-save-dequantize',保存时将权重反量化，并加入'--lora-ckpt-filter'，只保存qlora权重。

转换脚本同`2.3 只将lora权重转为huggingface格式`

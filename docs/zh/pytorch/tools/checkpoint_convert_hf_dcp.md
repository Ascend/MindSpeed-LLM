# MindSpeed LLM FSDP2后端DCP权重转换工具使用指南

## 使用场景

在大模型训练与部署过程中，DCP权重转换工具常见的使用场景包括：

- 使用DCP格式进行训练，但需要将训练后的模型权重文件转换为HuggingFace标准格式用于推理或下游任务。
- 模型参数规模较大，无法一次性加载完整权重到内存进行格式转换。

`merge_dcp_to_hf.py` 脚本采用分片逐步合并的方式，在保证正确性的同时，最大限度降低内存占用，适用于大模型权重的格式转换。

## 使用方法

### 1. 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--load-dir` | str | 无（必填） | DCP权重所在目录，需包含有效的PyTorch分布式权重文件 |
| `--save-dir` | str | `<load-dir>/hf_ckpt` | HuggingFace格式权重的输出目录 |
| `--model-configs` | str | None | 模型配置文件目录，将被复制到输出目录 |
| `--shard-size` | int | `5000000000`（5GB） | 单个权重分片文件的最大字节数 |

### 2. 基本转换

最简单的使用方式如下：

```bash
python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \   
        --load-dir <DCP权重路径>
```

执行后将在 `<DCP权重路径>/hf_ckpt` 目录下生成HuggingFace格式的权重。

### 3. 指定输出目录

如需将转换后的模型保存到指定位置，可通过`--save-dir`参数指定输出目录：

```bash
python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \
        --load-dir checkpoints/my_model/dcp_checkpoint \
        --save-dir hf_models/my_model
```

### 4. 同时拷贝模型配置文件

若希望生成的HuggingFace权重可直接通过 `from_pretrained()` 加载，需提供模型配置文件（如 `config.json`、`tokenizer.json`）：

```bash
python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \
        --load-dir checkpoints/my_model/dcp_checkpoint \
        --save-dir hf_models/my_model \
        --model-configs pretrained_models/qwen3-8b
```

脚本会将该目录下的配置文件一并复制到输出目录。

### 5. 控制权重分片大小

对于内存资源受限的环境，可以通过 `--shard-size` 控制单个权重分片的最大大小：

```bash
python mindspeed_llm/fsdp2/checkpoint/merge_dcp_to_hf.py \
        --load-dir checkpoints/my_model/dcp_checkpoint \
        --shard-size 2000000000
```

上述示例将单个分片大小限制为 2GB，有助于降低转换过程中的内存峰值。

## 输出说明

1. 当模型总大小不超过`shard-size`时，输出为单文件：

    ```text
    output_directory/
    ├── model.safetensors
    ├── config.json
    └── tokenizer.json
    ```

2. 当模型总大小超过`shard-size`时，输出为分片形式：

    ```text
    output_directory/
    ├── model-00001-of-00005.safetensors
    ├── model-00002-of-00005.safetensors
    ├── model-00003-of-00005.safetensors
    ├── model-00004-of-00005.safetensors
    ├── model-00005-of-00005.safetensors
    ├── model.safetensors.index.json
    ├── config.json
    └── tokenizer.json
    ```

## 注意事项

- `--load-dir`必须指向完整且有效的DCP权重目录。
- 脚本不会自动生成模型配置文件，如需HuggingFace直接加载，请显式指定`--model-configs`。
- 转换过程采用分片加载方式，但仍需保证单个分片可被当前环境正常加载。

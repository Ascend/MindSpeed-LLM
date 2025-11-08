# 一键集成训练 (All_in_One_Trainer)

## 使用场景

在之前的版本中，用户需要先离线执行权重转换，将 HuggingFace 格式的权重转换为 Megatron 格式，然后再启动训练过程。这种分离的操作方式增加了使用复杂度和时间成本。

本功能通过 `--enable-hf2mg-convert` 参数实现 HuggingFace 权重到 Megatron 格式的自动转换与训练合一，支持共享存储和非共享存储环境，用户无需独立执行权重转换步骤，真正实现从 HuggingFace 权重到训练任务的一键式集成。

## 使用方法

当前支持共享存储和非共享存储环境，系统会在训练初始化阶段自动检测存储类型并采用最优的权重转换策略，用户无需手动配置：

- **共享存储环境**：所有计算节点可访问同一存储路径，仅需 rank0 进程执行权重转换，其他进程等待转换完成后开始训练
- **非共享存储环境**：各计算节点使用本地独立存储，每个节点的 local_rank=0 进程分别执行权重转换
- **混合存储环境**：暂不支持部分节点为共享存储、部分节点为非共享存储的异构环境


**1.基本命令**

在`pretrain_xxx.sh` 或者`tune_xxx.sh`的预训练和微调脚本中，增加以下参数：

```bash
    --enable-hf2mg-convert \
    --model-type-hf <model_type> \
```

**2.参数说明**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--enable-hf2mg-convert` | flag | 是 | 启用 HF 到 Megatron 权重转换功能 |
| `--model-type-hf` | string | 是 | 指定 HuggingFace 模型类型 |


**3.CKPT_LOAD_DIR 说明**

在增加--enable-hf2mg-convert后，CKPT_LOAD_DIR要求权重类型：Huggingface格式权重。Huggingface权重路径下必须包含配置文件 `config.json` 和模型文件（`.bin` 或 `.safetensors` 格式），并且模型文件名中需包含 "model" 关键词

请确保有足够的磁盘空间存放转换后的权重，转换后的权重文件将保存在 `{load_dir}/megatron_cache` 目录中，训练过程会自动使用该路径作为权重加载路径。

训练初始化后自动进行权重转换过程，根据模型参数时间预计需要2分钟-2小时，请耐心等待。

注意：

- 开启--enable-hf2mg-convert 参数后，不支持使用离线转换的mcore格式Megatron权重。
- 请确保对`{load_dir}`路径有读写权限


## 使用约束

**1.支持的模型类型**

当前支持的 HuggingFace 模型类型：
- `qwen3`
- `qwen3-moe`
- `deepseek3` 
- `glm45-moe`
- `bailing_mini`
- `qwen3-next`
- `seed-oss`
- `deepseek32`
- `magistral`
- `deepseek2-lite`

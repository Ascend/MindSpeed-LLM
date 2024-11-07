# Falcon-40B模型-推理指导

## 概述

Falcon-40B是由TII构建的一个40B参数的因果解码器模型。

## 输入输出数据


| 输入数据       | 大小                               | 数据类型 | 数据排布格式 | 是否必选 |
| -------------- | ---------------------------------- | -------- | ------------ | -------- |
| input_ids      | BATCH_SIZE x SEQ_LEN               | INT64    | ND           | 是       |
| attention_mask | BATCH_SIZE x 1 x SEQ_LEN x SEQ_LEN | FLOAT32  | ND           | 否       |



| 输出数据   | 大小                        | 数据类型 | 数据排布格式 |
| ---------- | --------------------------- | -------- | ------------ |
| output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64    | ND           |

## 推理环境准备

该模型需要以下插件与驱动

**表 1** 版本配套表

| 配套           | 版本            | 下载链接 |
| -------------- | --------------- | -------- |
| 固件与驱动     | 23.0.rc3.6.b060 | -        |
| CANN           | 7.0.RC1         | -        |
| python         | 3.9.11          | -        |
| PytorchAdapter | 1.11.0          | -        |
| transformers   | 4.35.0          | -        |

**表 2** 推理引擎依赖

| 软件  | 版本要求 |
| ----- | -------- |
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device |
| ------- | ------ |
| aarch64 | Atlas 800T A2  |

## 多卡模型推理

Falcon-40B 可以在单机 4 卡上进行推理，切分权重时应将 world_size 设置为 4.

### 切分权重

从 HuggingFace 上下载的权重路径替换下面命令中的 `{HuggingFase下载的权重路径}`，然后选择一个想要保存的切分后的权重路径，替换下面命令中的 `{切分后的权重保存的路径}`：

```bash
python main.py --mode cut --model_path {HuggingFase下载的权重路径} --output_model_path {切分后的权重保存的路径} --world_size 4
```

切分所需时间较长，切分完成后，将会打印 'save model weights succcessfully' 并在指定输出路径下可看到路径中包含 'tokenizer' 和 'part_model' 两个文件夹。

### 执行推理

#### 推理性能测试

在上一步完成权重切分后，使用切分后的权重进行推理：

```bash
export ATB_OPERATION_EXECUTE_ASYNC=1
torchrun --nproc_per_node 4 --master_port 11751 main.py --device 0 1 2 3 --model_path {切分后的权重保存的路径}
```

#### 推理下游任务精度测试

下载 MMLU 数据集并解压，比如到 `/path/to/mmlu`，这时在改路径下会有"dev"、"val"、"test"三个子文件夹，不需要进入某个文件夹，直接用`/path/to/mmlu` 替换 {mmlu数据集路径}：

```bash
export ATB_OPERATION_EXECUTE_ASYNC=1
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar -xvf data.tar
torchrun --nproc_per_node 4 --master_port 11751 main.py --model_path {切分后的权重保存的路径} --device 0 1 2 3 --mode precision --dataset-path {mmlu数据集路径}
```

## 测试结果

### 模型输出精度

使用余弦相似度对比输出精度，在代码中计算npu和加速库的logits的余弦相似度并断言满足大于0.999：

```python
cos = torch.nn.CosineSimilarity(dim=0)
cos_output = cos(lm_logits.view(-1).to(torch.float64), lm_logits_to_comp.view(-1).to(torch.float64))
print(f"Cosine Similarity: {cos_output}")
assert cos_output > 0.999
```

测试脚本可以正常运行，断言没有抛出异常，满足余弦相似度大于0.999的要求。

### 下游任务准确率

下游任务精度达到竞品的 99.999% 以上，满足大于 99.9% 的要求。

| Model      | Data | Platform | STEM   | Social sciences | Humanities | Other  | Avg     |
| ---------- | ---- | -------- | ------ | --------------- | ---------- | ------ | ------- |
| Falcon-40B | MMLU | 参考     | 0.4112 | 0.6498          | 0.4980     | 0.6535 | 54.9305 |
| Falcon-40B | MMLU | Atlas 900 A2 PODc     | 0.4049 | 0.6528          | 0.5019     | 0.6507 | 54.9302 |

### 推理性能

NPU Atlas 800T A2 测试结果：

| Batch | TokensPerSecond | ResponseTime(ms) | FirstTokenTime(ms) | TimePerTokens(ms) |
| ----- | --------------- | ---------------- | ------------------ | ----------------- |
| 1     | 37.0206184      | 9204.322         | 92.83104           | 27.0287486        |

GPU A100 测试结果：

| Batch | TokensPerSecond | ResponseTime(ms) | FirstTokenTime(ms) | TimePerTokens(ms) |
| ----- | --------------- | ---------------- | ------------------ | ----------------- |
| 1     | 50.65095583     | 6816.21          | 150.4420922        | 19.74296405       |

首Token推理性能高于竞品，非首Token推理性能达到竞品的 0.73 (19.74296405/27.0287486=0.7304431419366564)。
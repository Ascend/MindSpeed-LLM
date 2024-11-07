# 启动脚本使用指南

```shell
bash start.sh ${device_ids} ${max_memory_gb} ${running_mode}
```

运行

```shell
```shell
bash start.sh 2 35 
```

调试

```shell
bash start.sh 2 35 debug
```

## 推荐参数

| 场景      | max_memory_gb |
|---------|---------------|
| Atalas推理系列产品 单芯 | 35            |
| Atalas推理系列产品 双芯 | 35            |
| Atlas 900 A2 PODc 单卡 | 57            |
| Atlas 900 A2 PODc 双卡 | 57            |


## 量化推理

# 量化工具使用

量化权重的获取需要使用大模型量化工具（集成至CANN包中），详细操作手册可见[大模型权重量化工具-ModelSlim](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/devtools/auxiliarydevtool/modelslim_0001.html)
。针对Baichuan2-13B的权重量化可参考如下步骤，运行时需将下述三个步骤的代码整合为一个python文件

**特别注意1**：本章节依赖**pytorch >= 2.0.0 CANN >= 7.0.0.B060**
环境，大模型量化工具依赖指定pytorch版本（不依赖torch_npu，只依赖原生torch）。该环境的pytorch版本与后续步骤可能不同，后续将优化pytorch版本依赖的限制

**特别注意2**：本章节依赖 hugging face 的标准 transformers 包。若环境中的 transformers 包被改动过，可能引起相关报错，此时建议重新安装
transformers 包

**特别注意3**：本章节执行完毕后，在`QUANT_WEIGHT_PATH`路径下生成如下权重文件，请检查是否缺失：

```
deq_scale.npy  fp_bias.npy
input_offset.npy  input_scale.npy
quant_bias.npy  quant_weight.npy
weight_offset.npy  weight_scale.npy
```

## 校准数据准备

```python
calib_list = ["中国的首都在哪里？",
              "请做一首诗歌：",
              "我想要学习python，该怎么学习？",
              "请帮我写一篇关于大模型推理优化的任职报告：",
              "中国最值得去的几个景点"]


# 获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], inputs.data['position_ids'], inputs.data['attention_mask']])
    return calib_dataset


dataset_calib = get_calib_dataset(tokenizer, calib_list)  # 校准数据获取
```

## 量化参数配置与运行

```python
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

quant_config = QuantConfig(w_bit=8, disable_names=['transformer.output_layer'], dev_type='cpu', act_method=3, pr=0.5,
                           mm_tensor=False, w_hessian=False)
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L1')
calibrator.run()  # 执行PTQ量化校准
calibrator.save('QUANT_WEIGHT_PATH')  # 保存量化参数
```

- 建议直接使用量化权重生成脚本，生成量化权重 quant.py脚本通过disable_idx_lst指定了需要回退的layerid，如果没有 可以通过disable_level自动选择缺省层数，具体参考modelslim使用方法
  ```
  python quant.py
  ```

> 注：要使用torch>=2.0.0导出量化权重，否则会有精度偏差 quant.py脚本需要修改calibrator.save('QUANT_WEIGHT_PATH') 最终量化全的指定路径
> > 注：tgi flash_baichuan2_13b_modeling_quant_ascend.py 脚本 self.quant_weight_path 改成此路径 self.roll_back_layer 改成与 disable_idx_lst 对应，如果是自动选择生成，需要查看缺省了哪些layer 将对应的layerid填入self.roll_back_layer
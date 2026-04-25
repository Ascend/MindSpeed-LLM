# DeepSeek-V4模型训练

2026年4月24日，DeepSeekV4系列模型DeepSeekV4-Flash和DeepSeekV4-Pro正式发布并开源，其以先进的 MoE 架构在参数效率与推理性能上实现了重要突破。MindSpeed LLM 目前已实现了 **DeepSeekV4-Flash模型** 的定长数据场景下的预训练支持，并同步开放源代码。目前提供预训练的实践参考，帮助用户快速上手。

**注：当前实现为preview版本，部分场景存在限制，后续我们将持续完善并同步跟进 DeepSeek-V4 技术报告中的演进方向：**

<table style="text-align: center">
  <thead>
    <tr>
      <th style="text-align: center;">场景</th>
      <th style="text-align: center;">特性</th>
      <th style="text-align: center;">支持情况</th>
      <th style="text-align: center;">备注</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">训练场景</td>
      <td>预训练/续训</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>全参微调</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td>Lora微调</td>
      <td>TODO</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">数据场景</td>
      <td>定长数据</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>pack数据</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td>变长数据</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td rowspan="4">切分策略</td>
      <td>TP</td>
      <td>OK</td>
      <td>支持TP=2</td>
    </tr>
    <tr>
      <td>PP</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>EP</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>CP</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td rowspan="1">功能特性</td>
      <td>Muon优化器</td>
      <td>DOING</td>
    </tr>
  </tbody>
</table>

## 安装指导

请参考仓库 [安装指导](../../../docs/zh/pytorch/training/install_guide.md)文档配置环境和拉取仓库代码

## 权重转换

1. 权重下载 

    从 [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base) 下载权重和配置文件

2. 权重转换
   
    开源DeepSeekV4-Flash权重为FP8 mixed数据格式，训练前需要对原始权重做反量化后获得bf16格式的权重，反量化方法请参考下述脚本

    ```sh
    cd MindSpeed-LLM
    bash examples/mcore/deepseek4_flash/ckpt_dequant_deepseek4_fp8_to_bf16.sh

    ```
    
    MindSpeed LLM提供[脚本](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash)将已经huggingface开源权重转换为mcore权重，用于训练、推理、评估等任务。
    使用方法如下，请根据实际需要的TP/PP等切分策略和权重路径修改权重转换脚本

    ```sh
    cd MindSpeed-LLM
    bash examples/mcore/deepseek4_flash/ckpt_convert_deepseek4_flash_hf2mcore.sh # 转换时，请指定路径到反量化后的权重
    ```

## 数据预处理

MindSpeed LLM提供[脚本](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) 进行数据集处理

使用方法如下，请根据实际需要修改以下参数

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/data_convert_deepseek4_pretrain.sh
```
  
| 参数名  | 含义                |
|--------|-----------------|
| --input | 数据集路径  |
| --tokenizer-name-or-path | 模型tokenizer目录    |
| --output-prefix | 数据集处理完的输出路径及前缀名  |
  
## 训练

MindSpeed LLM提供[脚本](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) 进行模型训练

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh
```

用户需要根据实际情况修改脚本中以下变量

  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CKPT_SAVE_DIR | 训练中权重保存路径  |
  | DATA_PATH | 数据预处理后的数据路径  |
  | TOKENIZER_PATH | tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重，如无初始权重则随机初始化  |

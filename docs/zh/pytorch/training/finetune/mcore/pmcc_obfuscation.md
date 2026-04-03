# 微调PMCC混淆

## 使用场景

PMCC混淆是指对微调过程使用的模型文件和数据集进行混淆处理，避免存储或计算过程中潜在的模型或数据泄露风险，保护模型与数据的机密性。

## 使用方法

由于PMCC混淆功能当前仅支持Qwen3-32b模型，因此本文档以该模型为例介绍PMCC使能方法，具体步骤如下：

1. 参考[MindSpeed LLM安装指导](../../../../install_guide.md)，完成环境安装。 
    
    请在训练开始前配置好昇腾NPU套件相关的环境变量，如下所示：

    ```shell
    source /usr/local/Ascend/cann/set_env.sh     # 修改为实际安装的Toolkit包路径
    source /usr/local/Ascend/nnal/atb/set_env.sh # 修改为实际安装的nnal包路径
    ```

2. 准备好模型权重和微调数据集。

    完整的[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/tree/main)模型文件夹应该包括以下内容：

    ```shell
    .
    ├── README.md                    # 模型说明文档
    ├── config.json                  # 模型结构配置文件
    ├── generation_config.json       # 文本生成时的配置
    ├── merges.txt                   # tokenizer的合并规则文件
    ├── model-00001-of-00017.safetensors  # 模型权重文件第1部分（共17部分）
    ├── model-00002-of-00017.safetensors  # 模型权重文件第2部分
    ├── ...
    ├── model-00016-of-00017.safetensors  # 模型权重文件第16部分
    ├── model-00017-of-00017.safetensors  # 模型权重文件第17部分
    ├── model.safetensors.index.json      # 权重分片索引文件，指示各个权重参数对应的文件
    ├── tokenizer.json              # Hugging Face格式的tokenizer
    ├── tokenizer_config.json       # tokenizer相关配置
    └── vocab.json                  # 模型词表文件
    ```

3. 安装PMCC混淆包。

    ```shell
    pip3 install ai_asset_obfuscate
    ```

4. 进行模型混淆。

    新建模型混淆脚本`obf_model.py`，具体内容如下，需要设置模型权重路径和混淆因子内容：

    ```python
    from sys import argv
    from ai_asset_obfuscate import ModelAssetObfuscation, ModelType
    
    # 模型混淆
    seed_content = "xxxxxx"    # 混淆因子内容，要求长度32位的字符串
    model_path = "./model_from_hf/qwen3_hf/"     # 原始HF模型权重路径
    save_path = "./model_from_hf/qwen3_obf_hf/"  # 混淆后的HF模型权重路径
    model = ModelAssetObfuscation.create_model_obfuscation(model_path, ModelType.QWEN3) # ModelType.QWEN3表示需要混淆的模型类型
    
    res = model.set_seed_content(seed_type=2, seed_content=seed_content) # seed_type=1表示模型混淆因子，2表示数据混淆因子，微调场景只涉及基于数据混淆因子对模型进行保护
    print(res)
    
    res = model.model_weight_obf(obf_type=2, model_save_path=save_path, device_type="npu", device_id=[0, 1, 2, 3, 4, 5, 6, 7]) # obf_type=1表示用于模型保护的模型混淆，2表示用于数据保护的模型混淆，微调场景只涉及2。device_type="npu"表示使用npu加速，device_id表示npu设备id
    print(res)
    ```

    确认配置无误后运行模型混淆脚本：
    
    ```shell
    python obf_model.py
    ```

5. 进行权重转换，即将混淆后的HF权重转换成Megatron权重。

    以Qwen3-32B模型在TP8PP2切分为例，详细配置请参考[Qwen3权重转换脚本](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh)。需要修改相关路径参数和模型切分配置：

    ```shell
    --target-tensor-parallel-size 8          # TP切分大小
    --target-pipeline-parallel-size 2        # PP切分大小
    --load-dir ./model_from_hf/qwen3_obf_hf/ # 混淆后的HF模型权重路径
    --save-dir ./model_weights/qwen3_mcore/  # Megatron权重保存路径
    ```
    
    确认路径无误后运行权重转换脚本：
    
    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

6. 进行数据预处理。

    以[Alpaca数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)为例执行数据预处理，详细配置请参考[Qwen3数据预处理脚本](../../../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh)。需要修改脚本内的路径并增加数据混淆相关参数：

    ```shell
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # 原始数据集路径 
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf # HF的tokenizer路径
    --output-prefix ./finetune_dataset/alpaca         # 保存路径
    ......
    --data-obfuscation             # 数据混淆开关
    --obf-seed-content "xxxxxx"    # 混淆因子内容，长度为32位的字符串，必须与模型混淆时设置一致
    ```
    
    相关参数设置完毕后，运行数据预处理脚本：
    
    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

7. 启动微调。

    配置模型微调脚本，详细配置请参考[Qwen3-32b微调脚本](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh)，需要修改相关路径参数和模型切分配置。注意：训练参数的并行配置，如TP/PP等需要与第五步权重转换时的配置保持一致。

    ```shell
    CKPT_LOAD_DIR="your model ckpt path"      # 权重加载路径，填入权重转换时保存的权重路径
    CKPT_SAVE_DIR="your model save ckpt path" # 微调完成后的权重保存路径
    DATA_PATH="your data path"                # 数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
    TOKENIZER_PATH="your tokenizer path"      # 词表路径，填入下载的开源权重词表路径
    TP=8                                      # 权重转换时target-tensor-parallel-size的值
    PP=2                                      # 权重转换时target-pipeline-parallel-size的值
    ```
    
    相关参数设置完毕后，运行微调脚本：
    
    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

8. 模型解混淆。

    微调后的模型仍处于混淆态，需要解混淆还原为明文状态。解混淆需要先将微调后的模型权重转回HF格式，详细配置请参考[Qwen3权重转换脚本](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh)。需要修改相关路径参数和模型切分配置：

    ```shell
    --load-dir ./ckpt/qwen3_obf_mg/           # 权重加载路径，填入权重转换时保存的权重路径
    --save-dir ./model_from_hf/qwen3_obf_hf/  # Megatron→HF权重转换时，HuggingFace权重保存目录
    --hf-cfg-dir ./model_from_hf/qwen3_hf/    # HuggingFace配置文件目录
    ```
    
    参数说明：
    
    - `hf-cfg-dir`：由于Megatron→HF权重转换仅生成权重以及`model.safetensors.index.json`，不会生成解混淆需要读取的配置文件，通过指定此参数，将原HuggingFace模型的配置文件复制到权重转换生成的HuggingFace权重目录。
    
    确认路径无误后运行权重转换脚本：
    
    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh
    ```
    
    新建模型解混淆脚本`unobf_model.py`，具体内容如下，需要设置模型权重路径和混淆因子内容：
    
    ```python
    from sys import argv
    from ai_asset_obfuscate import ModelAssetObfuscation, ModelType
    
    # 模型解混淆
    seed_content = "xxxxxx"    # 混淆因子内容，必须与模型混淆时设置一致
    model_path = "./model_from_hf/qwen3_obf_hf/"     # Megatron→HF权重转换时，HuggingFace权重保存目录
    save_path = "./model_from_hf/qwen3_un_obf_hf/"   # 解混淆后的HF模型权重保存路径
    model = ModelAssetObfuscation.create_model_obfuscation(model_path, ModelType.QWEN3, is_obfuscation = False) # is_obfuscation = False表示解混淆
    
    res = model.set_seed_content(seed_type=2, seed_content=seed_content)
    print(res)
    
    res = model.model_weight_obf(obf_type=2, model_save_path=save_path, device_type="npu", device_id=[0, 1, 2, 3, 4, 5, 6, 7])
    print(res)
    ```
    
    确认配置无误后运行模型混淆脚本：
    
    ```shell
    python unobf_model.py
    ```

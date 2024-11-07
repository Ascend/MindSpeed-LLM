# Aquila-7B
<p align="left">
        <b>简体中文</b> |
        <b><a href="https://gitee.com/ascend/ModelLink/blob/modellink/examples/bloom/README_en.md">English</a> </b> 
    </p>
</p>

- [Aquila-7B](#Aquila-7B)
  - [训练](#训练)
    - [脚本](#脚本)
  - [推理](#推理)
  - [评估](#评估)

## 训练

Aquila-7B 训练的硬件配置如下:


|  硬件 |  设备 |        配置       |
|:---:|:---:|:---------------:|
| NPU | A+K | 8 x Ascend NPUs |

### 脚本

1. 克隆仓库到本地服务器并切换到modellink 1.1分支

    ```shell
    git clone https://gitee.com/ascend/ModelLink.git
    cd ModelLink
    git checkout 1.0.RC2
    cd ..
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../ModelLink/
    cd ..
    cd ModelLink
    mkdir logs
    mkdir model_from_hf
    mkdir dataset
    mkdir ckpt
    ```

2. 搭建conda环境

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test
    # 通过互联网上提供的pip源安装 torch，可能需要尝试合适的包含这个torch==2.1.0版本的pip源
    pip install torch==2.1.0
    # 通过PTA上提供的安装包，以whl文件方式安装aarch64架构上的2.1.0版本的torch_npu
    pip install torch_npu-2.1.0.postxxxx-cp38-cp38-xxxx_aarch64.whl
    # 通过PTA上提供的安装包，以whl文件方式安装apex
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl
    
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 拉取MindSpeed源代码，进入MindSpeed目录，然后源码方式安装mindspeed加速包
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed/
    git checkout 2b0edd2
    pip install -r requirements.txt
    pip install -e .
    cd ..
    
    # 安装其余依赖包
    pip install -r requirements.txt
    ```

3. 使用浏览器下载 [Aquila-7B模型的配置，tokenizer，和预训练权重](https://huggingface.co/BAAI/Aquila-7B/tree/main)

    保存在 ModelLink/model_from_hf/Aquila-7B/ 目录。

4. 数据预处理

    第一步，使用浏览器 [下载数据集](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet)， 保存在ModelLink/dataset/ 目录

    ```shell
    cd dataset/
    wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
    cd ..
    ```

    第二步，使用Aquila-7B指定的tokenizer处理数据集：

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    mkdir ./dataset/Aquila-7B/
    python ./tools/preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/Aquila-7B/ \
        --output-prefix ./dataset/Aquila-7B/alpaca \
        --workers 4 \
        --log-interval 1000  \
        --tokenizer-type PretrainedFromHF
    ```

5. 权重转换

    将模型权重文件从 HuggingFace权重 格式转化为 Megatron 权重
    ***（该场景一般用于使能开源的HuggingFace模型在Megatron上进行训练）***

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py \
        --model-type GPT \
        --load-dir ./model_from_hf/Aquila-7B/ \
        --save-dir ./model_weights/Aquila-7B-v0.1-tp8-pp1/ \
        --loader llama2_hf \
        --saver megatron \
        --target-tensor-parallel-size 8 \
        --tokenizer-model ./model_from_hf/Aquila-7B/tokenizer.json
    ```

    任意并行切分策略的Megatron权重 格式转化为 HuggingFace权重
    ***（该场景一般用于将训练好的megatron模型重新转回HuggingFace格式）***

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python tools/checkpoint/convert_ckpt.py --model-type GPT \
        --loader megatron \
        --saver megatron \
        --save-model-type save_huggingface_llama \
        --load-dir ./model_weights/Aquila-7B-v0.1-tp8-pp1/ \
        --target-tensor-parallel-size 1 \
        --target-pipeline-parallel-size 1 \
        --save-dir ./model_from_hf/Aquila-7B/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/Aquila-7B/mg2hg/
    ```

6. 配置 Aquila-7B 预训练脚本

    需要在预训练脚本中配置相关参数

    ```shell
    # 根据实际情况配置词表、数据集、模型参数保存路径
    TOKENIZER_PATH="./model_from_hf/Aquila-7B/"  #tokenizer 路径
    DATA_PATH="./dataset/Aquila-7B/alpaca_text_document"  #数据集 路径
    CKPT_LOAD_DIR="./model_weights/Aquila-7B-v0.1-tp8-pp1/"
    CKPT_SAVE_DIR="./ckpt/Aquila-7B/"
    # 如果不需要保存权重，就不需要设置CKPT_SAVE_DIR, 并且启动脚本里应不使用 `--save` 参数
    # 如果需要保存权重，则需要设置CKPT_SAVE_DIR, 并且启动脚本里应使用 `--save $CKPT_SAVE_DIR` 进行类似配置。
    
    # 如果不需要加载权重，就不需要设置CKPT_LOAD_DIR, 并且启动脚本里应不使用 `--load` 参数
    # 如果需要加载权重，则需要设置CKPT_LOAD_DIR, 并且启动脚本里应使用 `--load $CKPT_LOAD_DIR` 进行类似配置。
    # 进行断点续训时，应先按以上save的场景配置，待完成ckpt保存后，再修改相应参数，按以上load的场景加载已保存的ckpt。
    ```

    **注意**：如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据分布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据。

7. 启动 Aquila-7B 预训练脚本

    运行预训练脚本前，需先执行set_env.sh脚本以便设置环境参数，或者也可将其放入预训练脚本中执行。

    ```shell
    # 请按照您的真实环境修改 set_env.sh 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

    按以下方式启动Aquila-7B预训练：

    ```shell
    bash examples/aquila/pretrain_aquila_7b_ptd.sh
    ```



## 推理

我们支持使用 Aquila-7B进行文本生成的推理。

推理与预训练不同，我们必须加载预训练权重，请注意：在转换权重时使用的模型结构参数，和运行评估任务时使用的模型结构参数，应保持一致。

权重转换完成后，我们配置Aquila-7B推理脚本`example/aquila/generate_aquila_7b_ptd.sh`，需要正确指定加载权重的路径，词表路径等（下面样例仅供参考）

```shell
# 请按实际情况修改模型权重路径和分词器路径
CKPT_LOAD_DIR="./model_weights/Aquila-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Aquila-7B/"
```

启动Aquila-7B推理:

```shell
bash examples/aquila/generate_aquila_7b_ptd.sh
```

部分推理样例如下：

Aquila-7B:

![aquila-7B_generate.png](https://gitee.com/ascend/ModelLink/raw/master/sources/images/aquila/aquila_7B_generate_ptd_0205.png)

## 评估

我们使用 BoolQ benchmark 来评估我们的模型。在[Benchmark下载页面](https://github.com/google-research-datasets/boolean-questions)找到[数据集](https://storage.cloud.google.com/boolq/dev.jsonl)下载后保存。例如，保存在ModelLink/boolq/test目录下。

评估与推理类似，也必须加载转换后的权重，请注意：在转换权重时使用的模型结构参数，和运行评估任务时使用的模型结构参数，应保持一致。

权重转换完成后，我们配置Aquila-7B评估脚本 `examples/aquila/evaluate_aquila_7b_ptd.sh`，需要正确指定加载权重的路径，词表路径，评估数据的路径，以及评估任务的名字等(下面样例仅供参考)：

```shell
CKPT_LOAD_DIR="./model_weights/Aquila-7B-v0.1-tp8-pp1/"
TOKENIZER_PATH="./model_from_hf/Aquila-7B/"
EVAL_DATA_PATH="./boolq/test"
TASK="boolq"
```

启动Aquila-7B评估

```shell
bash examples/aquila/evaluate_aquila_7b_ptd.sh
```

Aquila-7B在**Ascend NPU**中的评测表现：

| 任务                                                                   | 模型       | 昇腾值|社区值|
|------------------------------------------------------------------------|------------|--------|------|
| [BoolQ](https://github.com/google-research-datasets/boolean-questions) | Aquila-7B  | 77.3% |      |

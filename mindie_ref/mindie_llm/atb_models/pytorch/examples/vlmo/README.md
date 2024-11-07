[TOC]

# Vlmo模型-推理指导

# 概述

VLMo 是由微软提出的一种多模态 Transformer 模型，Mixture-of-Modality-Experts (MOME)，即混合多模态专家。VLMo 相当于是一个混合专家 Transformer 模型。预训练完成后，使用时既可以是双塔结构实现高效的图像文本检索，又可以是单塔结构成为分类任务的多模态编码器。

- 参考实现：

  ```
  https://github.com/microsoft/unilm/tree/master/vlmo
  ```

# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

| 配套                 | 版本          | 下载链接 |
|--------------------|-------------|------|
| Ascend HDK         | 23.0.0.B060 |      |
| CANN               | 7.0.0.B060  |      |
| python             | 3.9.18      |      |           
| FrameworkPTAdapter | 5.0.0.B060  |      |

**表 2** 推理引擎依赖

| 软件    | 版本要求     |
|-------|----------|
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device |
|---------|--------|
| aarch64 | Atlas 800T A2  |


# 快速上手

## 获取源码及依赖

### 1. 环境部署

#### 1.1 安装HDK

先安装firmwire，再安装driver

##### 1.1.1 安装firmwire

安装方法: xxx代表具体版本

| 包名                                   |
|--------------------------------------|
| Ascend-hdk-{型号}-npu-firmware_xxx.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装firmwire
chmod +x Ascend-hdk-{型号}-npu-firmware_xxx.run
./Ascend-hdk-{型号}-npu-firmware_xxx.run --full
```

##### 1.1.2 安装driver

安装方法：

| cpu     | 包名                                               | 
|---------|--------------------------------------------------|
| aarch64 | Ascend-hdk-{型号}-npu-driver_xxx_linux-aarch64.run |
| x86     | Ascend-hdk-{型号}-npu-driver_xxx_linux-x86_64.run  |

```bash
# 根据CPU架构 以及npu型号 安装对应的 driver
chmod +x Ascend-hdk-{型号}-npu-driver_23.0.rc3.b060_*.run
./Ascend-hdk-{型号}-npu-driver_23.0.rc3.b060_*.run --full
```

#### 1.2 安装CANN

先安装toolkit 再安装kernel

##### 1.2.1 安装toolkit

安装方法：xxx代表具体的版本

| cpu     | 包名                                        |
|---------|-------------------------------------------|
| aarch64 | Ascend-cann-toolkit_xxx_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_xxx_linux-x86_64.run  |

```bash
# 安装toolkit  以arm为例
chmod +x Ascend-cann-toolkit_xxx_linux-aarch64.run
./Ascend-cann-toolkit_xxx_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

##### 1.2.2 安装kernel

安装方法：xxx代表具体的版本

| 包名                                     |
|----------------------------------------|
| Ascend-cann-kernels-{型号}_xxx_linux.run |

```bash
# 安装 kernel
chmod +x Ascend-cann-kernels-{型号}_xxx_linux.run
./Ascend-cann-kernels-{型号}_xxx_linux.run --install
```

#### 1.3 安装PytorchAdapter

首先安装torch，其次安装torch_npu，支持torch1.11.1、2.0.1，下面以torch2.0.1为例进行说明

##### 1.3.1 安装torch

安装方法：

| 包名                                              |
|-------------------------------------------------|
| torch-2.0.1+cpu-cp38-cp38-linux_x86_64.whl      |
| torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl      |
| torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl |
| torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl |
| ...                                             |

根据所使用python版本，以及CPU架构，选择对应的包

```bash
# 以安装torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl包为例
pip install torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl
```

##### 1.3.2 安装torch_npu

安装方法：

| 包名                         |
|----------------------------|
| pytorch_v2.0.1_py38.tar.gz |
| pytorch_v2.0.1_py39.tar.gz |
| ...                        |

选择安装与torch版本以及python版本一致的torch_npu版本

```bash
# 安装torch_npu，以torch2.0.1对应的python3.9的aarch64版本为例
tar -zxvf pytorch_v2.0.1_py39.tar.gz
pip install torch*_aarch64.whl
```

#### 1.3.3 requirements

| 包名               | 推荐版本   |  
|-----------------|--------|
| transformers     | 4.33.1 | 
| decorator        | 5.1.1  |
| sympy            | 1.11.1 |
| scipy            | 1.11.3 |
| attrs            | 23.1.0 |
| psutil           | 5.9.6  |
| sentencepiece    | 0.1.99 |
| pytorch_lightning| 1.5.5  |
| Pillow| 8.3.1 |
| tqdm |4.53.0|
| ipdb |0.13.7|
| einops| 0.3.0|
| pyarrow |14.0.1|
| sacred |0.8.2|
| pandas |2.2.0|
| timm |0.4.12|
| torchmetrics| 0.7.3|
| fairscale |0.4.0|
| numpy |1.26.4|
| scipy |1.12.0|
| opencv-python |4.9.0.80|
| opencv-python-headless| 4.5.3.56|
| psutil |5.9.8|




### 2. 安装依赖

#### 路径变量解释

| 变量名                 | 含义                                                                   |  
|---------------------|----------------------------------------------------------------------|
| model_download_path | 开源权重放置目录                                                             | 
| data_download_path|  数据集放置目录
| llm_path            | 加速库及模型库下载后放置目录                                                       |
| model_path          | 工作时模型所在的目录，可以和model_download_path相同，但一般模型是公共的，为了避免影响其他用户，单独建一个模型工作目录 |

#### 2.1 推理环境准备

1. 下载代码，通过git工具将vlmo代码下载至本地 `${model_path}` 中。
   ```
   git clone https://github.com/microsoft/unilm.git
   ```
   
2. 下载模型权重，放置到自定义`${model_download_path}` (请查看 README.md 下载链接中'Configs'页签下所需测试集的'finetuned weight')
   ```
   https://github.com/microsoft/unilm/tree/master/vlmo
   ```
   分类任务请使用 VQAv2数据集进行评估，检索任务情使用 COCO 数据集进行评估\
   以VQAv2为例，下载 vlmo_base_patch16_480_vqa.pt 将其放在 `${model_download_path}` 目录中。

3. 下载数据集，同上url下载相应数据集。(请查看 DATA.md 下载指定测试集的数据，并整理成所需目录结构)放置`${data_download_path}`目录\
   以VQAv2为例，将文件按照文档说明整理为如下格式：
   ```
      `${data_download_path}`
      ├── train2014            
      │   ├── COCO_train2014_000000000009.jpg                
      |   └── ...
      ├── val2014              
      |   ├── COCO_val2014_000000000042.jpg
      |   └── ...  
      ├── test2015              
      |   ├── COCO_test2015_000000000001.jpg
      |   └── ...         
      ├── v2_OpenEnded_mscoco_train2014_questions.json
      ├── v2_OpenEnded_mscoco_val2014_questions.json
      ├── v2_OpenEnded_mscoco_test2015_questions.json
      ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
      ├── v2_mscoco_train2014_annotations.json
      └── v2_mscoco_val2014_annotations.json
   ```
   在 `${model_path}`/unilm/vlmo 目录下新建文件 makearrow.py 内容如下：
   ```python
   from vlmo.utils.write_vqa import make_arrow
   make_arrow('{data_download_path}', '{data_download_path}/vqa_arrow')
   ```
   
   对于VQA v2数据集，vlmo的write_vqa脚本不会生成分类结果与答案的映射关系，需要在`${model_path}`/unilm/vlmo/vlmo/utils/write_vqa.py 中最下方手动添加代码进行输出。

   ```python
       # 注意行对齐
       with open(os.path.join(dataset_root, "answer2label.txt"), mode="w", encoding="utf-8") as writer:
            for ans in ans2label:
                to_json = {
                    "answer": ans, 
                    "label": ans2label[ans]
                }
                writer.write("%s\n" % json.dumps(to_json))
   ```

   执行该脚本，将会在 vqa_arrow文件夹下生成相应的二进制数据集文件：
   ```shell
   python makearrow.py
   ```
   生成目录结构如下：
   ```
      `${data_download_path}`W
        arrow
          ├── vqav2_val.arrow
          ├── vqav2_trainable_val.arrow
          ├── vqav2_train.arrow
          ├── vqav2_test.arrow
          ├── vqav2_test-dev.arrow
          ├── vqav2_test.arrow
          └── answer2label.txt
   ```


4. 下载Bert 词表
   ```
   https://huggingface.co/google-bert/bert-base-uncased/tree/main
   ```
   在Files and versions 页签中找到 vocab.txt 下载后放入 `${model_download_path}` 中备用。
   
5. 根据版本发布链接，安装加速库
   将加速库下载至 `${llm_path}` 目录

   | 加速库包名                                                 |
   |-------------------------------------------------------|
   | Ascend-cann-atb_{version}_cxx11abi0_linux-aarch64.run |
   | Ascend-cann-atb_{version}_cxx11abi1_linux-aarch64.run |
   | Ascend-cann-atb_{version}_cxx11abi1_linux-x86_64.run  |
   | Ascend-cann-atb_{version}_cxx11abi0_linux-x86_64.run  |

   具体使用cxx11abi0 还是cxx11abi1 可通过python命令查询

   ```python
   import torch
   torch.compiled_with_cxx11_abi()
   ```

   若返回True 则使用 cxx11abi1，否则相反。

   ```bash
   # 安装atb 
   chmod +x Ascend-cann-atb_*.run
   ./Ascend-cann-atb_*.run --install
   source /usr/local/Ascend/atb/set_env.sh
   ```

6. 根据版本发布链接，安装加速库
   将加速库下载至 `${llm_path}` 目录

   | 大模型包名                                                             |
   |-------------------------------------------------------------------|
   | Ascend-cann-llm_{version_id}_linux-x86_64_torch2.0.1-abi0.tar.gz  |
   | Ascend-cann-llm_{version_id}_linux-x86_64_torch2.0.1-abi1.tar.gz  |
   | Ascend-cann-llm_{version_id}_linux-aarch64_torch2.0.1-abi0.tar.gz |
   | Ascend-cann-llm_{version_id}_linux-aarch64_torch2.0.1-abi1.tar.gz |

   具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

   ```bash
   # 安装大模型加速库
   cd ${llm_path}
   tar -xzvf Ascend-cann-llm_*.tar.gz
   source set_env.sh
   ```


7. 设置环境变量

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   source /usr/local/Ascend/atb/set_env.sh
   source ${llm_path}/set_env.sh
   ```
   > 注： 每次运行前都需要 source CANN， 加速库，大模型

### 拷贝文件

### 准备

#### 1. 将大模型加速库中 vlmo 相关的 文件替换至 model_path 中的指定路径

```shell
cd ${llm_path}/atb_speed/pytorch/examples/models/vlmo/
cp multiway_transformer.py ${model_path}/unilm/vlmo/vlmo/modules
cp vlmo_module.py ${model_path}/unilm/vlmo/vlmo/modules
cp run_om_ascend_vqa.py ${model_path}/unilm/vlmo/vlmo/
cp run_om_ascend_vqa.sh ${model_path}/unilm/vlmo/vlmo/
```

#### 2.修改配置

以VQA v2 task_finetune_vqa_base_image480 微调评估为例。\
打开 `${model_path}`/unilm/vlmo/run_om_ascend_vqa.sh \
修改 `<Finetuned_VLMo_WEIGHT>`  为 `${model_download_path}` ；修改 `<CONFIG_NAME>` 为 task_finetune_vqa_base_image480

打开 `${model_path}`/unilm/vlmo/run_om_ascend_vqa.py \
修改 `VQA_ARROW_DIR`  路径为 '`${data_download_path}`/vqa_arrow' ；修改 `<BERT_VOCAB>` 为 '`${model_download_path}`/vocab.txt'
 
# CPU高性能模式

可开启CPU Performance模式以提高模型推理性能。

```
cpupower frequency-set -g performance
```

### 执行推理

#### run_inf_ascend_vqa.sh

用于执行已经基于VQA v2 数据集微调好的权重，执行图片分类任务。输入为一张图片以及一个问题，推理结果为一个特征值，通过分类器可将其从3129个备选答案中选出一个结果。

```shell
bash run_inf_ascend_vqa
```

#### FAQ


1. ImportError: /root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block  

如果遇到

```text
Traceback (most recent call last):
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/__init__.py", line 31, in <module>
    import torch_npu.npu
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/npu/__init__.py", line 46, in <module>
    from .utils import (is_initialized, _lazy_call, _lazy_init, init, set_dump,
  File "/root/miniconda3/envs/wqh39/lib/python3.9/site-packages/torch_npu/npu/utils.py", line 27, in <module>
    import torch_npu._C
ImportError: /root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1: cannot allocate memory in static TLS block
Segmentation fault (core dumped)
```

则可取消run_inf_ascend_*.sh 脚本中的注释，修改为报错中相应的路径。如

```shell
LD_PRELOAD=/root/miniconda3/envs/wqh39/bin/../lib/libgomp.so.1:$LD_PRELOAD
```

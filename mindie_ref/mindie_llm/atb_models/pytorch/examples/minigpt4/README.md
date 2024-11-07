# MiniGPT-4

## 概述

MiniGPT-4使用先进的大型语言模型增强视觉语言理解，将语言能力与图像能力结合。其利用视觉编码器BLIP-2和大语言模型Vicuna进行结合训练，共同提供了新兴视觉语言能力。

- 参考实现：

  ```
   https://github.com/Vision-CAIR/MiniGPT-4
  ```

## 环境准备

- 该模型需要以下插件与驱动

  **表 1** 版本配套表

| 配套                 | 版本          | 下载链接 |
|--------------------|-------------|------|
| Ascend HDK         | 23.0.0.B070 | -    |
| CANN               | 7.0.0.B070  | -    |
| python             | 3.9.18      | -    |
| FrameworkPTAdapter | 5.0.0.B070  | -    |

**表 2** 推理引擎依赖

| 软件    | 版本要求     |
|-------|----------|
| glibc | >= 2.27  |
| gcc   | >= 7.5.0 |

**表 3** 硬件形态

| CPU     | Device |
|---------|--------|
| aarch64 | Atlas 800T A2  |

### 安装NPU环境

#### 安装HDK

先安装firmwire，再安装driver

##### 安装firmwire

安装方法:

| 包名                                             |
|------------------------------------------------|
| Ascend-hdk-{型号}-npu-firmware_7.0.t9.0.b221.run |

根据芯片型号选择相应的安装包安装

```bash
# 安装firmwire
chmod +x Ascend-hdk-{型号}-npu-firmware_7.0.t9.0.b221.run
./Ascend-hdk-{型号}-npu-firmware_7.0.t9.0.b221.run --full
```

##### 安装driver

安装方法：

| cpu     | 包名                                                         |
|---------|------------------------------------------------------------|
| aarch64 | Ascend-hdk-{型号}-npu-driver_23.0.rc3.b060_linux-aarch64.run |
| x86     | Ascend-hdk-{型号}-npu-driver_23.0.rc3.b060_linux-x86_64.run  |

```bash
# 根据CPU架构 以及npu型号 安装对应的 driver
chmod +x Ascend-hdk-{型号}-npu-driver_23.0.rc3.b060_*.run
./Ascend-hdk-{型号}-npu-driver_23.0.rc3.b060_*.run --full
```

#### 安装CANN

先安装toolkit 再安装kernel

##### 安装toolkit

安装方法：

| cpu     | 包名                                            |
|---------|-----------------------------------------------|
| aarch64 | Ascend-cann-toolkit_7.0.T10_linux-aarch64.run |
| x86     | Ascend-cann-toolkit_7.0.T10_linux-x86_64.run  |

```bash
# 安装toolkit  以arm为例
chmod +x Ascend-cann-toolkit_7.0.T10_linux-aarch64.run
./Ascend-cann-toolkit_7.0.T10_linux-aarch64.run --install
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

##### 安装kernel

安装方法：

| 包名                                         |
|--------------------------------------------|
| Ascend-cann-kernels-{型号}_7.0.T10_linux.run |

```bash
# 安装 kernel
chmod +x Ascend-cann-kernels-{型号}_7.0.T10_linux.run
./Ascend-cann-kernels-{型号}_7.0.T10_linux.run --install
```

#### 安装PytorchAdapter

先安装torch 再安装torch_npu

##### 安装torch

安装方法：

| 包名                                           |
|----------------------------------------------|
| torch-2.0.1+cpu-cp38-cp38-linux_x86_64.whl   |
| torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl   |
| torch-2.0.1+cpu-cp310-cp310-linux_x86_64.whl |
| torch-2.0.1-cp310-cp310-linux_aarch64.whl    |
| torch-2.0.1-cp38-cp38-linux_aarch64.whl      |
| torch-2.0.1-cp39-cp39-linux_aarch64.whl      |
| ...                                          |

根据所使用的环境中的python版本以及cpu类型，选择torch-2.0.1相应的安装包。

```bash
# 安装torch 2.0.1 的python 3.9 的arm版本为例
pip install torch-2.0.1-cp39-cp39-linux_aarch64.whl
```

##### 安装torch_npu

安装方法：

| 包名                          |
|-----------------------------|
| pytorch_v2.0.1_py38.tar.gz  |
| pytorch_v2.0.1_py39.tar.gz  |
| pytorch_v2.0.1_py310.tar.gz |
| ...                         |

- 安装选择与torch版本 以及 python版本 一致的npu_torch版本

```bash
# 安装 torch_npu 以torch 2.0.1 的python 3.9的版本为例
tar -zxvf pytorch_v2.0.1_py39.tar.gz
pip install torch*_aarch64.whl
```

#### requirements

|          包名           |    推荐版本    |
|:---------------------:|:----------:|
|         torch         |   2.0.1    |
|      torchaudio       |   2.0.1    |
|      torchvision      |   0.15.1   |
|    huggingface-hub    |   0.18.0   |
|      matplotlib       |   3.7.0    |
|        psutil         |   5.9.4    |
|        iopath         |   0.1.10   |
|        pyyaml         |    6.0     |
|         regex         | 2022.10.31 |
|      tokenizers       |   0.13.2   |
|         tqdm          |   4.64.1   |
|     transformers      |   4.30.0   |
|         timm          |   0.6.13   |
|      webdataset       |   0.2.48   |
|       omegaconf       |   2.3.0    |
|     opencv-python     |  4.7.0.72  |
| sentence-transformers |   2.2.2    |
|      accelerate       |   0.20.3   |
|     scikit-image      |   0.22.0   |
|     visual-genome     |   1.1.1    |
|         wandb         |   0.16.1   |
|         attrs         |   23.1.0   |
|       decorator       |   5.1.1    |

另外，使用om格式模型进行推理需要aclruntime和ais_bench两个三方库，可参考https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench 中的工具安装方式进行安装。

### 推理环境准备

根据版本发布链接，安装加速库，将加速库下载至 `${llm_path}` 目录

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

若返回 True 则使用 cxx11abi1，否则相反。

```bash
# 安装atb 
chmod +x Ascend-cann-atb_*.run
./Ascend-cann-atb_*.run --install
source /usr/local/Ascend/atb/set_env.sh
```

根据版本发布链接，下载模型仓至 `${llm_path}` 目录

| 大模型包名                                                                     |
|---------------------------------------------------------------------------|
| Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi1.tar.gz  |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz |
| Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi1.tar.gz |

具体使用cxx11abi0 还是cxx11abi1 方法同安装atb

 ```bash
 # 安装模型仓
 cd ${llm_path}
 tar -xzvf Ascend-cann-llm_*.tar.gz
 source set_env.sh
 ```

### 文件下载和配置

#### 文件下载
1. 下载MiniGPT-4源码，下载地址为： https://github.com/Vision-CAIR/MiniGPT-4 ，将源码保存在 `${work_space}` 路径下。

2. 下载Vicuna-7b的模型权重，下载地址：https://hf-mirror.com//Vision-CAIR/vicuna-7b/tree/main
。下载完成后，保存在路径：`${model_path}/weights/`.

3. 下载好Vicuna-7b模型权重后，在配置文件将`llama_model`参数的值设置为Vicuna-7b权重的路径，配置文件路径为：
`${work_space}/minigpt4/configs/models/minigpt4_vicuna0.yaml`，参考配置`llama_model: "${model_path}/weights/"`.

4. 下载MiniGPT-4模型的预训练checkpoint，用于模型推理。下载地址：https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing ，
文件名为`prerained_minigpt4_7b.pth`，保存在路径`${model_path}/pretrain/`.

5. 在配置文件中将`ckpt`参数的值设置为checkpoint文件所在路径, 配置文件路径为：`${work_space}/eval_configs/minigpt4_eval.yaml`
参考配置`ckpt: "${model_path}/pretrain/prerained_minigpt4_7b.pth"`.

6. 下载图像处理相关模型VIT(eva_vit_g.pth)及Qformer(blip2_pretrained_flant5xxl.pth)，下载地址分别是：https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
，https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth
下载完成后保存在路径：`${model_path}/othfiles/`. 下载Bert(bert-base-uncased)的Tokenizer：https://hf-mirror.com//bert-base-uncased
同样保存在路径：`${model_path}/othfiles/`. 全部下载完成后`${model_path}/othfiles/`文件夹内所有文件如下：

```bash
> ls -al /data/model/MiniGPT-4/othfiles --block-size=K
> total 2401128K
> drwxr-xr-x 2 root root 4K May 5 02:09 .
> drwxr-xr-x 3 root root 4K May 7 02:34 ..
> drw------- 2 root root 4K Dec 1 14:31 bert-base-uncased
> -rw------- 1 root root 423322K May 5 02:09 blip2_pretrained_flant5xxl.pth
> -rw------- 1 root root 1977783K May 5 02:08 eva_vit_g.pth
```

bert-base-uncased文件夹内文件清单如下：

```bash
> ls -al bert-base-uncased --block-size=K                
total 244K
drwxr-xr-x 2 root root   4K May  7 09:03 .
drwxrwxrwx 9 root root   4K May  7 09:02 ..
-rw-r--r-- 1 root root   1K May  7 09:03 config.json
-rw-r--r-- 1 root root   1K May  7 09:03 tokenizer_config.json
-rw-r--r-- 1 root root 227K May  7 09:03 vocab.txt
```

修改eva_vit.py文件的相关配置（用于om模型转换），路径为 `${work_space}/minigpt4/models/eva_vit.py`

```bash
state_dict = torch.load("${model_path}/othfiles/eva_vit_g.pth", map_location="cpu")
```

修改minigpt4.py文件的相关配置，路径为 `${work_space}/minigpt4/models/minigpt4.py`

```bash
q_former_model = "${model_path}/othfiles/blip2_pretrained_flant5xxl.pth",
```

```bash
encoder_config = BertConfig.from_pretrained("${model_path}/othfiles/bert-base-uncased")
```

```bash
q_former_model = cfg.get("q_former_model", "${model_path}/othfiles/blip2_pretrained_flant5xxl.pth")",
```
修改eva_vit_model.py文件的相关配置，路径为 `${work_space}/minigpt4/models/eva_vit_model.py`

```bash
encoder_config = BertConfig.from_pretrained("${model_path}/othfiles/bert-base-uncased")
```

模型推理需要三个类型的外部文件：分别为原始模型文件、图像部分离线模型eva_vit_g.om以及测试图片。

1. 模型相关文件可在huggingface官网下载：https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main

2. eva_vit_g.om为将minigpt4中图像模型eva_vit_g.pth的权重导出转换为适合NPU使用的om模型，具体的转换方式见本文件底部附录。

3. 测试图片可使用源代码仓的example图片，测试过程使用了下面链接中的三张图片：

https://github.com/Vision-CAIR/MiniGPT-4/blob/main/examples_v2/2000x1372_wmkn_0012149409555.jpg

https://github.com/Vision-CAIR/MiniGPT-4/blob/main/examples_v2/KFC-20-for-20-Nuggets.jpg

https://github.com/Vision-CAIR/MiniGPT-4/blob/main/examples_v2/office.jpg

下载图片后存放到`${image_path}`目录下.

### 模型推理

1. 将models目录下的全部文件拷贝到下载好的MiniGPT-4源码 `${work_space}/minigpt4/models` 目录下；将 onnx_model_export.py 和 run_predict.py拷贝到 `${work_space}` 目录中。

2. 根据附录中代码修改清单，修改对应的代码配置，保证模型推理功能正常。

3. 根据附录中om模型构造过程部分将图像处理模型转换为离线模型eva_vit_g.om文件，将生成的om文件存放在 `${om_model_path}` 目录下。

4. 安装ais_bench和aclruntime包，参考 https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench ，保证om模型可正常推理。

5. 打开下载好的模型仓，进入`ModelLink\mindie_ref\mindie_llm\atb_models\pytorch\examples\atb_speed_sdk`目录执行`pip install .`，安装SDK

6. 完成上述步骤后，即可进行模型推理任务。在 `${work_space}` 目录下，执行如下命令：

`python run_predict.py --cfg-path eval_configs/minigpt4_eval.yaml --image-path ${image_path}/office.jpg  --npu-id ${npu-id}`

## 测试

### 精度测试

在 `${work_space}` 目录下执行以下脚本：

`python run_predict.py --cfg-path eval_configs/minigpt4_eval.yaml --image-path ${image_path}/office.jpg  --npu-id ${npu-id}`

执行完成后模型的回答会打印在终端。


### 图像处理时间测试

将图像处理部分转换为OM模型后，图像处理时间约为0.018s；GPU图像处理时间约为1.185s

### 性能测试

在功能运行正常的基础上，执行以下步骤进行性能测试。

#### 1. 替换transformer库中的utils文件

以transformers 4.30.2为例，需要替换transformer库的原生utils文件，来执行性能测试。

执行如下命令，找到需要被替换的文件所在的路径。

`python -c "import os;import transformers;print(os.path.join(os.path.dirname(transformers.__file__), 'generation'))"`

可以得到：/root/miniconda3/envs/cytest/lib/python3.8/site-packages/transformers/generation

进入上面得到的目录下做如下操作：

（1）将utils.py文件备份一份，命名为utils_ori.py

（2）将pytorch/examples/atb_speed_sdk/atb_speed/common/transformers_patch/4.30.2/utils_performance_test_npu_greedy.py拷贝到当前目录下，并重命名为utils.py

#### 2. 替换权重路径下的modeling文件
1. 将pytorch/examples/minigpt4/models/modeling_vicuna_ascend_performance.py拷贝到权重目录`${model_path}/weights/`下
2. 修改config.json , 将第5行 `"bos_token_id": 1,` 前的内容修改如下:

   ```json
   "auto_map": {
      "AutoModelForCausalLM":"modeling_vicuna_ascend_performance.LlamaForCausalLM"
   },
   ```
#### 3. 修改性能测试脚本内容
将`run_performance.py`的第188行修改为对应的输入输出长度：`temp = [[${input_length}, ${output_length}]]`

#### 4. 执行性能测试

 ```bash
  ATB_CONTEXT_TILING_RING=1 ATB_LAYER_INTERNAL_TENSOR_REUSE=1 PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048" python run_performance.py --model_path ${model_path} --device_id ${device_id}] --batch ${batch_size}
 ```

为了不影响正常使用，将RETURN_PERF_DETAIL设置成1来返回具体的性能测试的值，默认是0

测试了batch_size为1，输入输出长度分别为[[256, 64], [512, 128], [1024, 256], [3584, 512]]，输出 `multi_batch_performance.csv`

NPU性能测试结果如下：

| batch_size | input_length | output_length | reponse_time(ms) | 首token耗时（ms） | 非首token平均耗时（ms） | E2E吞吐（token/s） |
|:----------:|:------------:|:-------------:|:----------------:|:---------------------:|:-------------------:|:--------------:|
|     1      |     256      |      64       | 847.203969955444 |   5.98406791687011    |  13.3526968577551   |  75.54261107   |
|     1      |     512      |      128      | 1736.02199554443 |   11.7170810699462    |  13.5772040509802   |  73.73178469   |
|     1      |     1024     |      256      | 3599.91073608398 |   35.1405143737792    |  13.9794910655302   |  71.11287439   |
|     1      |     3584     |      512      | 8483.46471786499 |   344.723224639892    |  15.9270870708906   |  60.35269987   |


### 结果验证

加速库代码与GPU版本代码精度的对比，因为MiniGPT-4 GPU的实现没有数据集的精度测试，精度验证使用了github源代码仓的example图片，进行图文问答并比较问答结果。

#### 结果展示

图片 + 问题：Describe this image in detail.

使用三张图片进行测试，每次实验随机生成的文本不尽相同。

GPU输出结果：

1. The image shows a group of men holding up the World Cup trophy while standing in front of a crowded stadium. The
   players are wearing blue and white jerseys and have their hands raised in the air, holding the trophy up to the
   camera. In the background, there is a large crowd of people cheering and waving flags. There are also fireworks going
   off in the sky, adding to the celebratory atmosphere. The image was taken during a soccer match where one team won
   the World Cup trophy.
2. This image shows a person holding two pieces of fried chicken in their hand, which are placed inside a cardboard
   container with the logo "20 for $10" on it. The container is placed on a table, and there are various dipping sauces
   visible in the background.
   The image depicts an individual enjoying a meal of fried chicken at a low price. The cardboard container suggests
   that the food is being offered as a bargain or deal. The dipping sauces add flavor to the chicken, making it more
   enjoyable to eat. Overall, the image suggests affordability and deliciousness.
3. This image shows a group of men holding up a trophy, all wearing the same blue and white striped jerseys. They are
   standing on a field with fireworks in the background. Some of the players are raising their arms in the air, while
   others are holding the trophy with both hands. The team's logo is on the front of the jersey, and they all have
   smiles on their faces. It seems to be a celebratory moment, with the players and crowd sharing in the joy of winning
   the trophy.

NPU输出结果：

1. This image shows the Argentina soccer team holding up the World Cup trophy in celebration. The team is dressed in
   blue and white jerseys, with some players holding up their arms in the air while others hold up the trophy. Fireworks
   are exploding behind them, creating a colorful display of light and smoke.
2. The image is a box of fried dough balls, or tempura, sitting on a table next to a bowl of sauce. The box has the logo
   for KFC written on it in red and white letters. There are four balls of tempura in the box, each one a different
   color. One of the balls has been taken out and is being held by a hand in the foreground.
   The background of the image is a plain white with a few shadows and reflections visible from the lighting. The plate
   and
   sauce are also plain white. The colors used in the tempura add visual interest to the image, making it more dynamic
   and
   appetizing.

3. The image shows a man in a business suit sitting at a desk with a laptop in front of him. He is looking down at the
   screen and seems to be typing on the keyboard. There are several other computers and pieces of office equipment
   visible in the background, including a printer and a phone. The room is brightly lit, and there are shadows on the
   wall behind the man's head. The image has a professional and modern feel to it.

## 附录

### om模型构造过程

适用情况：需要拆分出一部分模型在NPU上做离线推理;

整体过程分为两步，第一步使用torch.onnx.export把需要转换的模型部分转换为onnx模型，第二步使用昇腾ATC工具将onnx转换为om.

#### ONNX转换

1. MiniGPT-4为多模态模型，其中图像部分每次推理时仅使用一次，比较适合转换为离线模型;

2. 首先识别图像部分代码进行分离；即原始代码中minigpt4.py的第125行的 `image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)` 及其配套代码;

3. 将图像部分逻辑分离后整合为一个单独的模型，详情见eva_vit_model.py;

4. 使用torch.onnx.export将该部分模型与权重转换为onnx，详情见onnx_model_export.py. 运行该文件，可生成对应的onnx中间模型。
参考运行命令: `python onnx_model_export.py --onnx-model-dir /data/model/MiniGPT-4/onnx_model --image-path ../test_image/01.jpg`

#### OM转换

OM模型转换使用昇腾ATC工具，使用流程参考该链接https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000005.html

1. 环境准备：安装并source CANN包；可参考上述链接中环境搭建的部分；

2. 模型转换：参考快速入门中ONNX网络模型转换成离线模型章节

注：om模型转换时，要进入到已转换好的onnx模型目录中执行转成om模型的命令，否则会找不到权重文件。

### 代码修改清单

由于minigpt4是为多模态模型，比其他语言模型多了图像部分，且源码较复杂，需对代码做出如下修改：

1. `${work_space}/minigpt4/models/base_model.py` 文件，具体修改如下：

（1）删除训练部分需用到的三方件引入

```python
from peft import (
   LoraConfig,
   get_peft_model,
   prepare_model_for_int8_training,
)
```
（2）modeling文件导入修改为已适配加速库的新加速库modeling文件
```python
from minigpt4.models.modeling_llama import LlamaForCausalLM
```
替换为
```python
from minigpt4.models.modeling_vicuna_ascend import LlamaForCausalLM
```

2. `${work_space}/minigpt4/models/minigpt_base.py` 文件，具体修改如下：

（1）在文件头导入图像OM模型推理类
```python
from minigpt4.models.image_encoder import IMAGE_ENCODER_OM
```
（2）在40行新增如下代码，初始化加载om模型
```python
self.image_encoder = IMAGE_ENCODER_OM("${om_model_path}/", device_8bit)
```
(3) 注释或删除原始图像处理部分代码
```python
self.visual_encoder, self.ln_vision = self.init_vision_encoder(
    vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit
)
```

3. `${work_space}/minigpt4/models/minigpt4.py` 文件，具体修改如下：

（1）在文件头导入om模型推理类
```python
from ais_bench.infer.interface import InferSession
```
（2）原文件第63行和70行，将 `self.visual_encoder.num_features` 修改为 VisionTransformer 类入参embed_dim的固定值1408.
```python
self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408, freeze_qformer)
```

```python
img_f_dim = 1408 * 4
```
（3）原文件第125行，图像embedding的计算不再走原始逻辑，而使用转换后的om模型计算
```python
image_embeds = torch.tensor(self.image_encoder.image_encoder_om.infer(image.cpu().numpy())[0]).to(device)
```

4. `${work_space}/minigpt4/datasets/data_utils.py` 文件，具体修改如下：

（1）删除原文件18、19行
```python
import decord
from decord import VideoReader
```
（2）删除原文件29行
```python
decord.bridge.set_bridge("torch")
```
5. 由于无法使用CUDA的8位优化器，需将 `${work_space}/eval_configs/minigpt4_eval.yaml` 中 `low_resource` 参数值设置为False。

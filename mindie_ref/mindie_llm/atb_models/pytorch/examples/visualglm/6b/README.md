# VISUALGLM-6B

## 概述
VisualGLM-6B 是一个开源的，支持图像、中文和英文的多模态对话语言模型，语言模型基于 ChatGLM-6B，具有 62 亿参数；图像部分通过训练 BLIP2-Qformer 构建起视觉模型与语言模型的桥梁，整体模型共78亿参数。

- 参考实现：

  ```
   https://github.com/THUDM/VisualGLM-6B
  ```

## 环境准备

### 需求文件下载和配置

#### 文件下载

模型需要三个类型的外部文件：分别为原始模型文件，测试图片，以及blip2.om图像部分离线模型。  

1. 模型相关文件也可在上述huggingface官网下载：https://huggingface.co/THUDM/visualglm-6b/blob/main/ice_text.model  
   其中模型权重下载速度较慢，可使用清华大学云盘下载：https://cloud.tsinghua.edu.cn/d/43ffb021ca5f4897b56a/  

2. 测试图片存放在examples文件夹可在模型github官网下载： https://github.com/THUDM/VisualGLM-6B/tree/main/examples  

3. blip2.om为将visualglm中图像模型blip2的权重导出转换为NPU使用的om模型，具体的转换方式见本文件底部附录。  


### 安装NPU环境

1. 根据机器型号与python版本依次安装Ascend驱动，CANN包，并在python环境中装入torch_npu；  

2. 引入CANN包，将CANN包路径改为自己安装的路径即可。命令如： source /usr/local/Ascend/ascend-toolkit/set_env.sh    


### Python环境准备

1. 基础需求：参考https://github.com/THUDM/VisualGLM-6B 中的requirements.txt进行安装;  

2. om模型需求： aclruntime、ais_bench两个三方库，可参考https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench中的工具安装方式进行安装。  


### 编译

编译ascend-transformer-boost与ascend-speed-inference代码仓并设置环境变量；下载完代码仓只需编译一次，每次重新进入环境均需要设置环境变量：  

1. cd ascend-transformer-boost     编译： bash scripts/build.sh   
设置环境变量 cd output/atb    source set_env.sh  

2. cd ascend-speed-inference       编译： bash scripts/build.sh   
设置环境变量 cd output/atb_speed    source set_env.sh  


## 软硬件版本

### 硬件版本
| Hardware |                      Value                      |
| :------: | :---------------------------------------------: |
|   CPU    |                464 Core, 256CPUS                |
|   RAM    |                  32x64 GB DDR4                  |
|   NPU    |                 8 x Ascend910 64G               |


### 软件版本

|         Software          |                 Version                 |
| :-----------------------: | :-------------------------------------: |
|            OS             |        ubuntu15.4.0-163-generic         |
|           uname           |                 aarch64                 |
|          Python           |                  3.8.10                 |
|          driver           |              23.0.RC3.b070              |
|         firmware          |              23.0.RC3.b070              |
|           CANN            |                 7.0.RC1                 |
|           torch           |                 1.11.0                  |
|         torch_npu         |          1.11.0.post4-20230927          |
|       transformers        |                 4.30.2                  |


## 测试

### 主入口：main.py

1. --mode参数选择运行模式，run|performance|precision|predict|；
对应七类任务：自由对话 | 模型性能测试 | 模型精度测试 | 模型结果验证

2. --model_path是模型文件路径，上述所有外部文件均存放在此路径；  

3. --device_id选择使用的NPU卡号。


### 图像处理时间测试

visualglm模型的图像处理在最前方完成，每轮对话可能有一张图片或者没有图片，本模型的每个脚本均会输出picture process time；

将图像部分转换为OM模型后，图像处理时间约为0.024-0.03s；GPU图像处理时间首轮为1-2s，非首轮约为0.05-0.06s。


### 性能测试

 ```bash
  python3 main.py --mode performance --model_path path --device id
  ```
测试了batch_size:1, 2, 4, 8，输入输出token长度分别为32,64,128,256,512,1024的模型用时，输出{batch_level}_batch_performance_visualglm.csv

1. NPU各batch性能测试结果平均值如下：

|  Batch   |   TokensPerSecond   | ResponseTime(ms) | FirstTokenTime(ms) | TimePerTokens(ms) |
| :------: | :----------------:  | :--------------: | :----------------: |:-----------------:|
|    1     |        59.91        |      5700.1      |        54.25       |       16.73       |
|    2     |        54.95        |      6269.79     |        83.32       |       18.21       |
|    4     |        51.34        |      6820.75     |        157.42      |       19.53       |
|    8     |        42.94        |      16791.57    |        8716.3      |       23.37       |

2. GPU各batch性能测试结果平均值如下：

|  Batch   |   TokensPerSecond   | ResponseTime(ms) | FirstTokenTime(ms) | TimePerTokens(ms) |
| :------: | :----------------:  | :--------------: | :----------------: | :---------------: |
|    1     |        29.47        |      11500.52    |        80.14       |      34.03        |
|    2     |        28.94        |      11708.85    |        165.48      |      35.8         |
|    4     |        26.22        |      13219.19    |        218.29      |      38.49        |
|    8     |        25.65        |      15546.87    |        398.45      |      40.43        |


### 内存测试

对于代码占用的峰值显存进行测试，输入输出为均1024的情况下，使用上述性能测试脚本的同时进行利用npu-smi info得出显存情况(MB)；

|  Batch   |    GPU    |    NPU    |
| :------: | :------:  | :-------: |
|    1     |   19000   |   20500   |


### 精度测试

```bash
  python3 main.py --mode precision --model_path path --device id
  ```

加速库代码与对应的torch_npu代码精度进行对比，每对token的余弦相似度均为0.9999以上。  

### 结果验证

```bash
  python3 main.py --mode predict --model_path path --device id
  ```

加速库代码与GPU版本代码精度的对比，因为VisualGLM GPU的实现没有数据集的精度测试，我们的精度验证使用github源代码仓的example图片，进行图文问答，比较问答结果。

#### 结果展示

图片按照examples中的序号，每次实验随机生成的文本不尽相同。  
问题1 + 图片1：描述一下这个场景,这个场景是一部电影中的经典桥段  
问题2 + 图片2：这是什么东西  
问题3 + 图片3：这张图片描述了什么  

GPU输出结果,：  
1. 泰坦尼克号上，杰克和露丝拥抱在船头。他们的目光似乎沉浸在彼此的世界中，仿佛时间已经停滞了，
他们的身体紧紧相连，似乎能够感受到对方的温度。这一刻，爱情被诠释的如此美好而真实。  
2. 这张图片展示了一只可爱的卡通羊驼，它正站在一片白色的背景上。这只羊驼长着一张毛茸茸的棕色脸和一双蓝色的眼睛。它的耳朵是直立的，尾巴则呈扇形展开。它还戴着一个绿色的项圈，上面系着一条红色的带子。  
3. 这张照片描绘了一直戴着眼镜、系着领带的狗，它坐在木栅栏后面。这只狗看起来很可爱和有趣。
它的眼睛很明亮，看起来很高兴或好奇。照片捕捉到了狗狗的迷人微笑和友好的态度。
这种场景通常被用于捕捉动物与人类之间的互动和情感联系。

NPU输出结果：   
1. 这张照片展示了一对年轻夫妇在船上拥抱。这对年轻夫妇的拥抱姿势很经典和浪漫。他们一起站在船头，手拉着手，
看起来非常幸福、放松和享受彼此的存在。这个场景是电影《泰坦尼克号》中经典的一个镜头，两人相依相扶地坐在船头，
似乎正在分享他们的快乐或痛苦。  
2. 这张图片展示了一只可爱的白色羊驼，它正站在一个透明的背景上。这只羊驼有着白色的毛皮和棕色的毛发。
这只羊驼正在跳跃或奔跑。这只羊驼戴着帽子，穿着裤子。它的眼睛是绿色的，嘴巴是红色的。
这只羊驼站立在透明的背景上，两只脚呈弯曲状，尾巴则向上翘起来。  
3.  这张照片描绘了一只白色的狗，戴着眼镜和领带。这只狗看起来很时尚，穿着时髦的装扮，比如戴眼镜或领带。
这个场景展示了一种有趣的、有创意的氛围。这种服装为狗狗创造了一个独特的形象，使照片具有吸引力和趣味性。  


## 附录  

### om模型构造过程

适用情况：需要拆分出一部分模型在NPU上做离线推理；  

整体过程分为两步，第一步使用torch.onnx.export把需要转换的模型部分转换为onnx模型，第二步使用昇腾ATC工具将onnx转换为om。

#### ONNX转换

```bash
  cp utils/blip2_model.py $MODEL_PATH/modeling_chatglm.py
  python3 utils/blip2_transfor.py --model_path $MODEL_PATH
  ```

该过程需要运行在GPU环境torch==1.13.0中，因为torch1.13.0以下以及部分测试的高版本torch不支持onnx转换中的某些算子，cpu不支持float16的某些算子，npu没有适用的torch==1.13.0版本；

1. visualglm为多模态模型，其中图像部分每次推理最多使用一次，比较适合转换为离线模型；  

2. 首先识别图像部分代码进行分离；即原始代码中modeling_chatglm.py的第1464行 image_embeds = self.image_encoder(images) 及其配套代码；

3. 将图像部分逻辑分离后整合为一个单独的模型，详情见picture_model.py;

4. 使用torch.onnx.export将该部分模型与权重转换为onnx，详情见picture_model_transfer.py，注意运行此脚本需要更改config.json中AutoModel与AutoModelForSeq2SeqLM开头部分的modeling_target为picture_model。


#### OM转换

```bash
  atc --model=$MODEL_PATH/transfer_model/blip2.onnx --framework=5 --output=$MODEL_PATH/blip2 --soc_version={soc-version} --input_shape="input:1,224,224,3"
  ```
  
该过程需要运行在NPU环境中；

OM模型转换使用昇腾ATC工具，使用流程参考该链接https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000005.html 

1. 环境准备：安装并source CANN包；可参考上述链接中环境搭建的部分；   

2. 模型转换：参考快速入门中ONNX网络模型转换成离线模型章节。  

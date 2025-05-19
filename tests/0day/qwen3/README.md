## 0day提供Qwen3系列模型同步首发支持

`认证`【Pass】表示经过昇腾官方版本测试的模型，【Test】表示待测试模型

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>魔乐社区链接</th>
      <th>脚本位置</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3-dense</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base">0.6B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-0.6B-Base">0.6B</a></td>
      <td><a href="./qwen3-0.6b/">Qwen3-0.6B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-1.7B-Base">1.7B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-1.7B-Base">1.7B</a></td>
      <td><a href="./qwen3-1.7b/">Qwen3-1.7B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-4B-Base">4B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-4B-Base">4B</a></td>
      <td><a href="./qwen3-4b/">Qwen3-4B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-8B-Base">8B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-8B-Base">8B</a></td>
      <td><a href="./qwen3-8b/">Qwen3-8B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-14B-Base">14B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-14B-Base">14B</a></td>
      <td><a href="./qwen3-14b/">Qwen3-14B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">32B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-32B">32B</a></td>
      <td><a href="./qwen3-32b/">Qwen3-32B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3-moe</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Base">30B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-30B-A3B-Base">30B</a></td>
      <td><a href="./qwen3-30b-a3b/">Qwen3-30B-A3B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B">235B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-235B-A22B">235B</a></td>
      <td><a href="./qwen3-235b-a22b/">Qwen3-235B-A22B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 16x16 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>


## MindSpeed跑通Qwen3模型手把手教程：快速部署与应用
本教程将指导从环境准备到模型训练的完整过程，确保快速上手并顺利跑通Qwen3模型。下文示例qwen3-1.7B的运行步骤，以下是详细说明：


###  目录

- [环境配置](#环境配置)
  - [硬件要求](#硬件要求)
  - [环境搭建](#环境搭建)
- [权重转换](##权重转换)
- [数据预处理](##数据预处理)
- [模型训练](##训练)
- [推理](##推理)
- [评估](##评估)



# 环境配置

## 硬件要求


|类型|  硬件 |       配置        |
|:---:|:---:|:---------------:|
|预训练| NPU | 8 x Ascend NPUs |

## 环境搭建

MindSpeed-LLM的主要依赖配套如下表，安装步骤参考[安装指导](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/features/install_guide.md)。

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">商发版本</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>CANN Toolkit（开发套件）</td>
      <td rowspan="3">商发版本</td>
  </tr>
  <tr>
    <td>CANN Kernel（算子包）</td>
  </tr>
  <tr>
    <td>CANN NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>>=3.10</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>2.1.0</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
    <td >2.1.0</td>
  </tr>
  <tr>
    <td>apex</td>
    <td >商发版本</td>
  </tr>
  <tr>
    <td>transformers</td>
    <td >4.51.3</td>
  </tr>
</table>


#### 1. 仓库拉取

```shell
git clone https://gitee.com/ascend/MindSpeed-LLM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-LLM/
cd ..
cd MindSpeed-LLM
mkdir logs
mkdir dataset
mkdir ckpt
```

<a id="jump1.2"></a>

#### 2. 环境搭建

torch npu 与 CANN包参考链接：[安装包参考链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl
pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl

# apex for Ascend 参考 https://gitee.com/ascend/apex
# 建议从原仓编译安装

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.8.0
git checkout 2c085cc9
pip install -r requirements.txt
pip install -e .
cd ../MindSpeed-LLM


# 安装其余依赖库
pip install -r requirements.txt
```
**注意** ： 由于首发最新版本支持，要求transformers版本为4.51.3,用户需执行以下命令：

```
  pip install transformers == 4.51.3
```


### 权重转换

1. 权重下载 

    从[huggingface](https://huggingface.co/Qwen/Qwen3-1.7B-Base)或者[魔乐社区](https://modelers.cn/models/MindSpeed/Qwen3-1.7B-Base)下载权重和配置文件

2. 权重转换

    提供脚本将huggingface开源权重转换为mcore权重，用于训练、推理、评估等任务。

    使用方法如下，请根据实际需要的TP/PP等切分策略和权重路径修改权重转换脚本
    ```sh
    cd MindSpeed-LLM
    bash tests/0day/qwen3/qwen3-1.7b/ckpt_convert_qwen3_1.7b_hf2mcore.sh
      ```



### 数据预处理

数据集处理使用方法如下，请根据实际需要修改以下参数

```sh
cd MindSpeed-LLM
bash tests/0day/qwen3/qwen3-1.7b/data_convert_qwen3_1.7b_pretrain.sh
  ```

| 参数名  | 含义                |
|--------|-----------------|
| --input | 数据集路径  |
| --tokenizer-name-or-path | 模型tokenizer目录    |
| --output-prefix | 数据集处理完的输出路径及前缀名  |



### 训练

预训练使用方法如下

  ```sh
  cd MindSpeed-LLM
  bash tests/0day/qwen3/qwen3-1.7b/pretrain_qwen3_1.7b_ptd.sh
   ```
用户需要根据实际情况修改脚本中以下变量
  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CKPT_SAVE_DIR | 训练中权重保存路径  |
  | DATA_PATH | 数据预处理后的数据路径  |
  | TOKENIZER_PATH | qwen3 tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重，如无初始权重则随机初始化  |

### 推理

推理使用方法如下

 ```sh
 cd MindSpeed-LLM
 bash tests/0day/qwen3/qwen3-1.7b/generate_qwen3_1.7b_ptd.sh
   ```
用户需要根据实际情况修改脚本中以下变量
  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | CHECKPOINT | 训练保存的权重路径  |
  | TOKENIZER_PATH | qwen3 tokenizer目录  |

### 评估

评估使用方法如下

 ```sh
 cd MindSpeed-LLM
 bash tests/0day/qwen3/qwen3-1.7b/evaluate_qwen3_1.7b_ptd.sh
   ```
用户需要根据实际情况修改脚本中以下变量
  | 变量名  | 含义                |
  |--------|-----------------|
  | MASTER_ADDR | 多机情况下主节点IP  |
  | NODE_RANK | 多机下，各机对应节点序号    |
  | TOKENIZER_PATH | qwen3 tokenizer目录  |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，为初始加载的权重，如无初始权重则随机初始化  |
  |  DATA_PATH |  评估采用的数据集路径，当前推荐使用MMLU     |
  | TASK  |  评估采用的数据集，当前推荐使用MMLU      | 


### 声明

0day系列模型处于内部测试阶段，未完成充分的性能测试和验收。在实际使用中可能存在未被发现的问题，待后续充分验证后会发布正式版本。相关使用问题请反馈至[MindSpeed-LLM/issues](https://gitee.com/ascend/MindSpeed-LLM/issues)。

MindSpeed-LLM框架将持续支持相关主流模型演进，并根据开源情况面向全体开发者提供支持。




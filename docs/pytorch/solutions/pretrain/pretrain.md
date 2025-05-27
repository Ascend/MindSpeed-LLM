### 大模型分布式预训练

#### 1. 准备工作

参考[安装指导](../../../features/install_guide.md)，完成环境安装，参考[权重转换](../checkpoint_convert.md)和[预训练数据处理](pretrain_dataset.md)完成权重准备和数据预处理。

#### 2. 配置预训练参数

#### 模型脚本环境变量声明：
脚本中的环境变量配置见[环境变量说明](../../features/environment_variable.md)

legacy分支的预训练脚本保存在 example/legacy 中各模型文件夹下：pretrain_xxx_xx.sh

mcore分支的预训练脚本保存在 example/mcore 中各模型文件夹下：pretrain_xxx_xx.sh

需根据实际情况修改路径和参数值：

**示例：**

examples/legacy/llama2/pretrain_llama2_7b_ptd.sh *(legacy分支)*

examples/mcore/llama2/pretrain_llama2_7b_ptd.sh *(mcore分支)*

路径配置：包括**权重保存路径**、**权重加载路径**、**词表路径**、**数据集路径**

 ```shell
# 根据实际情况配置权重保存、权重加载、词表、数据集路径
# 注意：提供的路径需要加双引号
CKPT_SAVE_DIR="./ckpt/llama-2-7b"  # 训练完成后的权重保存路径
CKPT_LOAD_DIR="./model_weights/llama-2-7b-mcore/"  # 权重加载路径，填入权重转换时保存的权重路径
TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf/tokenizer.model"  # 词表路径，填入下载的开源权重词表路径
DATA_PATH="./dataset/enwiki_text_document"  # 数据集路径，填入数据预处理时保存的数据路径
 ```

【--tokenizer-type】 

参数值为PretrainedFromHF时， 词表路径仅需要填到模型文件夹即可，不需要到tokenizer.model文件

**示例：**

```shell 
TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf/"
--tokenizer-name-or-path ${TOKENIZER_PATH}
```

参数值不为PretrainedFromHF时，例如Llama2Tokenizer，需要指定到tokenizer.model文件

**示例：**

```shell 
TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf/tokenizer.model"
--tokenizer-model ${TOKENIZER_MODEL} \
```


【--data-path】 

支持多数据集训练，参数格式如下

格式一（数据集权重根据提供的weight参数）

```shell 
--data-path "dataset1-weight dataset1-path dataset2-weight dataset2-path"
```

**示例：**

```shell 
--data-path "0.5 ./dataset/enwiki_text_document1 0.5 ./dataset/enwiki_text_document2"
```

格式二（根据数据集的长度推出数据集的权重）

```shell 
--data-path "dataset1-path dataset2-path"
```

**示例：**

```shell 
--data-path "./dataset/enwiki_text_document1 ./dataset/enwiki_text_document2"
```

【单机运行】 

```shell
GPUS_PER_NODE=8
MASTER_ADDR=locahost
MASTER_PORT=6000
NNODES=1  
NODE_RANK=0  
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

【多机运行】 

```shell
# 根据分布式集群实际情况配置分布式参数
GPUS_PER_NODE=8  # 每个节点的卡数
MASTER_ADDR="your master node IP"  # 都需要修改为主节点的IP地址（不能为localhost）
MASTER_PORT=6000
NNODES=2  # 集群里的节点数，以实际情况填写,
NODE_RANK="current node id"  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```


#### 3. 启动预训练

##### 初始化环境变量 

`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

`source /usr/local/Ascend/nnal/atb/set_env.sh`

【legacy分支】

```shell
bash examples/legacy/模型文件夹/pretrain_xxx_xxx.sh
```

**示例：** *(以llama2-7B为例)*

```shell
bash examples/legacy/llama2/pretrain_llama2_7b_ptd.sh
```

【mcore分支】 

```shell
bash examples/mcore/模型文件夹/pretrain_xxx_xxx.sh
```

**示例：** 

```shell
bash examples/mcore/llama2/pretrain_llama2_7b_ptd.sh
```

**注意**：
- "--save-interval" 用于设定运行多少个epoch后存储权重。
- 如需存储日志到脚本文件中，请在运行路径目录下创建logs文件夹。
- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据

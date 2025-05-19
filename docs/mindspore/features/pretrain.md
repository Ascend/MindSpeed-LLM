## 大模型分布式预训练
### 1. 准备工作

参考[安装指导](./install_guide.md)，完成环境安装和[预训练数据处理](../../pytorch/solutions/pretrain/pretrain_dataset.md)。


### 2. 配置预训练参数

预训练脚本保存在 MindSpeed-LLM/examples/mindspore 中各模型文件夹下：pretrain_xxx_ms.sh

需根据实际情况修改路径和参数值：

示例：

MindSpeed-LLM/examples/mindspore/deepseek3/pretrain_deepseek3_ms.sh

路径配置：包括权重保存路径、权重加载路径、词表路径、数据集路径


``` shell
# 根据实际情况配置权重保存、权重加载、词表、数据集路径
# 注意：提供的路径需要加双引号
CKPT_SAVE_DIR="./ckpt/deepseek3"  # 训练完成后的权重保存路径
CKPT_LOAD_DIR="./model_weights/deepseek3/"  # 权重加载路径，填入权重转换时保存的权重路径
DATA_PATH="./dataset/enwiki_text_document"  # 数据集路径，填入数据预处理时保存的数据路径
TOKENIZER_MODEL="./model_from_hf/deepseek3/tokenizer"  # 词表路径，填入下载的开源权重词表路径
```


【单机运行】

``` shell
GPUS_PER_NODE=8
MASTER_PORT=6000
MASTER_ADDR=locahost # 主节点IP
NNODES=1
NODE_RANK=0  
MASTER_PORT=9110
log_dir=msrun_log_pretrain # log输出路径
```

【多机运行】

```shell
# 根据分布式集群实际情况配置分布式参数
GPUS_PER_NODE=8
MASTER_ADDR="your master node IP" #主节点IP
MASTER_PORT=6000
NNODES=4 # 集群里的节点数，以实际情况填写
NODE_RANK="current node id" # 当前节点RANK,主节点为0，其他可以使1,2..
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
```

### 3. 启动预训练
**初始化环境变量**

`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

`source /usr/local/Ascend/nnal/atb/set_env.sh`

**示例：**
仓库与环境变量配置完成后运行预训练，注意工作目录在MindSpeed-LLM

```bash
cd MindSpeed-LLM
sh examples/mindspore/deepseek3/pretrain_deepseek3_ms.sh
```

**注意：**

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加--no-shared-storage参数，设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据

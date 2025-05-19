# 大模型分布式预训练——pack模式

## EOD Reset训练场景

预训练任务中，通常一个批次中输入进模型的文本序列是由多个文档（doc）拼接得到。在默认情况下，多个文档被视为同一序列，互相间的self attention没有掩盖。在特定情况下，如果多个文档不可以被直接拼接成一个序列，即要求多个文档间要求独立，此时文档间不能互相做self attention，在这种情况下attention mask和position ids需要在每个文档结束的位置（EOD）被重新设置。

MindSpeed-LLM支持多样本pack模式预训练，即在多个样本进行拼接的时候，使用文档结束符(eod)将不同的文档分割开，在训练过程中进行不同doc之间self attention的隔离。使用此功能，只需要在数据预处理和训练脚本中添加相应参数即可。

## 使用说明

可参考llama3-8b脚本：[pack预训练数据处理](../../../../examples/mcore/llama3/data_convert_llama3_pretrain_pack.sh)，[pack模式预训练](../../../../examples/mcore/llama3/pretrain_llama3_8b_pack_ptd.sh)。

#### 数据预处理

在[预训练数据预处理](pretrain_dataset.md)基础上，加上`--append-eod`参数，即可进行pack预训练模式的数据预处理:

```shell
python ./preprocess_data.py \
   --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
   --tokenizer-name-or-path ./model_from_hf/llama3_hf/ \
   --output-prefix ./dataset/enwiki \
   --workers 4 \
   --log-interval 1000  \
   --tokenizer-type PretrainedFromHF \
   --append-eod  # 预训练数据预处理添加此参数使能pack模式预训练
```

#### 训练脚本

在[普通预训练](pretrain.md)基础上，加上`--reset-position-ids`参数，即可进行pack模式预训练。

```shell
# examples/mcore/llama3/pretrain_llama3_8b_pack_ptd.sh
.....
ACCELERATE_ARGS="
    --swap-attention \
    --use-mc2 \
    --reuse-fp32-param \
    --reset-position-ids \  # 在普通预训练脚本中添加此参数进行pack模式预训练
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
"
....
```
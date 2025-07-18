## Mindspore后端提供Qwen3系列模型支持
<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
      <th>是否支持</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/Qwen">Qwen3</a></td>
      <td rowspan="1"><a href="https://huggingface.co/Qwen/Qwen3-0.6B/tree/main">0.6B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>✅</td>
      <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen/Qwen3-14B/tree/main">14B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>✅</td>
      <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen/Qwen3-32B/tree/main">32B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td>✅</td>
      <tr>
    </tr>
  </tbody>
</table>

## MindSpore后端跑通Qwen3模型手把手教程


### 环境配置

MindSpeed-LLM MindSpore后端的安装步骤参考[基础安装指导](../../../docs/mindspore/features/install_guide.md)。

### 训练

#### 预训练

预训练使用方法如下
```sh
# 以0.6b模型为例
cd MindSpeed-LLM
bash examples/mindspore/qwen3/pretrain_qwen3_0point6b_4K_ms.sh
```
用户需要根据实际情况修改脚本中的以下变量

  |变量名  | 含义                                |
  |--------|-----------------------------------|
  | MASTER_ADDR | 多机情况下主节点IP                        |
  | NODE_RANK | 多机下，各机对应节点序号                      |
  | CKPT_SAVE_DIR | 训练中权重保存路径                         |
  | DATA_PATH | 数据预处理后的数据路径                       |
  | TOKENIZER_PATH | qwen3 tokenizer目录                 |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，用于初始权重加载，如无初始权重则随机初始化 |

#### 微调
微调和预训练的使用方法类似
```sh
# 以全参微调0.6b模型为例
cd MindSpeed-LLM
bash examples/mindspore/qwen3/tune_qwen3_0point6b_4K_full_ms.sh
```
与预训练一样，用户需要根据实际情况修改脚本中的上述变量。

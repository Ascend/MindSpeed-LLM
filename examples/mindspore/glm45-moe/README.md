## Mindspore后端提供GLM4.5系列模型支持
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
      <td rowspan="7"><a href="https://huggingface.co/zai-org/GLM-4.5">GLM4.5</a></td>
      <td rowspan="2"><a href="https://huggingface.co/zai-org/GLM-4.5/tree/main">106B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>8x16</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## MindSpore后端跑通GLM4.5模型教程


### 环境配置

MindSpeed-LLM MindSpore后端的安装步骤参考[基础安装指导](../../../docs/mindspore/features/install_guide.md)。


### 训练

#### 预训练

预训练使用方法如下
```sh
cd MindSpeed-LLM
bash examples/mindspore/glm45-moe/pretrain_glm45_moe_106b_4k_A3_ms.sh
```
用户需要根据实际情况修改脚本中的以下变量

  |变量名  | 含义                                |
  |--------|-----------------------------------|
  | MASTER_ADDR | 多机情况下主节点IP                        |
  | NODE_RANK | 多机下，各机对应节点序号                      |
  | CKPT_SAVE_DIR | 训练中权重保存路径                         |
  | DATA_PATH | 数据预处理后的数据路径                       |
  | TOKENIZER_PATH | GLM4.5 tokenizer目录                |
  | CKPT_LOAD_DIR | 权重转换保存的权重路径，用于初始权重加载，如无初始权重则随机初始化 |


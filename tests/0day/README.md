## 0day系列模型提供昇腾同步首发支持

`认证`【Pass】表示经过昇腾官方版本测试的模型，【Test】表示待测试模型

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
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
      <td rowspan="3"> <a href="https://modelscope.cn/collections/GLM-4-0414-e4ecc89c179d4c">GLM-4</a> </td>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-4-9B-0414">9B</a></td>
      <td><a href="glm4-9b-0414/">GLM4-9B-0414</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-4-32B-0414">32B</a></td>
      <td><a href="glm4-32b-0414/">GLM4-32B-0414</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-4-32B-Base-0414">32B-Base</a></td>
      <td><a href="glm4-base-32b-0414/">GLM4-base-32B-0414</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="3"> <a href="https://modelscope.cn/collections/GLM-4-0414-e4ecc89c179d4c">GLM-Z1</a> </td>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-Z1-9B-0414">9B</a></td>
      <td><a href="glm-z1-9b-0414/">GLM-Z1-9B-0414</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-Z1-32B-0414">32B</a></td>
      <td><a href="glm-z1-32b-0414/">GLM-Z1-32B-0414</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/ZhipuAI/GLM-Z1-Rumination-32B-0414">Rumination-32B</a></td>
      <td><a href="glm-z1-rumination-32b-0414/">GLM-Z1-Rumination-32B-0414</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>


## 模型脚本声明：

上述模型目前仅支持0day首发下基本功能跑通，处于内部测试阶段，未完成充分的性能测试和验收。在实际使用中可能存在未被发现的问题，待后续充分验证后会发布正式版本。


## 配套版本说明：

以上模型的依赖配套如下表：

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">在研版本</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
      <td rowspan="3">在研版本</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
    <td>NNAL（Ascend Transformer Boost加速库）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td rowspan="3">2.5</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
  <tr>
    <td>transformers</td>
    <td>4.51.3</td>
  </tr>
</table>

## 版本声明

#### 参考实现：
```shell
url=https://github.com/huggingface/transformers/tree/v4.51.3
commit_id=5f4ecf2
```

#### 变更记录：

- 2025.4.22：首次发布
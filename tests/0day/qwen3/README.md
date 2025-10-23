## 0day提供Qwen3系列模型同步首发支持

`认证`【Pass】表示经过昇腾官方版本测试的模型，【Test】表示待测试模型

**目前qwen3模型已通过内部验证，可参考[安装指导](../../../docs/pytorch/install_guide.md)完成环境安装**

**[test/0day/qwen3](../qwen3/)文件夹下其他脚本仅作展示用，请移步[examples/mcore](../../../examples/mcore)使用完整功能。**


<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>魔乐社区链接</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"> <a href="../../../examples/mcore/qwen3/">Qwen3-dense</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base">0.6B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-0.6B-Base">0.6B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-1.7B-Base">1.7B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-1.7B-Base">1.7B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-4B-Base">4B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-4B-Base">4B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-8B-Base">8B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-8B-Base">8B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-14B-Base">14B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-14B-Base">14B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">32B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-32B">32B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="../../../examples/mcore/qwen3_moe/">Qwen3-moe</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Base">30B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-30B-A3B-Base">30B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B">235B</a></td>
      <td><a href="https://modelers.cn/models/MindSpeed/Qwen3-235B-A22B">235B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 16x16 </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>


# 环境配置

## 硬件要求


|类型|  硬件 |       配置        |
|:---:|:---:|:---------------:|
|预训练| NPU | 8 x Ascend NPUs |

## 环境搭建

MindSpeed-LLM的主要依赖配套如下表，安装步骤参考[安装指导](../../../docs/pytorch/install_guide.md)。

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
    <td >4.51.0</td>
  </tr>
</table>



**注意** ： Qwen 3系列模型要求transformers版本为4.51.0,用户需执行以下命令：

```
  pip3 install transformers==4.51.0
```

## 版本声明

#### 参考实现：
```shell
url=https://github.com/huggingface/transformers/tree/v4.51.0
commit_id=0720e20
```

### 声明

0day系列模型尚有部分功能处于内部测试阶段，未完成充分的性能测试和验收。在实际使用中可能存在未被发现的问题，待后续充分验证后会发布正式版本。相关使用问题请反馈至[MindSpeed-LLM/issues](https://gitcode.com/ascend/MindSpeed-LLM/issues)。

MindSpeed-LLM框架将持续支持相关主流模型演进，并根据开源情况面向全体开发者提供支持。
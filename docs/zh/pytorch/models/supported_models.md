# PyTorch框架模型支持列表

PyTorch框架根据模型架构特点分为稠密模型、稀疏模型和状态空间模型三大类别，详情请查看以下支持列表。

**表格字段说明**：

- **模型**：模型名称
- **下载链接**：模型权重下载地址，点击可直接访问Hugging Face等模型仓库
- **脚本位置**：模型在本项目中的训练脚本路径，可用于快速定位和使用模型。[脚本环境变量说明](../features/mcore/environment_variable.md)
- **序列长度**：支持的最大文本序列长度
- **训练后端**：支持Legacy、Mcore、FSDP2三种实现方式的模型。Legacy为历史实现方式，Mcore为当前推荐的Megatron主流实现方式，FSDP2为PyTorch官方推荐的分布式训练实现方式。
- **集群规模**：模型训练时推荐使用的集群规模配置，格式为"节点数×卡数"
- **支持版本**：最终支持的维护版本，空白表示从上线起到当前master分支均在维护
- **贡献方**：模型贡献来源
- **认证**：`【Pass】`已通过官方测试；`【Test】`内部测试中。如有问题请反馈至[issues](https://gitcode.com/ascend/MindSpeed-LLM/issues)

> [!NOTE]
> 
> - 仓库支持Atlas A2 和Atlas A3 训练系列产品，可通过[硬件信息查询](https://www.hiascend.com/cann/download)获取。并且要求单NPU的HBM为64GB及以上，[安装驱动固件](../training/install_guide.md)后可通过npu-smi info查看单NPU的HBM容量。
> - 提供[模型训练示例脚本](../../../../examples)，训练脚本命名为`pretrain_模型名_参数_文本序列长度_训练方案_训练产品.sh`，微调脚本命名为`tune_模型名_参数_文本序列长度_训练方案_训练产品.sh`。
> - 在模型训练示例脚本中可查看集群规模，其中`NNODES`指代机器数量，`NPUS_PER_NODE`指代每台机器上的NPU数量。如下表示需要32台机器，每台机器上有16个NPU：
>
> ```bash
> NNODES=32
> NPUS_PER_NODE=16
> ```

## 稠密模型

稠密模型（Dense Model）是传统的深度学习模型结构，其神经元之间的连接是密集的，每一层的大多数或所有神经元都与下一层的所有神经元相连。这种模型很简单，训练相对直接，但参数量较大，计算成本较高。

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>脚本位置</th>
      <th>序列长度</th>
      <th>训练后端</th>
      <th>集群规模</th>
      <th>支持版本</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/collections/BAAI/aquila-6698657124de09d10cd7a83f">Aquila</a></td>
      <td><a href="https://huggingface.co/BAAI/Aquila-7B/tree/main">7B</a></td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/aquila">aquila</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/collections/BAAI/aquila-6698657124de09d10cd7a83f">Aquila2</a></td>
      <td><a href="https://huggingface.co/BAAI/Aquila2-7B/tree/main">7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/aquila2">aquila2</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/BAAI/Aquila2-34B/tree/main">34B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 2x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/baichuan-inc">Baichuan</a></td>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main">7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/baichuan">baichuan</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main">13B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/baichuan-inc">Baichuan2</a></td>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main">7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/baichuan2">baichuan2</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/tree/main">13B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/bigscience">Bloom</a></td>
      <td><a href="https://huggingface.co/bigscience/bloom-7b1/tree/main">7B1</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/bloom">bloom</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/bigscience/bloom/tree/main">176B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td >12x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/THUDM">ChatGLM3</a></td>
      <td rowspan="3"><a href="https://huggingface.co/THUDM/chatglm3-6b-base/tree/main">6B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/chatglm3">chatglm3</a></td>
      <td>8K</td>
      <th>Mcore</th>
      <td >1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td>32K</td>
      <th>Mcore</th>
      <td >1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td>64K</td>
      <th>Mcore</th>
      <td >2x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/THUDM">GLM4</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/glm-4-9b">9B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/glm4">glm4</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th>Mcore</th>
      <td> 2x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/codellama">CodeLlama</a></td>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-hf/tree/main">34B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/codellama">codellama</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td> 2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/internlm">InternLM</a></td>
      <td><a href="https://huggingface.co/internlm/internlm-7b/tree/main">7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/intern">intern</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td >65B</td>
      <td>2K</td>
      <th>Legacy</th>
      <td >4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/internlm">InternLM2</a> </td>
      <td rowspan="2"> <a href="https://huggingface.co/Internlm/Internlm2-chat-20b/tree/main">20B</a> </td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/internlm2">internlm2</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"> <a href="https://huggingface.co/internlm">InternLM2.5</a> </td>
      <td><a href="https://huggingface.co/internlm/internlm2_5-1_8b/tree/main">1.8B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/internlm25">internlm25</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-7b/tree/main">7B</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-20b/tree/main">20B</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/internlm">InternLM3</a></td>
      <td><a href="https://huggingface.co/internlm/internlm3-8b-instruct/tree/main">8B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/internlm3">internlm3</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/meta-llama">LLaMA</a></td>
      <td><a href="https://huggingface.co/huggyllama/llama-7b/tree/main">7B</a></td>
      <td rowspan="4"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/llama">llama</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/ruibin-wang/llama-13b-hf">13B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/pinkmanlove/llama-33b-hf/tree/main">33B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/pinkmanlove/llama-65b-hf">65B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="5"><a href="https://huggingface.co/meta-llama">LLaMA2</a></td>
      <td><a href="https://huggingface.co/daryl149/llama-2-7b-hf/tree/main">7B</a></td>
      <td rowspan="5"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/llama2">llama2</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【NAIE】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/NousResearch/Llama-2-13b-hf/tree/main">13B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【NAIE】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/tree/main">34B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama/Llama-2-70b-hf">70B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 128K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">LLaMA3</a></td>
      <td><a href="https://huggingface.co/unsloth/llama-3-8b/tree/main">8B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/llama3">llama3</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/v2ray/Llama-3-70B/tree/main">70B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://modelscope.cn/organization/LLM-Research">LLaMA3.1</a></td>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B">8B</a></td>
      <td rowspan="8"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/llama31">llama31</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td>128K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 50B </td>
      <td> 128K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B">70B</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 128K </td>
      <th> Mcore </th>
      <td> 24x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 200B </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"> 405B </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 128K </td>
      <th> Mcore </th>
      <td> 36x8 </td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">LLaMA3.2</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-1B/tree/main">1B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/llama32">llama32</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-3B/tree/main">3B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/meta-llama">LLaMA3.3</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/tree/main">70B-Instruct</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/llama33">llama33</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 4x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/Qwen">Qwen</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen-7B/tree/main">7B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/qwen">qwen</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen-14B/tree/main">14B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen-72B/tree/main">72B</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>16x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B/tree/main">0.5B</a> </td>
      <td rowspan="9"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen15">qwen15</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="9"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main">1.8B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-4B/tree/main">4B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-7B/tree/main">7B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-14B/tree/main">14B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-32B/tree/main">32B</a> </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-72B/tree/main">72B</a> </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td>【GTS】</td>    
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-110B/tree/main">110B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 8x8 </td>
      <td>【GTS】</td>    
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">CodeQwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/CodeQwen1.5-7B">7B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>    
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td rowspan="2"> <a href="https://huggingface.co/Qwen/Qwen2-0.5B/tree/main">0.5B</a> </td>
      <td rowspan="8"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen2">qwen2</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td rowspan="8"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/Qwen/Qwen2-1.5B/tree/main">1.5B</a> </td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2-7B/tree/main">7B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2-72B/tree/main">72B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 16x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="7"><a href="https://huggingface.co/Qwen">Qwen2.5</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/main">0.5B</a></td>
      <td rowspan="7"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/qwen25">qwen25</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main">1.5B</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B/tree/main">3B</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-7B/tree/main">7B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-14B/tree/main">14B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-32B/tree/main">32B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-72B/tree/main">72B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>16x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="7"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base">0.6B</a></td>
      <td rowspan="6"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/qwen3">qwen3</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-1.7B-Base">1.7B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-4B-Base">4B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-8B-Base">8B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-14B-Base">14B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">32B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">32B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/fsdp2/qwen3">qwen3</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 1x16 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/collections/Qwen/qwq-674762b79b75eac01735070a">QwQ</a></td>
      <td><a href="https://huggingface.co/Qwen/QwQ-32B/tree/main">32B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwq">qwq</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/Qwen">Qwen2.5-Math</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/tree/main">1.5B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen25_math">qwen25_math</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Math-7B/tree/main">7B</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Math-72B/tree/main">72B</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
 <tr>
   <td rowspan="1"><a href="https://huggingface.co/Qwen">CodeQwen2.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen2.5-Coder-7B">7B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen25_coder">qwen25_coder</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【China Mobile Cloud】</td>    
      <td>【Test】</td>
    </tr>
 <tr>
      <td rowspan="2"><a href="https://huggingface.co/01-ai">Yi</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-9B/tree/main">9B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/yi">yi</a></td>
      <td> 4K</td>
      <th>Legacy</th>
      <td>1x4</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【OpenMind】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-34B/tree/main">34B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/01-ai">Yi1.5</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-6B/tree/main">6B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/yi15">yi15</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-9B/tree/main">9B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-34B/tree/main">34B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/mistralai">Mistral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main">7B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/mistral">mistral</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【NAIE】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma</a></td>
      <td><a href="https://huggingface.co/google/gemma-2b/tree/main">2B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/gemma">gemma</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-7b">7B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma2</a></td>
      <td><a href="https://huggingface.co/google/gemma-2-9b/tree/main">9B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/gemma2">gemma2</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-2-27b/tree/main">27B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>2x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/tree/main">2B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【NAIE】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/openbmb/MiniCPM3-4B/tree/main">MiniCPM3</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM3-4B/tree/main">4B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/minicpm3">minicpm3</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/tree/main">mini-instruct</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/phi35">phi35</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/deepseek-math-7b-base">DeepSeek-Math</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/deepseek-math-7b-base">7B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek_math">deepseek_math</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/deepseek-ai">DeepSeek-R1-Distill-Qwen</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">1.5B</a></td>
      <td rowspan="4"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek_r1_distill_qwen">deepseek_r1_distill_qwen</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="4"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">7B</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B">14B</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B">32B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 2x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/deepseek-ai">DeepSeek-R1-Distill-LLaMA</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B">8B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek_r1_distill_llama">deepseek_r1_distill_llama</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B">70B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 4x8 </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/ByteDance-Seed">Seed-OSS</a></td>
      <td><a href="https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Base/tree/main">36B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/seed_oss">seed_oss</a></td>
      <td> 2K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/mistralai">Magistral</a></td>
      <td><a href="https://huggingface.co/mistralai/Magistral-Small-2506/tree/main">24B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/magistral">magistral</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/PLM-Team">PLM</a></td>
      <td><a href="https://huggingface.co/PLM-Team/PLM-1.8B-Base/tree/main">1.8B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/plm">plm</a></td>
      <td> 2K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>

## 稀疏模型

稀疏模型（Sparse Model）采用了稀疏连接的神经元结构，只有少数神经元之间存在连接。典型的稀疏模型如混合专家模型（Mixture of Experts, MoE），包含多个专家网络，每次训练只激活部分专家。这种设计可以显著减少参数量和计算复杂度，提高训练效率，特别适合处理大规模数据集和复杂任务。但稀疏模型训练也存在缺点，易出现专家负载不均衡导致训练不稳定。

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>脚本位置</th>
      <th>序列长度</th>
      <th>训练后端</th>
      <th>集群规模</th>
      <th>支持版本</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Base">30B-A3B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/fsdp2/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 1x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B">235B-A22B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 16x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/fsdp2/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 16x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/collections/Qwen/qwen3-next">Qwen3-Next</a></td>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct">80B-A3B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/qwen3_next">qwen3_next</a></td>
      <td> 16K </td>
      <th> Mcore </th>
      <td> 4x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/fsdp2/qwen3_next">qwen3_next</a></td>
      <td> 16K </td>
      <th> FSDP2 </th>
      <td> 4x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main">Qwen3-Coder-Next</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main">80B-A3B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/qwen3_coder_next">qwen3_coder_next</a></td>
      <td> 16K </td>
      <th>Mcore</th>
      <td>4x16</td>
      <td rowspan="1"> <a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/qwen3_coder_next"></a> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-57B-A14B/tree/main">57B-A14B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen2_moe">qwen2_moe</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/xai-org/grok-1/tree/main">Grok-1</a></td>
      <td><a href="https://huggingface.co/xai-org/grok-1/tree/main">40B</a></td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/mcore/grok1">grok-1</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/mistralai">Mixtral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">8x7B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/mixtral">mixtral</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/tree/main">8x22B</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td>【NAIE】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 64K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td>【NAIE】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">DeepSeek-V2</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek2">deepseek2</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
        <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base">DeepSeek-V2-coder</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek2_coder">deepseek2_coder</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite">DeepSeek-V2-Lite</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main">16B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/deepseek2_lite">deepseek2_lite</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5">DeepSeek-V2.5</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek25">deepseek25</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【NAIE】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">DeepSeek-V3</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main">671B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/deepseek3">deepseek3</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 64x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
      <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp">DeepSeek-V3.2</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main">671B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/deepseek32">deepseek3.2</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 32x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-MoE-8x2B/tree/main">8x2B</a> </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>【NAIE】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/inclusionAI/Ling-mini-2.0">Ling-mini-2.0</a></td>
      <td> <a href="https://huggingface.co/inclusionAI/Ling-mini-2.0/tree/main">16B</a> </td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/ling_v2">ling_v2</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="1"></td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/inclusionAI/Ring-1T">Ring</a></td>
      <td> <a href="https://huggingface.co/inclusionAI/Ring-1T/tree/main">1T</a> </td>
      <td> 32K </td>
      <th>Mcore</th>
      <td> 32x8 </td>
      <td rowspan="1"></td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-MoE-instruct">MoE-instruct</a> </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/phi35">phi35</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td>  </td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/tencent">Hunyuan</a></td>
      <td> <a href="https://huggingface.co/tencent/Tencent-Hunyuan-Large">389B</a> </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/hunyuanLarge">hunyuanLarge</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td>GPT4</td>
      <td>MoE-175B</td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/gpt4">gpt4</a></td>
      <td> 128K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/zai-org">GLM4.5-Air</a></td>
      <td> <a href="https://huggingface.co/zai-org/GLM-4.5-Air/tree/main">MoE-106B</a> </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/glm45-air">glm45-air</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/THUDM">GLM5</a></td>
      <td><a href="https://huggingface.co/THUDM/GLM-5">MoE-744B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/glm5">glm5</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 32x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/stepfun-ai">Step3.5-Flash</a></td>
      <td><a href="https://huggingface.co/stepfun-ai/Step-3.5-Flash">MoE-196B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/fsdp2/step35">step35</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 12x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/meituan-longcat">LongCat</a></td>
      <td><a href="https://huggingface.co/meituan-longcat/LongCat-Flash-Chat">MoE-560B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/longcat">longcat</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 8x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/openai">GPT-OSS</a></td>
      <td><a href="https://modelscope.cn/models/unsloth/gpt-oss-20b-BF16/">MoE-20B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/fsdp2/gpt_oss">gpt_oss</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 1x16 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
GPT模型词表文件与常规模型不同，需按以下步骤配置：
>
> ```shell
> mkdir vocab_file 
> cd vocab_file
> wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
> wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
> cd ..
> 
> # 处理成训练数据
> python ./preprocess_data.py \
>     --input ./dataset/ \
>     --output-prefix ./dataset/gpt_text_sentence \
>     --tokenizer-type GPT2BPETokenizer \
>     --vocab-file ./vocab_file/gpt2-vocab.json \
>     --merge-file ./vocab_file/gpt2-merges.txt \
>     --append-eod \
>     --workers 4 \
>     --log-interval 1000
> 
> # 请根据真实存放路径配置预训练脚本以下参数
> VOCAB_FILE="./vocab_file/gpt2-vocab.json"   # 词表
> MERGE_FILE="./vocab_file/gpt2-merges.txt"   # BPE 合并表
> DATA_PATH="./dataset/gpt_text_sentence"     # 数据路径
> ```

## 状态空间模型

状态空间模型（State Space Model, SSM）是一类基于状态空间表示的序列模型，能够高效地建模长序列数据。与传统的Transformer模型相比，SSM在处理长序列时具有更好的计算效率和内存效率。

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>脚本位置</th>
      <th>序列长度</th>
      <th>训练后端</th>
      <th>集群规模</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Mamba2</td>
      <td><a href="https://huggingface.co/state-spaces/mamba2-2.7b/tree/main">2.7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/mamba2">mamba2</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td> 1x8</td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/main">8B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td> 1x8</td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>      
    <tr>
      <td rowspan="2">Mamba2Hybrid</td>
      <td><a href="https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-4k/tree/main">8B</a></td>
       <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/examples/mcore/mamba2">mamba2</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>   
  </tbody>
</table>

> [!NOTE]
> 该开源模型未提供词表文件。内部测试使用的词表文件 `mamba2_2.7b_from_8b.model` 为自定义设计，建议用户自行构建词表，训练效果不作保证。

## 多模态模型

多模态大模型（图文理解、文生视频/图、语音识别等）由MindSpeed-MM专项维护，如果需要进行多模态大模型的训练，请访问多模态仓库[MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM)获取详细的使用说明，当前MindSpeed-MM已支持以下的主流模型：

<table>
  <a id="jump1"></a>
  <caption>MindSpeed MM模型列表</caption>
  <thead>
    <tr>
      <th>模型任务</th>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
      <th>平均序列长度</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="46"> 多模态生成 </td>
      </tr>
      <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/lumina">Lumina-mGPT 2.0</a></td>
      <td><a href="https://huggingface.co/Alpha-VLLM/Lumina-mGPT-2.0">7B</a></td>
      <td> 微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 8.24 (SPS)</td>
      <td> 8.79 (SPS)</td>
      <td> 1024 </td>
      <td>【Pass】</td>
    </tr>
      <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/opensoraplan1.5/">OpenSoraPlan1.5</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.5.0">8.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.83 (SPS) </td>
      <td> / </td>
      <td> / </td>
      <td>【北大贡献】</td>
    </tr>
      <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/wan2.2">Wan2.2-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">5B</a></td>
      <td> 预训练 </td>
      <td> 1x4 (A3) </td>
      <td> BF16 </td>
      <td> 3.18 (SPS) </td>
      <td> 2.93 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers">A14B</a></td>
      <td> 预训练 </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
      <td> 0.710 (SPS) </td>
      <td> 0.292 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
      <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/wan2.2">Wan2.2-TI2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">5B</a></td>
      <td> 预训练 </td>
      <td> 1x4 (A3) </td>
      <td> BF16 </td>
      <td> 3.18 (SPS) </td>
      <td> 2.93 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/wan2.2">Wan2.2-I2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers">A14B</a></td>
      <td> 预训练 </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
      <td> 0.671 (SPS) </td>
      <td> 0.294 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/wan2.1">Wan2.1-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">1.3B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.918 (SPS) </td>
      <td> 1.04 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">1.3B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.954 (SPS) </td>
      <td> 1.042 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">14B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.160 (SPS) </td>
      <td> 0.160 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers">14B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.179 (SPS) </td>
      <td> 0.174 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/wan2.1">Wan2.1-I2V</a></td>
      <td>1.3B</td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.76 (SPS) </td>
      <td>  / </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">14B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.130 (SPS) </td>
      <td> / </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers">14B</a></td>
      <td> Lora微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.179 (SPS) </td>
      <td> 0.173 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/self_forcing">Self-Forcing</a></td>
      <td><a href="https://huggingface.co/gdhe17/Self-Forcing">1.3B</a></td>
      <td> DMD蒸馏 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.225 (FPS) </td>
      <td> 0.282 (FPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/hunyuanvideo">HunyuanVideo-T2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo">13B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.171 (SPS) </td>
      <td> 0.181 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/hunyuanvideo">HunyuanVideo-I2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo-I2V">13B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.164 (SPS) </td>
      <td> 0.202 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/hunyuanvideo1.5">HunyuanVideo1.5-T2V</a></td>
      <td><a href="https://huggingface.co/tencent/HunyuanVideo1.5-T2V">8B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensora1.0">OpenSora 1.0</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/Open-Sora/tree/main">5.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 3.18 (SPS)</td>
      <td> 2.04 (SPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensora1.2">OpenSora 1.2</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3">5.2B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 7.31 (SPS) </td>
      <td> 8.15 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/opensora2.0">OpenSora 2.0-T2V</a></td>
      <td><a href="https://huggingface.co/hpcai-tech/Open-Sora-v2">11B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1.33 (SPS) </td>
      <td> 1.46 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/opensoraplan1.2">OpenSoraPlan 1.2</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0">8.7B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 0.42 (SPS) </td>
      <td> 0.37 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.3">OpenSoraPlan 1.3-T2V</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.29 (SPS) </td>
      <td> 1.27 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/opensoraplan1.3">OpenSoraPlan 1.3-I2V</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0"> 8.6B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.17 (SPS) </td>
      <td> 1.15 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/vae">WFVAE</a></td>
      <td><a href="https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.3.0/tree/main/vae"> 0.18B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 23.860 (SPS) </td>
      <td> 26.091 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.14 (SPS) </td>
      <td> 1.00 (SPS) </td>
      <td> 6976 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.13 (SPS) </td>
      <td> 0.84 (SPS) </td>
      <td> 6976 </td>
      <td>【Pass】</td>
    </tr>
  <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX 1.5-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.44 (SPS) </td>
      <td> 1.75 (SPS) </td>
      <td> 6976 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2.76 (SPS) </td>
      <td> 2.64 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX 1.5-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.43 (SPS) </td>
      <td> 1.44 (SPS) </td>
      <td> 6976 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2.33 (SPS) </td>
      <td> 2.04 (SPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/qihoo_t2x">Qihoo-T2X</a></td>
      <td><a href="https://huggingface.co/qihoo360/Qihoo-T2X">1.1B</a></td>
      <td> 推理 </td>
      <td> 1x1 </td>
      <td> BF16 </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【奇虎360贡献】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/sdxl">SDXL</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5956b68a6927126daffc2c5a6d1a9a189defe288">3.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 29.92  (FPS)</td>
      <td> 30.65 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/5956b68a6927126daffc2c5a6d1a9a189defe288">3.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 28.51 (FPS)</td>
      <td> 30.23 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/sd3">SD3</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5f724735437d91ed05304da478f3b2022fe3f6fb">2B</a></td>
      <td> 全参微调 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 16.09 (FPS)</td>
      <td> 16.01 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/sd3">SD3.5</a></td>
      <td><a href="https://github.com/huggingface/diffusers/tree/5f724735437d91ed05304da478f3b2022fe3f6fb"> 8.1B </a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 26.20 (FPS)</td>
      <td> 28.33 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/huggingface/diffusers/tree/94643fac8a27345f695500085d78cc8fa01f5fa9"> 8.1B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> FP16 </td>
      <td> 47.93 (FPS)</td>
      <td> 47.95 (FPS)</td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/flux">Flux</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">12B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 55.23 (FPS) </td>
      <td> 53.65 (FPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/flux2">Flux2-T2I</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">32B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.28 (FPS) </td>
      <td> 1.24 (FPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/flux2">Flux2-I2I</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">32B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 0.61 (FPS) </td>
      <td> 0.60 (FPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/flux-kontext">Flux-Kontext</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">12B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.97 (FPS) </td>
      <td> 2.00 (FPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/sana">Sana</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">1.6B</a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 28.7 (FPS) </td>
      <td> 32.8 (FPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/hidream">HiDream</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">17B</a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 18.37 (FPS) </td>
      <td> 19.61 (FPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/kolors">Kolors</a></td>
      <td><a href="https://github.com/Kwai-Kolors/Kolors">2.6B</a></td>
      <td> 推理 </td>
      <td> 1x1 </td>
      <td> FP16 </td>
      <td> / </td>
      <td> / </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffusers/qwen_image">Qwen-Image</a></td>
      <td><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth">27B</a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 23.02 (FPS) </td>
      <td> 21.54 (FPS) </td>
      <td> / </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/diffsynth/qwen_image_edit">Qwen-Image-Edit</a></td>
      <td><a href="https://github.com/modelscope/Diffsynth-Studio/tree/main/examples/qwen_image">27B</a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 20.59 (FPS) </td>
      <td> 17.47 (FPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="25"> 多模态理解 </td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/glm4.1v">GLM-4.1V</a></td>
      <td><a href="https://github.com/THUDM/GLM-4.1V-Thinking">9B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1074.64(TPS) </td>
      <td> 908.49(TPS) </td>
      <td> 707 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/deepseekocr">DeepSeek-OCR</a></td>
      <td><a href="https://github.com/deepseek-ai/DeepSeek-OCR">3B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1327.694(TPS) </td>
      <td> / </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/llava1.5">LLaVA 1.5</a></td>
      <td><a href="https://github.com/haotian-liu/LLaVA">7B</a></td>
      <td> 全参微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 3632.31 (TPS) </td>
      <td> 3757.98 (TPS) </td>
      <td> 602 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/internvl2">InternVL 2.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-2B">2B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 7653.12 (TPS) </td>
      <td> 5089.99 (TPS) </td>
      <td> 1813 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-8B">8B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2914.39 (TPS) </td>
      <td> 2492.87 (TPS) </td>
      <td> 1813 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-26B">26B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 750.12 (TPS) </td>
      <td> 738.79 (TPS) </td>
      <td> 1813 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">76B</a></td>
      <td> 全参微调 </td>
      <td> 8x16 </td>
      <td> BF16 </td>
      <td> 214 (TPS) </td>
      <td> 191 (TPS) </td>
      <td> 1813 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/internvl2.5">InternVL 2.5</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2_5-78B">78B</a></td>
      <td> 微调 </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> 228.33 </td>
      <td> / </td>
      <td> 1896 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/internvl3">InternVL 3.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3-8B">8B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2344.58 (TPS) </td>
      <td> 2211.93 (TPS) </td>
      <td> 2653 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3-78B">78B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
      <td> 228.82 (TPS) </td>
      <td> 283.15 (TPS) </td>
      <td> 1932 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/internvl3.5">InternVL 3.5</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B-Instruct">30B</a></td>
      <td> 微调 </td>
      <td> 1x8 (A3)  </td>
      <td> BF16 </td>
      <td> 52.76 (TPS) </td>
      <td> 47.73 (TPS) </td>
      <td> 201 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/qwen2vl">Qwen2-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct">2B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2941.17 (TPS) </td>
      <td> 3004.04 (TPS) </td>
      <td> 689 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1143.74 (TPS) </td>
      <td> 1004.22 (TPS) </td>
      <td> 689 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct">72B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
      <td> 261.25 (TPS) </td>
      <td> 257.63 (TPS) </td>
      <td> 689 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/qwen2.5vl">Qwen2.5-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct">3B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2047.19 (TPS) </td>
      <td> 1876.66 (TPS) </td>
      <td> 689 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1620.87 (TPS) </td>
      <td> 1091.20 (TPS) </td>
      <td> 689 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct">32B</a></td>
      <td> 微调 </td>
      <td> 2x8 </td>
      <td> BF16 </td>
      <td> 257.50 (TPS) </td>
      <td> / </td>
      <td> 689 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">72B</a></td>
      <td> 微调 </td>
      <td> 4x8 (A3) </td>
      <td> BF16 </td>
      <td> 322.96 (TPS) </td>
      <td> 256.28 (TPS) </td>
      <td> 689 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/qwen3vl">Qwen3-VL</a></td>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 8B </a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 146.54 (TPS)</td>
      <td> 129.71 (TPS)</td>
      <td> 179 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 30B </a></td>
      <td> 微调 </td>
      <td> 1x8 (A3) </td>
      <td> BF16 </td>
      <td> 179.57 (TPS) </td>
      <td> / </td>
      <td> 185 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe"> 235B </a></td>
      <td> 微调 </td>
      <td> 16x8 (A3) </td>
      <td> BF16 </td>
      <td> 598.05 (TPS) </td>
      <td> / </td>
      <td> 16116 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/qwen2.5omni">Qwen2.5-Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 575.01 (TPS) </td>
      <td> 534.28 (TPS) </td>
      <td> 296 </td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/qwen3omni">Qwen3-Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct">30B</a></td>
      <td> 微调 </td>
      <td> 2x4 (A3) </td>
      <td> BF16 </td>
      <td> 131.3 (TPS) </td>
      <td> 16.4 (TPS) </td>
      <td> 288 </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/magistral-2509">Magistral-Small-2509</a></td>
      <td><a href="https://huggingface.co/mistralai/Magistral-Small-2509">24B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1.843 (SPS) </td>
      <td> 1.185 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td> 语音识别 </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/whisper">Whisper</a></td>
      <td><a href="https://github.com/openai/whisper">1.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 93.38 (SPS) </td>
      <td> 109.23 (SPS) </td>
      <td> / </td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td> 语音生成 </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/fsdp2/cosyvoice3">CosyVoice3</a></td>
      <td><a href="https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512">0.5B</a></td>
      <td> 预训练 </td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 290.91 (SPS) </td>
      <td> 326.11 (SPS) </td>
      <td> 24 </td>
      <td>【Test】</td>
    </tr>
    </tbody>
</table>

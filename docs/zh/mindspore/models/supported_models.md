# MindSpore框架模型支持列表

MindSpore框架根据模型实现方式分为稠密模型和稀疏模型，详情请查看以下支持列表。

> **表格字段说明**：
>
> - **模型**：模型名称
> - **下载链接**：模型权重下载地址，点击可直接访问Hugging Face等模型仓库
> - **脚本位置**：模型在本项目中的训练脚本路径，可用于快速定位和使用模型
> - **序列长度**：支持的最大文本序列长度
> - **训练后端**：仅支持以mcore方式实现的模型，对Legacy方式实现的模型无支持计划。
> - **集群规模**：模型训练时推荐使用的集群规模配置，格式为"节点数×卡数"
> - **支持版本**：最终支持的维护版本，空白表示从上线起到当前master分支均在维护

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/THUDM">ChatGLM3</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/chatglm3-6b-base/tree/main">6B</a></td>
      <td rowspan="2"><a href="../../../../examples/mindspore/chatglm3">chatglm3</a></td>
      <td>8K</td>
      <th>Mcore</th>
      <td >1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>64K</td>
      <th>Mcore</th>
      <td >2x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/THUDM">GLM4</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/glm-4-9b">9B</a></td>
      <td rowspan="3"><a href="../../../../examples/mindspore/glm4">glm4</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      </tr>
    <tr>
      <td> 32K </td>
      <th>Mcore</th>
      <td> 2x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/zai-org/GLM-4-32B-0414">32B</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/codellama">CodeLlama</a></td>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-hf/tree/main">34B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/codellama">codellama</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td> 2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/internlm">InternLM2</a> </td>
      <td rowspan="2"> <a href="https://huggingface.co/Internlm/Internlm2-chat-20b/tree/main">20B</a> </td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/internlm2">internlm2</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
      <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td rowspan="3"> <a href="https://huggingface.co/internlm">InternLM2.5</a> </td>
      <td><a href="https://huggingface.co/internlm/internlm2_5-1_8b/tree/main">1.8B</a></td>
      <td rowspan="3"><a href="../../../../examples/mindspore/internlm25">internlm25</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-7b/tree/main">7B</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-20b/tree/main">20B</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="6"><a href="https://huggingface.co/meta-llama">LLaMA2</a></td>
      <td rowspan="3"><a href="https://huggingface.co/daryl149/llama-2-7b-hf/tree/main">7B</a></td>
      <td rowspan="6"><a href="../../../../examples/mindspore/llama2">llama2</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>16K</td>
      <th>Mcore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td>32K</td>
      <th>Mcore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/NousResearch/Llama-2-13b-hf/tree/main">13B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/tree/main">34B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/meta-llama/Llama-2-70b-hf">70B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">LLaMA3</a></td>
      <td><a href="https://huggingface.co/unsloth/llama-3-8b/tree/main">8B</a></td>
      <td rowspan="2"><a href="../../../../examples/mindspore/llama3">llama3</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/v2ray/Llama-3-70B/tree/main">70B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://modelscope.cn/organization/LLM-Research">LLaMA3.1</a></td>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B">8B</a></td>
      <td rowspan="3"><a href="../../../../examples/mindspore/llama31">llama31</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>128K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>支持中</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B">70B</a></td>
      <td>8K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">LLaMA3.2</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-1B/tree/main">1B</a></td>
      <td rowspan="2"><a href="../../../../examples/mindspore/llama32">llama32</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-3B/tree/main">3B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/meta-llama">LLaMA3.3</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/tree/main">70B-Instruct</a></td>
      <td rowspan="1"><a href="../../../../examples/mindspore/llama33">llama33</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B/tree/main">0.5B</a> </td>
      <td rowspan="9"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/qwen15">qwen15</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="8"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main">1.8B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-4B/tree/main">4B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-7B/tree/main">7B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
    </tr>  
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-14B/tree/main">14B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-32B/tree/main">32B</a> </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-72B/tree/main">72B</a> </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-110B/tree/main">110B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 8x8 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">CodeQwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/CodeQwen1.5-7B">7B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td rowspan="1"> <a href="https://huggingface.co/Qwen/Qwen2-7B/tree/main">7B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/qwen2">qwen2</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="11"><a href="https://huggingface.co/Qwen">Qwen2.5</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/main">0.5B</a></td>
      <td rowspan="11"><a href="../../../../examples/mindspore/qwen25">qwen25</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main">1.5B</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B/tree/main">3B</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-7B/tree/main">7B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-14B/tree/main">14B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/">2.3.0</a> </td>
    </tr>
    <tr>
      <td>32K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-32B/tree/main">32B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td>32K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-72B/tree/main">72B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td>32K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="6"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base">0.6B</a></td>
      <td rowspan="6"><a href="../../../../examples/mindspore/qwen3">qwen3</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td rowspan="5"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-1.7B-Base">1.7B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-4B-Base">4B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-8B-Base">8B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-14B-Base">14B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">32B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/01-ai">Yi</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-34B/tree/main">34B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/yi">yi</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/01-ai">Yi1.5</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-6B/tree/main">6B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/yi15">yi15</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-9B/tree/main">9B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-34B/tree/main">34B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>2x8</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/mistralai">Mistral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main">7B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mindspore/mistral">mistral</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma</a></td>
      <td><a href="https://huggingface.co/google/gemma-2b/tree/main">2B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/gemma">gemma</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-2b/tree/main">7B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma2</a></td>
      <td><a href="https://huggingface.co/google/gemma-2-9b/tree/main">9B</a></td>
      <td rowspan="2"><a href="../../../../examples/mindspore/gemma2">gemma2</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-2-27b/tree/main">27B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>2x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/xai-org/grok-1/tree/main">grok-1</a></td>
      <td><a href="https://huggingface.co/xai-org/grok-1/tree/main">40B</a></td>
      <td><a href="../../../../examples/mindspore/grok1">grok-1</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/tree/main">2B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/tree/main">mini-instruct</a> </td>
      <td rowspan="1"><a href="../../../../examples/mindspore/phi35">phi35</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/deepseek-ai">DeepSeek-R1-Distill-Qwen</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">1.5B</a></td>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/deepseek_r1_distill_qwen">deepseek_r1_distill_qwen</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="4"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">7B</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B">14B</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B">32B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 2x8 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/deepseek-ai">DeepSeek-R1-Distill-LLaMA</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B">8B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/deepseek_r1_distill_llama">deepseek_r1_distill_llama</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="2">  <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.2.0</a>  </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B">70B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 4x8 </td>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Base">30B</a></td>
      <td rowspan="2"> <a href="../../../../examples/mindspore/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B">235B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 16x16 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-57B-A14B/tree/main">57B-A14B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/qwen2_moe">qwen2_moe</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/mistralai">Mixtral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">8x7B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/mixtral">mixtral</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/tree/main">8x22B</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>8x8</td>
    </tr>
    <tr>
      <td> 64K</td>
      <th>Mcore</th>
      <td>8x8</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">DeepSeek-V2</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/deepseek2">deepseek2</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base">DeepSeek-V2-coder</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/deepseek2_coder">deepseek2_coder</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite">DeepSeek-V2-Lite</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main">16B</a></td>
      <td><a href="../../../../examples/mindspore/deepseek2_lite">deepseek2_lite</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5">DeepSeek-V2.5</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5/tree/main">236B</a></td>
      <td>deepseek25</td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">DeepSeek-V3</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main">671B</a></td>
      <td><a href="../../../../examples/mindspore/deepseek3">deepseek3</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 64x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-MoE-8x2B/tree/main">8x2B</a> </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-MoE-instruct">MoE-instruct</a> </td>
      <td><a href="../../../../examples/mindspore/phi35">phi35</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/zai-org/GLM-4.5">GLM4.5</a></td>
      <td><a href="https://huggingface.co/zai-org/GLM-4.5/tree/main">106B</a></td>
      <td><a href="../../../../examples/mindspore/glm45-moe">glm45-moe</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 8x16 </td>
      <td></td>
    </tr>
  </tbody>
</table>

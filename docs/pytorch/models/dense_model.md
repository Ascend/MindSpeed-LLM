## MindSpeed-LLM 预置稠密大模型

认证`【Pass】`表示经过昇腾官方版本测试的模型。`【Test】`表示模型处于内部测试阶段，未完成充分的性能测试和验收，在实际使用中可能存在未被发现的问题，待后续充分验证后会发布正式版本。相关使用问题可反馈至[MindSpeed-LLM/issues](https://gitcode.com/ascend/MindSpeed-LLM/issues)。

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>脚本位置</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
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
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
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
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/BAAI/Aquila2-34B/tree/main">34B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 2x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
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
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main">13B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/baichuan-inc">Baichuan2</a></td>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main">7B</a></td>
      <td rowspan="2"><a href="../../../examples/mcore/baichuan2">baichuan2</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/tree/main">13B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td> 1x8</td>
      <td> </td>
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
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/bigscience/bloom/tree/main">176B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td >12x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/THUDM">ChatGLM3</a></td>
      <td rowspan="3"><a href="https://huggingface.co/THUDM/chatglm3-6b-base/tree/main">6B</a></td>
      <td rowspan="3"><a href="../../../examples/mcore/chatglm3">chatglm3</a></td>
      <td>8K</td>
      <th>Mcore</th>
      <td >1x8</td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td>32K</td>
      <th>Mcore</th>
      <td >1x8</td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td>64K</td>
      <th>Mcore</th>
      <td >2x8</td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/THUDM">GLM4</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/glm-4-9b">9B</a></td>
      <td rowspan="2"><a href="../../../examples/mcore/glm4">glm4</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td >65B</td>
      <td>2K</td>
      <th>Legacy</th>
      <td >4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
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
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="3"> <a href="https://huggingface.co/internlm">InternLM2.5</a> </td>
      <td><a href="https://huggingface.co/internlm/internlm2_5-1_8b/tree/main">1.8B</a></td>
      <td rowspan="3"><a href="../../../examples/mcore/internlm25">internlm25</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-7b/tree/main">7B</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-20b/tree/main">20B</a></td>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/internlm">InternLM3</a></td>
      <td><a href="https://huggingface.co/internlm/internlm3-8b-instruct/tree/main">8B</a></td>
      <td><a href="../../../examples/mcore/internlm3">internlm3</a></td>
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
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/ruibin-wang/llama-13b-hf">13B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/pinkmanlove/llama-33b-hf/tree/main">33B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/pinkmanlove/llama-65b-hf">65B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="5"><a href="https://huggingface.co/meta-llama">LLaMA2</a></td>
      <td><a href="https://huggingface.co/daryl149/llama-2-7b-hf/tree/main">7B</a></td>
      <td rowspan="5"><a href="../../../examples/mcore/llama2">llama2</a></td>
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
      <td> </td>
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
      <td rowspan="2"><a href="../../../examples/mcore/llama3">llama3</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/v2ray/Llama-3-70B/tree/main">70B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://modelscope.cn/organization/LLM-Research">LLaMA3.1</a></td>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B">8B</a></td>
      <td rowspan="8"><a href="../../../examples/mcore/llama31">llama31</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td>128K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 50B </td>
      <td> 128K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B">70B</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 128K </td>
      <th> Mcore </th>
      <td> 24x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td> 200B </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td> </td>
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
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">LLaMA3.2</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-1B/tree/main">1B</a></td>
      <td rowspan="2"><a href="../../../examples/mcore/llama32">llama32</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-3B/tree/main">3B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/meta-llama">LLaMA3.3</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/tree/main">70B-Instruct</a></td>
      <td rowspan="1"><a href="../../../examples/mcore/llama33">llama33</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 4x8</td>
      <td> </td>
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
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen-14B/tree/main">14B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen-72B/tree/main">72B</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>16x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B/tree/main">0.5B</a> </td>
      <td rowspan="9"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen15">qwen15</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="9"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main">1.8B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-4B/tree/main">4B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-7B/tree/main">7B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-14B/tree/main">14B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-32B/tree/main">32B</a> </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 4x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-72B/tree/main">72B</a> </td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td>【GTS】</td>    
      <td>【Pass】</td>
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
      <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td rowspan="2"> <a href="https://huggingface.co/Qwen/Qwen2-0.5B/tree/main">0.5B</a> </td>
      <td rowspan="8"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen2">qwen2</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td rowspan="8"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td rowspan="2"> <a href="https://huggingface.co/Qwen/Qwen2-1.5B/tree/main">1.5B</a> </td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> 32K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2-7B/tree/main">7B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td> 32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2-72B/tree/main">72B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td>【GTS】</td>
      <td>【Pass】</td>
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
      <td rowspan="7"><a href="../../../examples/mcore/qwen25">qwen25</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main">1.5B</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B/tree/main">3B</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-7B/tree/main">7B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-14B/tree/main">14B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-32B/tree/main">32B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> </td>
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
      <td rowspan="6"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base">0.6B</a></td>
      <td rowspan="6"><a href="../../../examples/mcore/qwen3">qwen3</a></td>
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
      <td rowspan="1"><a href="https://huggingface.co/collections/Qwen/qwq-674762b79b75eac01735070a">QwQ</a></td>
      <td><a href="https://huggingface.co/Qwen/QwQ-32B/tree/main">32B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwq">qwq</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
      <td>【OpenMind】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-34B/tree/main">34B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td rowspan="2"><a href="../../../examples/mcore/gemma2">gemma2</a></td>
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
      <td rowspan="1"><a href="https://huggingface.co/xai-org/grok-1/tree/main">grok-1</a></td>
      <td><a href="https://huggingface.co/xai-org/grok-1/tree/main">40B</a></td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/mcore/grok1">grok-1</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/">2.0.0</a></td>
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
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/tree/main">mini-instruct</a> </td>
      <td rowspan="1"><a href="../../../examples/mcore/phi35">phi35</a></td>
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
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td rowspan="4"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td><a href="../../../examples/mcore/seed_oss">seed_oss</a></td>
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
      <td><a href="../../../examples/mcore/magistral">magistral</a></td>
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
      <td><a href="../../../examples/mcore/plm">plm</a></td>
      <td> 2K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>

## 社区BUG列表

1. Baichuan-13B: 在任务执行过程中如果出现报错：AttributeError: 'BaichuanTokenizer’ object has no attribute 'sp_model'，请执行下面命令解决这个问题：

    ```shell
    pip3 install transformers==4.32.0 --force
    ```

2. GPT: GPT词表文件与常规模型不同：

    ```shell
    mkdir vocab_file 
    cd vocab_file
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    cd ..

    # 处理成训练数据
    python ./preprocess_data.py \
        --input ./dataset/ \
        --output-prefix ./dataset/gpt_text_sentence \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file ./vocab_file/gpt2-vocab.json \
        --merge-file ./vocab_file/gpt2-merges.txt \
        --append-eod \
        --workers 4 \
        --log-interval 1000

    # 请根据真实存放路径配置预训练脚本以下参数
    VOCAB_FILE="./vocab_file/gpt2-vocab.json"   # 词表
    MERGE_FILE="./vocab_file/gpt2-merges.txt"   # BPE 合并表
    DATA_PATH="./dataset/gpt_text_sentence"     # 数据路径
    ```

3. Bloom-176B: config.json中同字段对应的key值与其他模型不一致，将文件中的n_embed改为hidden_size， 将num_attention_heads修改为n_head
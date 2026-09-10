# Supported Models for the MindSpore Framework

MindSpore models fall into dense models and sparse models based on how they are implemented. See the support list below for details.

> **Table field descriptions**:
>
> - **Model**: Model name.
> - **Download Link**: Download address for the model weights. Click it to visit repositories such as Hugging Face directly.
> - **Script Location**: Training script path for the model in this project. You can use it to quickly locate and run the model.
> - **Sequence Length**: Maximum supported text sequence length.
> - **Training Backend**: Only models implemented with `mcore` are supported. Legacy implementations are not planned for support.
> - **Cluster Size**: Recommended cluster configuration for model training, in the format "number of nodes × number of devices".
> - **Supported Version**: Final supported maintenance version. A blank value means the model is maintained from launch through the current master branch.

## Dense Models

Dense models are a traditional deep learning model structure. Their neurons are densely connected, and most or all neurons in each layer connect to all neurons in the next layer. These models are simple and relatively straightforward to train, but they have a large parameter count and higher computational cost.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Download Link</th>
      <th>Script Location</th>
      <th>Sequence Length</th>
      <th>Training Backend</th>
      <th>Cluster Size</th>
      <th>Supported Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/THUDM">ChatGLM3</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/chatglm3-6b-base/tree/main">6B</a></td>
      <td rowspan="2"><a href="../../../../examples/mindspore/chatglm3">chatglm3</a></td>
      <td>8K</td>
      <th>MCore</th>
      <td >1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>64K</td>
      <th>MCore</th>
      <td >2x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/THUDM">GLM4</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/glm-4-9b">9B</a></td>
      <td rowspan="3"><a href="../../../../examples/mindspore/glm4">glm4</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      </tr>
    <tr>
      <td> 32K </td>
      <th>MCore</th>
      <td> 2x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/zai-org/GLM-4-32B-0414">32B</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/codellama">CodeLlama</a></td>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-hf/tree/main">34B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/codellama">codellama</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td> 2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/internlm">InternLM2</a> </td>
      <td rowspan="2"> <a href="https://huggingface.co/Internlm/Internlm2-chat-20b/tree/main">20B</a> </td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/internlm2">internlm2</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
      <tr>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td rowspan="3"> <a href="https://huggingface.co/internlm">InternLM2.5</a> </td>
      <td><a href="https://huggingface.co/internlm/internlm2_5-1_8b/tree/main">1.8B</a></td>
      <td rowspan="3"><a href="../../../../examples/mindspore/internlm25">internlm25</a></td>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-7b/tree/main">7B</a></td>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-20b/tree/main">20B</a></td>
      <td> 32K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="6"><a href="https://huggingface.co/meta-llama">Llama-2</a></td>
      <td rowspan="3"><a href="https://huggingface.co/daryl149/llama-2-7b-hf/tree/main">7B</a></td>
      <td rowspan="6"><a href="../../../../examples/mindspore/llama2">llama2</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>16K</td>
      <th>MCore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td>32K</td>
      <th>MCore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/NousResearch/Llama-2-13b-hf/tree/main">13B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/tree/main">34B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>2x8</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/meta-llama/Llama-2-70b-hf">70B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">Llama-3</a></td>
      <td><a href="https://huggingface.co/unsloth/llama-3-8b/tree/main">8B</a></td>
      <td rowspan="2"><a href="../../../../examples/mindspore/llama3">llama3</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/v2ray/Llama-3-70B/tree/main">70B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://modelscope.cn/organization/LLM-Research">Llama-3.1</a></td>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B">8B</a></td>
      <td rowspan="3"><a href="../../../../examples/mindspore/llama31">llama31</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>128K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td>In progress</td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B">70B</a></td>
      <td>8K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">Llama-3.2</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-1B/tree/main">1B</a></td>
      <td rowspan="2"><a href="../../../../examples/mindspore/llama32">llama32</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-3B/tree/main">3B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/meta-llama">Llama-3.3</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/tree/main">70B-Instruct</a></td>
      <td rowspan="1"><a href="../../../../examples/mindspore/llama33">llama33</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B/tree/main">0.5B</a> </td>
      <td rowspan="9"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/qwen15">qwen15</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="8"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main">1.8B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-4B/tree/main">4B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-7B/tree/main">7B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-14B/tree/main">14B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-32B/tree/main">32B</a> </td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 4x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-72B/tree/main">72B</a> </td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 8x8 </td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-110B/tree/main">110B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 8x8 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">CodeQwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/CodeQwen1.5-7B">7B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td rowspan="1"> <a href="https://huggingface.co/Qwen/Qwen2-7B/tree/main">7B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/qwen2">qwen2</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="11"><a href="https://huggingface.co/Qwen">Qwen2.5</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/main">0.5B</a></td>
      <td rowspan="11"><a href="../../../../examples/mindspore/qwen25">qwen25</a></td>
      <td> 32K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main">1.5B</a></td>
      <td> 32K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B/tree/main">3B</a></td>
      <td> 32K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-7B/tree/main">7B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>32K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-14B/tree/main">14B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td>32K</td>
      <th>MCore</th>
      <td>2x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-32B/tree/main">32B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td>32K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-72B/tree/main">72B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td>32K</td>
      <th>MCore</th>
      <td>8x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="6"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base">0.6B</a></td>
      <td rowspan="6"><a href="../../../../examples/mindspore/qwen3">qwen3</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td rowspan="5"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-1.7B-Base">1.7B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-4B-Base">4B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-8B-Base">8B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-14B-Base">14B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">32B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/01-ai">Yi</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-34B/tree/main">34B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/yi">yi</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>2x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/01-ai">Yi1.5</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-6B/tree/main">6B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/yi15">yi15</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-9B/tree/main">9B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-34B/tree/main">34B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>2x8</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/mistralai">Mistral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main">7B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mindspore/mistral">mistral</a></td>
      <td> 32K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma</a></td>
      <td><a href="https://huggingface.co/google/gemma-2b/tree/main">2B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/gemma">gemma</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-2b/tree/main">7B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma2</a></td>
      <td><a href="https://huggingface.co/google/gemma-2-9b/tree/main">9B</a></td>
      <td rowspan="2"><a href="../../../../examples/mindspore/gemma2">gemma2</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-2-27b/tree/main">27B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>2x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/xai-org/grok-1/tree/main">grok-1</a></td>
      <td><a href="https://huggingface.co/xai-org/grok-1/tree/main">40B</a></td>
      <td><a href="../../../../examples/mindspore/grok1">grok-1</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>4x8</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/tree/main">2B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/tree/main">mini-instruct</a> </td>
      <td rowspan="1"><a href="../../../../examples/mindspore/phi35">phi35</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/deepseek-ai">DeepSeek-R1-Distill-Qwen</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">1.5B</a></td>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/deepseek_r1_distill_qwen">deepseek_r1_distill_qwen</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="4"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">7B</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B">14B</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B">32B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 2x8 </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/deepseek-ai">DeepSeek-R1-Distill-Llama</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B">8B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/deepseek_r1_distill_llama">deepseek_r1_distill_llama</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="2">  <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.2.0</a>  </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B">70B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 4x8 </td>
    </tr>
  </tbody>
</table>

## Sparse Models

Sparse models use a sparsely connected neuron structure, where only a small number of neurons are connected. A typical sparse model is the Mixture of Experts (MoE) model, which contains multiple expert networks and activates only a subset of experts during each training step. This design can significantly reduce the parameter count and computational complexity, improve training efficiency, and is especially suitable for large datasets and complex tasks. However, sparse model training also has drawbacks. It can easily suffer from unbalanced expert loads, which leads to unstable training.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Download Link</th>
      <th>Script Location</th>
      <th>Sequence Length</th>
      <th>Training Backend</th>
      <th>Cluster Size</th>
      <th>Supported Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Base">30B</a></td>
      <td rowspan="2"> <a href="../../../../examples/mindspore/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B">235B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 16x16 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-57B-A14B/tree/main">57B-A14B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/qwen2_moe">qwen2_moe</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>8x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/mistralai">Mixtral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">8x7B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/mixtral">mixtral</a></td>
      <td> 32K</td>
      <th>MCore</th>
      <td>8x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/tree/main">8x22B</a></td>
      <td> 32K</td>
      <th>MCore</th>
      <td>8x8</td>
    </tr>
    <tr>
      <td> 64K</td>
      <th>MCore</th>
      <td>8x8</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">DeepSeek-V2</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/deepseek2">deepseek2</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base">DeepSeek-V2-coder</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/deepseek2_coder">deepseek2_coder</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite">DeepSeek-V2-Lite</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main">16B</a></td>
      <td><a href="../../../../examples/mindspore/deepseek2_lite">deepseek2_lite</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5">DeepSeek-V2.5</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5/tree/main">236B</a></td>
      <td>deepseek25</td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 20x8 </td>
      <td>In progress</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">DeepSeek-V3</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main">671B</a></td>
      <td><a href="../../../../examples/mindspore/deepseek3">deepseek3</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 64x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-MoE-8x2B/tree/main">8x2B</a> </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mindspore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-MoE-instruct">MoE-instruct</a> </td>
      <td><a href="../../../../examples/mindspore/phi35">phi35</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/zai-org/GLM-4.5">GLM4.5</a></td>
      <td><a href="https://huggingface.co/zai-org/GLM-4.5/tree/main">106B</a></td>
      <td><a href="../../../../examples/mindspore/glm45-moe">glm45-moe</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 8x16 </td>
      <td></td>
    </tr>
  </tbody>
</table>

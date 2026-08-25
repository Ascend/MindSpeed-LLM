# Supported Models

PyTorch models are divided into three categories based on architecture: dense models, sparse models, and state space models (SSMs). See the support lists below for details.

**Table fields**

| Field | Description |
| :--- | :--- |
| **Model** | Model name |
| **Download Link** | Model weight download URL. Click to visit model repositories such as Hugging Face. |
| **Script Path** | The training script path for the model in this project. Use it to quickly locate and run the model. See [Model Script Environment Variables](../features/mcore/environment_variable.md) for details. |
| **Sequence Length** | Maximum supported text sequence length. |
| **Training Backend** | Supports Legacy, MCore, and FSDP2 implementations.<br>• Legacy: Legacy implementation<br>• MCore: Current recommended Megatron implementation<br>• FSDP2: Distributed training implementation recommended by PyTorch |
| **Cluster Size** | Recommended cluster size configuration for model training, in the format "number of nodes × number of devices". |
| **Supported Version** | Final supported maintenance version. A blank entry means the model has been maintained from launch through the current master branch. |
| **Contributor** | Model source. |
| **Certification** | [Pass] has passed official tests. [Test] is under internal testing. If you find issues, please report them on [issues](https://gitcode.com/ascend/MindSpeed-LLM/issues). |

> [!NOTE]
>
> - The repository supports the Ascend 950 products, Atlas A3 training series products, and Atlas A2 training series products. You can obtain hardware information from [Hardware Information Query](https://www.hiascend.com/en/cann/download). It also requires at least 64 GB of on-chip memory on a single NPU. After you [install the driver firmware](../training/install_guide.md), you can use `npu-smi info` to check the on-chip memory capacity of a single NPU.
> - [Model training example scripts](../../../../examples) are available. Training scripts are named `pretrain_model_name_parameter_sequence_length_training_scheme_training_product.sh`, and fine-tuning scripts are named `tune_model_name_parameter_sequence_length_training_scheme_training_product.sh`.
> - Among the provided training scripts, [model training example scripts](../../../../examples) are designed for demonstrating normal weight loading and training, while [model training high-performance scripts](../../../../tests/poc) are designed for demonstrating high performance under fixed routes. You can choose the appropriate ones based on your usage scenario.
> - You can check the cluster size in the model training example scripts. `NNODES` indicates the number of machines, and `NPUS_PER_NODE` indicates the number of NPUs on each machine. The following example uses 32 nodes, each with 16 NPUs:
>
>   ```bash
>   NNODES=32
>   NPUS_PER_NODE=16
>   ```
>
> - If you cannot access the Hugging Face community to download weight resources, you are advised to visit [ModelScope](https://www.modelscope.cn/models) to download the weights. Pay attention to the correctness and security of the files to be downloaded.

## Dense Models

Dense models are traditional deep learning architectures. Their neurons are densely connected, and most or all neurons in each layer connect to all neurons in the next layer. These models are simple and relatively straightforward to train, but they have more parameters and higher computational cost.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Download Link</th>
      <th>Script Path</th>
      <th>Sequence Length</th>
      <th>Training Backend</th>
      <th>Cluster Size</th>
      <th>Supported Version</th>
      <th>Contributor</th>
      <th>Certification</th>
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
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/collections/BAAI/aquila-6698657124de09d10cd7a83f">Aquila2</a></td>
      <td><a href="https://huggingface.co/BAAI/Aquila2-7B/tree/main">7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/aquila2">aquila2</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/BAAI/Aquila2-34B/tree/main">34B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 2x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/baichuan-inc">Baichuan</a></td>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan-7B/tree/main">7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/baichuan">baichuan</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan-13B-Base/tree/main">13B</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/baichuan-inc">Baichuan2</a></td>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/tree/main">7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/baichuan2">baichuan2</a></td>
      <td>4K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/baichuan-inc/Baichuan2-13B-Base/tree/main">13B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/bigscience">Bloom</a></td>
      <td><a href="https://huggingface.co/bigscience/bloom-7b1/tree/main">7B1</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/bloom">bloom</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td> 1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/bigscience/bloom/tree/main">176B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td >2x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/THUDM">ChatGLM3</a></td>
      <td rowspan="3"><a href="https://huggingface.co/THUDM/chatglm3-6b-base/tree/main">6B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/chatglm3">chatglm3</a></td>
      <td>8K</td>
      <th>MCore</th>
      <td >1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td>32K</td>
      <th>MCore</th>
      <td >1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td>64K</td>
      <th>MCore</th>
      <td >2x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/THUDM">GLM4</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/glm-4-9b">9B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/glm4">glm4</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th>MCore</th>
      <td> 2x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/codellama">CodeLlama</a></td>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-hf/tree/main">34B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/codellama">codellama</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td> 2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/internlm">InternLM</a></td>
      <td><a href="https://huggingface.co/internlm/internlm-7b/tree/main">7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/intern">intern</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td >65B</td>
      <td>2K</td>
      <th>Legacy</th>
      <td >4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/internlm">InternLM2</a> </td>
      <td rowspan="2"> <a href="https://huggingface.co/Internlm/Internlm2-chat-20b/tree/main">20B</a> </td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/internlm2">internlm2</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="3"> <a href="https://huggingface.co/internlm">InternLM2.5</a> </td>
      <td><a href="https://huggingface.co/internlm/internlm2_5-1_8b/tree/main">1.8B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/internlm25">internlm25</a></td>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-7b/tree/main">7B</a></td>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/internlm/internlm2_5-20b/tree/main">20B</a></td>
      <td> 32K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/internlm">InternLM3</a></td>
      <td><a href="https://huggingface.co/internlm/internlm3-8b-instruct/tree/main">8B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0/examples/mcore/internlm3">internlm3</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/26.0.0">26.0.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/meta-llama">Llama-</a></td>
      <td><a href="https://huggingface.co/huggyllama/llama-7b/tree/main">7B</a></td>
      <td rowspan="4"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/llama">llama</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/ruibin-wang/llama-13b-hf">13B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/pinkmanlove/llama-33b-hf/tree/main">33B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/pinkmanlove/llama-65b-hf">65B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="5"><a href="https://huggingface.co/meta-llama">Llama-2</a></td>
      <td><a href="https://huggingface.co/daryl149/llama-2-7b-hf/tree/main">7B</a></td>
      <td rowspan="5"><a href="../../../../examples/mcore/llama2">llama2</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/26.0.0">26.0.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/NousResearch/Llama-2-13b-hf/tree/main">13B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/tree/main">34B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama/Llama-2-70b-hf">70B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/26.0.0">26.0.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 128K </td>
      <th> MCore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/26.0.0">26.0.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">Llama-3</a></td>
      <td><a href="https://huggingface.co/unsloth/llama-3-8b/tree/main">8B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/llama3">llama3</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/v2ray/Llama-3-70B/tree/main">70B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>4x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://modelscope.cn/organization/LLM-Research">Llama-3.1</a></td>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B">8B</a></td>
      <td rowspan="8"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0/examples/mcore/llama31">llama31</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td>128K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 50B </td>
      <td> 128K </td>
      <th> MCore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B">70B</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 4x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 128K </td>
      <th> MCore </th>
      <td> 24x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 200B </td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"> 405B </td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/26.0.0">26.0.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 128K </td>
      <th> MCore </th>
      <td> 36x8 </td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">Llama-3.2</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-1B/tree/main">1B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/llama32">llama32</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-3B/tree/main">3B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/meta-llama">Llama-3.3</a></td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct/tree/main">70B-Instruct</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/llama33">llama33</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 4x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/Qwen">Qwen</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen-7B/tree/main">7B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/qwen">qwen</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen-14B/tree/main">14B</a></td>
      <td>2K</td>
      <th>Legacy</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen-72B/tree/main">72B</a></td>
      <td> 8K </td>
      <th>Legacy</th>
      <td>16x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-0.5B/tree/main">0.5B</a> </td>
      <td rowspan="9"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen15">qwen15</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="9"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main">1.8B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-4B/tree/main">4B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-7B/tree/main">7B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-14B/tree/main">14B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-32B/tree/main">32B</a> </td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 4x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-72B/tree/main">72B</a> </td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 8x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-110B/tree/main">110B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 8x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">CodeQwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/CodeQwen1.5-7B">7B</a> </td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="8"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td rowspan="2"> <a href="https://huggingface.co/Qwen/Qwen2-0.5B/tree/main">0.5B</a> </td>
      <td rowspan="8"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen2">qwen2</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td rowspan="8"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"> <a href="https://huggingface.co/Qwen/Qwen2-1.5B/tree/main">1.5B</a> </td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2-7B/tree/main">7B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 32K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2-72B/tree/main">72B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 32K </td>
      <th> MCore </th>
      <td> 16x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="7"><a href="https://huggingface.co/Qwen">Qwen2.5</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/main">0.5B</a></td>
      <td rowspan="7"><a href="../../../../examples/mcore/qwen25">qwen25</a></td>
      <td> 32K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main">1.5B</a></td>
      <td> 32K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B/tree/main">3B</a></td>
      <td> 32K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-7B/tree/main">7B</a></td>
      <td>32K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-14B/tree/main">14B</a></td>
      <td>32K</td>
      <th>MCore</th>
      <td>2x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-32B/tree/main">32B</a></td>
      <td>32K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-72B/tree/main">72B</a></td>
      <td>32K</td>
      <th>MCore</th>
      <td>16x8</td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="7"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base">0.6B</a></td>
      <td rowspan="6"><a href="../../../../examples/mcore/qwen3">qwen3</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-1.7B-Base">1.7B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-4B-Base">4B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-8B-Base">8B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-14B-Base">14B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">32B</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-32B">32B</a></td>
      <td><a href="../../../../examples/fsdp2/qwen3">qwen3</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 1x16 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/collections/Qwen/qwq-674762b79b75eac01735070a">QwQ</a></td>
      <td><a href="https://huggingface.co/Qwen/QwQ-32B/tree/main">32B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwq">qwq</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/Qwen">Qwen2.5-Math</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Math-1.5B/tree/main">1.5B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen25_math">qwen25_math</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Math-7B/tree/main">7B</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Math-72B/tree/main">72B</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td>4x8</td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
 <tr>
   <td rowspan="1"><a href="https://huggingface.co/Qwen">CodeQwen2.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen2.5-Coder-7B">7B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen25_coder">qwen25_coder</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[China Mobile Cloud]</td>
      <td>[Test]</td>
    </tr>
 <tr>
      <td rowspan="2"><a href="https://huggingface.co/01-ai">Yi</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-9B/tree/main">9B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/legacy/yi">yi</a></td>
      <td> 4K</td>
      <th>Legacy</th>
      <td>1x4</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[OpenMind]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-34B/tree/main">34B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>2x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/01-ai">Yi1.5</a></td>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-6B/tree/main">6B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/yi15">yi15</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-9B/tree/main">9B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/01-ai/Yi-1.5-34B/tree/main">34B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>2x8</td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/mistralai">Mistral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main">7B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/mistral">mistral</a></td>
      <td> 32K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma</a></td>
      <td><a href="https://huggingface.co/google/gemma-2b/tree/main">2B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/gemma">gemma</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-7b">7B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/google">Gemma2</a></td>
      <td><a href="https://huggingface.co/google/gemma-2-9b/tree/main">9B</a></td>
      <td rowspan="2"><a href="../../../../examples/mcore/gemma2">gemma2</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>1x8</td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/google/gemma-2-27b/tree/main">27B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>2x8</td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/tree/main">2B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/openbmb/MiniCPM3-4B/tree/main">MiniCPM3</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM3-4B/tree/main">4B</a> </td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/minicpm3">minicpm3</a></td>
      <td> 32K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct/tree/main">mini-instruct</a> </td>
      <td rowspan="1"><a href="../../../../examples/mcore/phi35">phi35</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/deepseek-math-7b-base">DeepSeek-Math</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/deepseek-math-7b-base">7B</a></td>
      <td rowspan="1"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek_math">deepseek_math</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://huggingface.co/deepseek-ai">DeepSeek-R1-Distill-Qwen</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">1.5B</a></td>
      <td rowspan="4"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek_r1_distill_qwen">deepseek_r1_distill_qwen</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="4"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B">7B</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B">14B</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B">32B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 2x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/deepseek-ai">DeepSeek-R1-Distill-Llama-</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B">8B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek_r1_distill_llama">deepseek_r1_distill_llama</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="2"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B">70B</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 4x8 </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/ByteDance-Seed">Seed-OSS</a></td>
      <td><a href="https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Base/tree/main">36B</a></td>
      <td><a href="../../../../examples/mcore/seed_oss">seed_oss</a></td>
      <td> 2K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/mistralai">Magistral</a></td>
      <td><a href="https://huggingface.co/mistralai/Magistral-Small-2506/tree/main">24B</a></td>
      <td><a href="../../../../examples/mcore/magistral">magistral</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/PLM-Team">PLM</a></td>
      <td><a href="https://huggingface.co/PLM-Team/PLM-1.8B-Base/tree/main">1.8B</a></td>
      <td><a href="../../../../examples/mcore/plm">plm</a></td>
      <td> 2K </td>
      <th> MCore </th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
  </tbody>
</table>

## Sparse Models

Sparse models use a sparse neuron connectivity pattern. Therefore, only a small number of neurons connect to one another. A typical sparse model is the mixture-of-experts (MoE) model, which contains multiple expert networks and activates only some experts during each training step. This design can significantly reduce the parameter count and computational complexity and improve training efficiency. It is especially suitable for large-scale datasets and complex tasks. However, sparse model training also has drawbacks. It can easily produce unbalanced expert loads and unstable training.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Download Link</th>
      <th>Script Path</th>
      <th>Sequence Length</th>
      <th>Training Backend</th>
      <th>Cluster Size</th>
      <th>Supported Version</th>
      <th>Contributor</th>
      <th>Certification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Base">30B-A3B</a></td>
      <td><a href="../../../../examples/mcore/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="../../../../examples/fsdp2/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 1x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B">235B-A22B</a></td>
      <td><a href="../../../../examples/mcore/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 16x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="../../../../examples/fsdp2/qwen3_moe">qwen3_moe</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 16x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/collections/Qwen/qwen3-next">Qwen3-Next</a></td>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct">80B-A3B</a></td>
      <td><a href="../../../../examples/mcore/qwen3_next">qwen3_next</a></td>
      <td> 16K </td>
      <th> MCore </th>
      <td> 4x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td><a href="../../../../examples/fsdp2/qwen3_next">qwen3_next</a></td>
      <td> 16K </td>
      <th> FSDP2 </th>
      <td> 4x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main">Qwen3-Coder-Next</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-Coder-Next/tree/main">80B-A3B</a></td>
      <td><a href="../../../../examples/mcore/qwen3_coder_next">qwen3_coder_next</a></td>
      <td> 16K </td>
      <th>MCore</th>
      <td>4x16</td>
      <td rowspan="1"> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-57B-A14B/tree/main">57B-A14B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/qwen2_moe">qwen2_moe</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>8x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/xai-org/grok-1/tree/main">Grok-1</a></td>
      <td><a href="https://huggingface.co/xai-org/grok-1/tree/main">40B</a></td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0/examples/mcore/grok1">grok-1</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td>4x8</td>
      <td><a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.0.0">2.0.0</a></td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/mistralai">Mixtral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">8x7B</a></td>
      <td rowspan="3"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/mixtral">mixtral</a></td>
      <td> 32K</td>
      <th>MCore</th>
      <td>8x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/tree/main">8x22B</a></td>
      <td> 32K</td>
      <th>MCore</th>
      <td>8x8</td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td> 64K</td>
      <th>MCore</th>
      <td>8x8</td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">DeepSeek-V2</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek2">deepseek2</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
        <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base">DeepSeek-V2-coder</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek2_coder">deepseek2_coder</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite">DeepSeek-V2-Lite</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main">16B</a></td>
      <td><a href="../../../../examples/mcore/deepseek2_lite">deepseek2_lite</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5">DeepSeek-V2.5</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5/tree/main">236B</a></td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/deepseek25">deepseek25</a></td>
      <td> 8K </td>
      <th>MCore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">DeepSeek-V3</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main">671B</a></td>
      <td><a href="../../../../examples/mcore/deepseek3">deepseek3</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 64x8 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
      <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp">DeepSeek-V3.2</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main">671B</a></td>
      <td><a href="../../../../examples/mcore/deepseek32">deepseek3.2</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 32x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base">DeepSeek-V4-Flash</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/tree/main">284B</a></td>
      <td><a href="../../../../examples/mcore/deepseek4_flash">deepseekv4-flash</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 8x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-MoE-8x2B/tree/main">8x2B</a> </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0/examples/mcore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0">2.2.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/inclusionAI/Ling-mini-2.0">Ling-mini-2.0</a></td>
      <td> <a href="https://huggingface.co/inclusionAI/Ling-mini-2.0/tree/main">16B</a> </td>
      <td rowspan="2"><a href="../../../../examples/mcore/ling_v2">ling_v2</a></td>
      <td> 4K </td>
      <th>MCore</th>
      <td> 1x8 </td>
      <td rowspan="1"></td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/inclusionAI/Ring-1T">Ring</a></td>
      <td> <a href="https://huggingface.co/inclusionAI/Ring-1T/tree/main">1T</a> </td>
      <td> 32K </td>
      <th>MCore</th>
      <td> 32x8 </td>
      <td rowspan="1"></td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/microsoft">Phi3.5</a></td>
      <td> <a href="https://huggingface.co/microsoft/Phi-3.5-MoE-instruct">MoE-instruct</a> </td>
      <td><a href="../../../../examples/mcore/phi35">phi35</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 2x8 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/tencent">Hunyuan</a></td>
      <td> <a href="https://huggingface.co/tencent/Tencent-Hunyuan-Large">389B</a> </td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/hunyuanLarge">hunyuanLarge</a></td>
      <td> 8K </td>
      <th> MCore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td>GPT4</td>
      <td>MoE-175B</td>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0/examples/mcore/gpt4">gpt4</a></td>
      <td> 128K </td>
      <th> MCore </th>
      <td> 8x8 </td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.3.0">2.3.0</a> </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/zai-org">GLM4.5-Air</a></td>
      <td> <a href="https://huggingface.co/zai-org/GLM-4.5-Air/tree/main">MoE-106B</a> </td>
      <td><a href="../../../../examples/mcore/glm45-air">glm45-air</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 8x8 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/zai-org">GLM5</a></td>
      <td><a href="https://huggingface.co/zai-org/GLM-5">MoE-744B</a></td>
      <td><a href="../../../../examples/mcore/glm5">glm5</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 32x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Pass]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/zai-org">GLM5.2</a></td>
      <td><a href="https://huggingface.co/zai-org/GLM-5.2">MoE-744B</a></td>
      <td><a href="../../../../examples/mcore/glm52">glm52</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 32x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/stepfun-ai">Step3.5-Flash</a></td>
      <td><a href="https://huggingface.co/stepfun-ai/Step-3.5-Flash">MoE-196B</a></td>
      <td><a href="../../../../examples/fsdp2/step35">step35</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 12x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/meituan-longcat">LongCat</a></td>
      <td><a href="https://huggingface.co/meituan-longcat/LongCat-Flash-Chat">MoE-560B</a></td>
      <td><a href="../../../../examples/mcore/longcat">longcat</a></td>
      <td> 4K </td>
      <th> MCore </th>
      <td> 8x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/openai">gpt-oss</a></td>
      <td><a href="https://modelscope.cn/models/unsloth/gpt-oss-20b-BF16/">MoE-20B</a></td>
      <td><a href="../../../../examples/fsdp2/gpt_oss">gpt_oss</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 1x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/MiniMaxAI">MiniMax-M2.7</a></td>
      <td><a href="https://huggingface.co/MiniMaxAI/MiniMax-M2.7">MoE-229B</a></td>
      <td><a href="../../../../examples/fsdp2/minimax_m27">minimax_m27</a></td>
      <td> 4K </td>
      <th> FSDP2 </th>
      <td> 8x16 </td>
      <td>  </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> GPT model vocabulary files differ from those of standard models. Configure them as follows:
>
> ```shell
> mkdir vocab_file
> cd vocab_file
> wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
> wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
> cd ..
>
> # Process into training data
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
> # Configure the following pretraining script parameters based on the actual file paths
> VOCAB_FILE="./vocab_file/gpt2-vocab.json"   # Vocabulary
> MERGE_FILE="./vocab_file/gpt2-merges.txt"   # BPE merges
> DATA_PATH="./dataset/gpt_text_sentence"     # Data path
> ```

## State Space Models

SSMs are sequence models based on state-space representations and can efficiently model long sequence data. Compared with traditional Transformer models, SSMs offer better compute efficiency and memory efficiency when processing long sequences.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Download Link</th>
      <th>Script Path</th>
      <th>Sequence Length</th>
      <th>Training Backend</th>
      <th>Cluster Size</th>
      <th>Supported Version</th>
      <th>Contributor</th>
      <th>Certification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Mamba2</td>
      <td><a href="https://huggingface.co/state-spaces/mamba2-2.7b/tree/main">2.7B</a></td>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0/examples/mcore/mamba2">mamba2</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td> 1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/26.0.0">26.0.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/main">8B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td> 1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/26.0.0">26.0.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1">Mamba2Hybrid</td>
      <td><a href="https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-4k/tree/main">8B</a></td>
       <td><a href="https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0/examples/mcore/mamba2">mamba2</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/26.0.0">26.0.0</a> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
    <tr>
      <td rowspan="1">Mamba3</td>
      <td>/</td>
       <td><a href="../../../../examples/fsdp2/mamba3">mamba3</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td> </td>
      <td>[Ascend]</td>
      <td>[Test]</td>
    </tr>
  </tbody>
</table>

> [!NOTE]
>
> The open-source Mamba2 series does not provide vocabulary files. The vocabulary file used in internal tests, `mamba2_2.7b_from_8b.model`, is a custom design. You are advised to build your own vocabulary, and training results are not guaranteed.
>
> The Mamba3 model is not open source yet. The repository only provides a runnable demo example. To run the demo, make the following changes:
>
>1. Configuration file: Reuse `mamba2-2.7b` `config.json` and add the following two settings:
>
>    ```json
>     "model_type": "mamba2",
>     "is_mimo": false
>     ```
>
>     Set `is_mimo` to `false` or `true` as needed.
>2. Vocabulary: You are advised to build your own vocabulary. You can also use the vocabulary from other open-source models, such as Qwen3-Next, but training results are not guaranteed.

## Multimodal Models

Multimodal models, including image-text understanding, text-to-video/image generation, and speech recognition, are maintained separately by MindSpeed-MM. If you need to train a multimodal model, visit the multimodal repository [MindSpeed MM](https://gitcode.com/Ascend/MindSpeed-MM) for detailed instructions. For the current supported models by MindSpeed-MM, refer to the [MindSpeed MM model support list](https://gitcode.com/Ascend/MindSpeed-MM#%E9%85%8D%E5%A5%97%E7%89%88%E6%9C%AC%E4%B8%8E%E6%94%AF%E6%8C%81%E6%A8%A1%E5%9E%8B).

## MindSpeed-LLM MindSpore后端稀疏模型支持

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>脚本位置</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
      <th>是否支持</th>
    </tr>
  </thead>
  <tbody>
    </tr>
       <tr>
       <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-57B-A14B/tree/main">57B-A14B</a></td>
      <td><a href="../../../examples/mindspore/">qwen2_moe</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td>支持中</td>
      <tr>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">DeepSeek-V2</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2/tree/main">236B</a></td>
      <td><a href="../../../examples/mindspore/">deepseek2</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite">DeepSeek-V2-Lite</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main">16B</a></td>
      <td><a href="../../../examples/mindspore/deepseek2_lite">deepseek2_lite</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td>✅</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5">DeepSeek-V2.5</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2.5/tree/main">236B</a></td>
      <td><a href="../../../examples/mindspore/">deepseek25</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">DeepSeek-V3</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main">671B</a></td>
      <td><a href="../../../examples/mindspore/deepseek3">deepseek3</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 64x8 </td>
      <td>✅</td>
    </tr>
  </tbody>
</table>
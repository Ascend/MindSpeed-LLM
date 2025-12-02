## MindSpeed-LLM 预置MoE大模型

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
      <td rowspan="2"> <a href="https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f">Qwen3</a> </td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-30B-A3B-Base">30B</a></td>
      <td rowspan="2"><a href="../../../examples/mcore/qwen3_moe/">qwen3_moe</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 2x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-235B-A22B">235B</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 16x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
       <tr>
       <td rowspan="1"><a href="https://huggingface.co/collections/Qwen/qwen3-next">Qwen3-next</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct">80B-A3B</a></td>
      <td><a href="../../../examples/mcore/qwen2_moe">Qwen3-Next-80B-A3B</a></td>
      <td> 16K </td>
      <th>Mcore</th>
      <td>4x16</td>
      <td rowspan="1"> <a href="../../../examples/mcore/qwen3_next/"></a> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
      <tr>
    </tr>
    <tr>
       <tr>
       <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-57B-A14B/tree/main">57B-A14B</a></td>
      <td><a href="../../../examples/mcore/qwen2_moe">qwen2_moe</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【GTS】</td>
      <td>【Pass】</td>
      <tr>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/mistralai">Mixtral</a></td>
      <td><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">8x7B</a></td>
      <td rowspan="3"><a href="../../../examples/mcore/mixtral">mixtral</a></td>
      <td> 32K</td>
      <th>Mcore</th>
      <td>8x8</td>
      <td rowspan="3"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
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
      <td><a href="../../../examples/mcore/deepseek2">deepseek2</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
        <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base">DeepSeek-V2-coder</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base/tree/main">236B</a></td>
      <td><a href="../../../examples/mcore/deepseek2_coder">deepseek2_coder</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite">DeepSeek-V2-Lite</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main">16B</a></td>
      <td><a href="../../../examples/mcore/deepseek2_lite">deepseek2_lite</a></td>
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
      <td><a href="../../../examples/mcore/deepseek25">deepseek25</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【NAIE】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3">DeepSeek-V3</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main">671B</a></td>
      <td><a href="../../../examples/mcore/deepseek3">deepseek3</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 64x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-MoE-8x2B/tree/main">8x2B</a> </td>
      <td><a href="../../../examples/mcore/minicpm">minicpm</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td rowspan="1"> <a href="https://gitcode.com/ascend/MindSpeed-LLM/tree/2.2.0/">2.2.0</a> </td>
      <td>【NAIE】</td>
      <td>【Test】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/inclusionAI/Ling-mini-base-2.0">Ling-mini-2.0</a></td>
      <td> <a href="https://huggingface.co/inclusionAI/Ling-mini-base-2.0/tree/main">16B</a> </td>
      <td rowspan="2"><a href="../../../examples/mcore/ling_v2">ling_v2</a></td>
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
      <td><a href="../../../examples/mcore/phi35">phi35</a></td>
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
      <td><a href="../../../examples/mcore/hunyuanLarge">hunyuanLarge</a></td>
      <td> 8K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td>GPT4</td>
      <td>MoE-175B</td>
      <td rowspan="1"><a href="../../../examples/mcore/gpt4">gpt4</a></td>
      <td> 128K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Pass】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/zai-org">GLM4.5</a></td>
      <td> <a href="https://huggingface.co/zai-org/GLM-4.5-Air/tree/main">MoE-106B</a> </td>
      <td><a href="../../../examples/mcore/glm45-moe">glm45-moe</a></td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 8x8 </td>
      <td>  </td>
      <td>【Ascend】</td>
      <td>【Test】</td>
    </tr>
  </tbody>
</table>

## 说明

### DeepSeek3模型

注意：当前master分支未适配DeepSeek-V3，如需使用请切换到2.1.0商发分支
    
版本要求：CANN版本≥8.1.RC1，PTA版本≥7.0.RC1。

MTP说明：master分支是参考Megatron-LM实现，与2.0.0分支实现方案不同，训练loss表现不一致，使能方式如下：

  ```shell
  # MTP层数
  --mtp-num-layers
  # MTP loss系数
  --mtp-loss-scaling-factor
  ```

### 社区BUG列表

1. DeepSeek2：使用examples/mcore/deepseek2/pretrain_deepseek2_100b_8k_C_ptd.sh进行八机预训练任务时，需确保首节点有1.2T的host内存，第二节点有1.1T的host内存，以此类推。可通过以下命令进行查询

    ```shell
    # 查询host内存，通过free字段确定当前可用host内存
    free -h
    ```

    

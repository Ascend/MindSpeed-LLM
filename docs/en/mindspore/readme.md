# MindSpore Backend

## Support Overview

MindSpeed LLM now supports integration with Huawei's in-house AI framework MindSpore. It aims to provide an easy-to-use end-to-end large language model training solution across Huawei's full stack and deliver a more extreme performance experience. The MindSpore backend provides a set of APIs that align with PyTorch. Therefore, you can switch seamlessly without additional code adaptation.

---

## News

🚀🚀🚀The MindSpore backend now supports **[DeepSeek-V3](../../../examples/mcore/deepseek3/README_en.md)/[Qwen3](../../../examples/mindspore/qwen3/README_en.md)/[GLM-4.5](../../../examples/mindspore/glm45-moe/README_en.md)**! 🚀🚀🚀

## Version Compatibility Table

The dependency compatibility for MindSpeed LLM plus the MindSpore backend is listed below. For installation steps, see [MindSpeed LLM Installation Guide](../mindspore/install_guide.md).

<table>
  <tr>
    <th>Dependency</th>
    <th>Version</th>
  </tr>
  <tr>
    <td>Ascend NPU driver</td>
    <td rowspan="2">In development</td>
  </tr>
  <tr>
    <td>Ascend NPU firmware</td>
  </tr>
  <tr>
    <td>Toolkit</td>
    <td rowspan="3">CANN 8.5.0</td>
  </tr>
  <tr>
    <td>Kernel</td>
  </tr>
  <tr>
    <td>Ascend Transformer Boost acceleration library (NNAL)</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>MindSpore</td>
    <td>2.8.0</td>
  </tr>
</table>
Note: The master branch uses development versions of the driver and CANN package. Therefore, some new features on master may not be supported by older dependency versions. To use a stable version, switch to the commercial release branch and install the corresponding dependency versions.

## Model Support

The MindSpore backend only supports models implemented in `mcore`. The current model support details are listed below, and support for more models will be added gradually.

<table><thead>
  <tr>
    <th>Model Category</th>
    <th>Model List</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="1">Supported Models</td>
    <td><a href="models/supported_models.md">supported_models</a></td>
  </tr>
</tbody></table>

## Feature Support

The following table shows support for the key acceleration features of MindSpeed on the MindSpore backend. Some unsupported features will be added in later iterations. Stay tuned.

<table><thead>
  <tr>
    <th>Scenario</th>
    <th>Feature</th>
    <th>Support</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="6">SPTD parallelism</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/tensor-parallel.md">Tensor parallelism</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/pipeline-parallel.md">Pipeline parallelism</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../pytorch/features/mcore/virtual_pipeline_parallel.md">Virtual pipeline parallelism</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/sequence-parallel.md">Sequence parallelism</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/noop-layers.md">Noop Layers</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/custom_fsdp.md">Fully sharded parallelism</a></td>
    <td>Does not currently support enabling `pp` or the `--reuse-fp32-param` parameter.</td>
  </tr>
  <tr>
    <td rowspan="2">Long-sequence parallelism</td>
    <td><a href="../pytorch/features/mcore/ring-attention-context-parallel.md">Ascend Ring Attention long-sequence parallelism</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/ulysses-context-parallel.md">Ulysses long-sequence parallelism</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="2">MOE</td>
    <td><a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md">MOE expert parallelism</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/megatron_moe/megatron-moe-allgather-dispatcher.md">MOE dispatch communication optimization</a></td>
    <td>Only `alltoall` is supported.</td>
  </tr>
  <tr>
    <td rowspan="5">Memory optimization</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/reuse-fp32-param.md">Parameter copy reuse</a></td>
    <td>Must be used together with the distributed optimizer feature.</td>
  </tr>
    <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/distributed-optimizer.md">Distributed optimizer</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../pytorch/features/mcore/recompute_relative.md">Recomputation</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/norm-recompute.md">Norm recomputation</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/virtual-optimizer.md">Virtual Optimizer</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="7">Fused operators</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/flash-attention.md">Flash attention</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../pytorch/features/mcore/variable_length_flash_attention.md">Variable-length flash attention</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/rms_norm.md">Fused RMSNorm</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/swiglu.md">Fused SwiGLU</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/rotary-embedding.md">Fused rotary position embedding</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/megatron_moe/megatron-moe-gmm.md">GMM</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/npu_matmul_add.md">Matmul Add</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td rowspan="3">Communication optimization</td>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/async-ddp-param-gather.md">Gradient reduce communication overlap</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/async-ddp-param-gather.md">Weight all-gather communication overlap</a></td>
    <td>✅</td>
  </tr>
  <tr>
    <td><a href="../pytorch/features/mcore/communication-over-computation.md">CoC</a></td>
    <td>✅</td>
  </tr>
</tbody></table>

### Online Inference

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Supported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="../pytorch/training/inference/inference.md">Streaming inference</a></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="../pytorch/training/inference/chat.md">Chat conversation</a></td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="../pytorch/features/mcore/yarn.md">YARN context extension</a></td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

### Open Dataset Evaluation

Coming soon. Stay tuned.

## Developer Toolchain

### Data Preprocessing

The MindSpore backend fully supports data preprocessing for MindSpeed LLM pretraining, supervised fine-tuning, RLHF, and other tasks.

<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Dataset</th>
      <th>MCore</th>
      <th>Released</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Pretraining</td>
      <td><a href="../pytorch/tools/data_process_pretrain.md">Pretraining data processing</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">Fine-tuning</td>
      <td><a href="../pytorch/tools/data_process_sft_alpaca_style.md">Alpaca style</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="../pytorch/tools/data_process_sft_sharegpt_style.md">ShareGPT style</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>DPO</td>
      <td><a href="../pytorch/tools/data_process_dpo_pairwise.md">Pairwise data processing</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【NAIE】</td>
    </tr>
  </tbody>
</table>

### Weight Conversion

The weight conversion for the MindSpeed MindSpore backend is consistent with the PyTorch backend. It currently supports mutual conversion between Hugging Face and Megatron-Core weight formats. For the weight conversion parameters and usage, see [Weight Conversion](../pytorch/tools/checkpoint_convert_hf_mcore.md).

<table>
  <thead>
    <tr>
      <th>Source Format</th>
      <th>Target Format</th>
      <th>Sharding Features</th>
      <th>LoRA</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>huggingface</td>
      <td>megatron-core</td>
      <td>tp, pp, dpp, vpp, cp, ep, loop layer</td>
      <td>❌</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">megatron-core</td>
      <td>huggingface</td>
      <td></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>megatron-core</td>
      <td>tp, pp, dpp, vpp, cp, ep, loop layer</td>
      <td>❌</td>
    </tr>
  </tbody>
</table>

### Performance Profiling

<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Feature</th>
      <th>MCore</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">Performance profiling</td>
      <td><a href="../pytorch/tools/profiling.md">Collect profiling data based on Ascend chips</a></td>
      <td>✅</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>

### High Availability

<table>
  <thead>
    <tr>
      <th>Scenario</th>
      <th>Feature</th>
      <th>MCore</th>
      <th>Contributor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">High availability</td>
      <td><a href="../pytorch/tools/deterministic_computation.md">Enable deterministic computation based on Ascend chips</a></td>
      <td>✅</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
  </tbody>
</table>

# MindSpore Backend Support for GLM4.5 Series Models

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Download link</th>
      <th>Sequence Length</th>
      <th>Implementation</th>
      <th>Cluster</th>
      <th>Supported</th>
    </tr>
  </thead>
  <tbody>
      <tr>
      <td rowspan="7"><a href="https://huggingface.co/zai-org/GLM-4.5">GLM4.5</a></td>
      <td rowspan="2"><a href="https://huggingface.co/zai-org/GLM-4.5/tree/main">106B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>8x16</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## Step-by-Step Guide to Running the GLM-4.5 Model on the MindSpore Backend

### Environment Configuration

For the installation steps for the MindSpore backend of MindSpeed LLM, see [MindSpeed LLM Installation Guide](../../../docs/en/mindspore/install_guide.md).

### Training

#### Pretraining

Use the following commands for pretraining:

```sh
cd MindSpeed-LLM
bash examples/mindspore/glm45-moe/pretrain_glm45_moe_106b_4k_A3_ms.sh
```

Modify the following variables in the script based on your actual environment:

  | Variable name | Meaning |
  |--------|-----------------------------------|
  | MASTER_ADDR | IP address of the primary node in a multi-node scenario. |
  | NODE_RANK | Node sequence number of each node in a multi-node scenario. |
  | CKPT_SAVE_DIR | Path for saving weights during training. |
  | DATA_PATH | Data path after data preprocessing. |
  | TOKENIZER_PATH | GLM-4.5 tokenizer directory. |
  | CKPT_LOAD_DIR | Weight path saved during weight conversion. This path is used to load the initial weights. If no initial weights exist, random initialization is used. |

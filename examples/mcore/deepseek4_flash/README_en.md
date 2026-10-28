# DeepSeek-V4 Model Training

On April 24, 2026, the DeepSeek-V4 series models DeepSeek-V4-Flash and DeepSeek-V4-Pro were officially released and open-sourced. With an advanced MoE architecture, they achieve significant breakthroughs in parameter efficiency and inference performance. MindSpeed LLM has implemented pretraining support for the **DeepSeek-V4-Flash model** in fixed-length data scenarios and has also open-sourced the code. This document provides practical references for pretraining to help you get started quickly.

**Note: The current implementation is a preview version with limitations in some scenarios. We will continue to improve it and follow the evolution directions described in the DeepSeek-V4 technical report:**

<table style="text-align: center">
  <thead>
    <tr>
      <th style="text-align: center;">Scenario</th>
      <th style="text-align: center;">Feature</th>
      <th style="text-align: center;">Support Status</th>
      <th style="text-align: center;">Remarks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Training scenario</td>
      <td>Pretraining/Continued training</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>Full-parameter fine-tuning</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td>LoRA fine-tuning</td>
      <td>DOING</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan="3">Data scenario</td>
      <td>Fixed-length data</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>Packed data</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td>Variable-length data</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td rowspan="4">Parallelism strategy</td>
      <td>TP</td>
      <td>OK</td>
      <td>Supports TP=2</td>
    </tr>
    <tr>
      <td>PP</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>EP</td>
      <td>OK</td>
    </tr>
    <tr>
      <td>CP</td>
      <td>DOING</td>
    </tr>
    <tr>
      <td rowspan="1">Features</td>
      <td>Muon optimizer</td>
      <td>DOING</td>
    </tr>
  </tbody>
</table>

## Installation Guide

Refer to [MindSpeed LLM software installation](../../../docs/en/pytorch/training/install_guide.md) to configure the environment and pull the repository code.

## Weight Conversion

1. Download the weights

    Download the weights and configuration file from [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base).

2. Convert the weights

    The open-source DeepSeek-V4-Flash weights use the FP8 mixed data format. Before training, you must dequantize the original weights to obtain weights in the BF16 format. For the dequantization method, refer to the following script.

    ```sh
    cd MindSpeed-LLM
    bash examples/mcore/deepseek4_flash/ckpt_dequant_deepseek4_fp8_to_bf16.sh

    ```

    MindSpeed LLM provides a [script](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) to convert open-source Hugging Face weights into MCore weights for tasks such as training, inference, and evaluation.
    The usage is as follows. Modify the weight conversion script according to the actual parallelism strategy you need, such as TP/PP, and the weight path.

    ```sh
    cd MindSpeed-LLM
    bash examples/mcore/deepseek4_flash/ckpt_convert_deepseek4_flash_hf2mcore.sh # During conversion, specify the path to the dequantized weights
    ```

## Data Preprocessing

MindSpeed LLM provides a [script](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) to process the dataset.

The usage is as follows. Modify the following parameters as needed.

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/data_convert_deepseek4_pretrain.sh
```

| Parameter  | Description                |
|--------|-----------------|
| --input | Dataset path  |
| --tokenizer-name-or-path | Model tokenizer directory    |
| --output-prefix | Output path and prefix name of the processed dataset  |

## Pretraining

MindSpeed LLM provides a [script](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) for model training.

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/pretrain_deepseek4_flash_4k_A3_ptd.sh
```

Modify the following variables in the script as needed.

  | Variable  | Description                |
  |--------|-----------------|
  | MASTER_ADDR | IP address of the primary node in a multi-node setup  |
  | NODE_RANK | Node index of each machine in a multi-node setup    |
  | CKPT_SAVE_DIR | Path for saving weights during training  |
  | DATA_PATH | Path to the preprocessed data  |
  | TOKENIZER_PATH | Tokenizer directory  |
  | CKPT_LOAD_DIR | Path to the weights saved by weight conversion. These weights are used as the initially loaded weights. If no initial weights exist, the weights are initialized randomly  |

## Full-Parameter Fine-Tuning

MindSpeed LLM provides a [script](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) for model training.

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/tune_deepseek4_flash_4k_A3_ptd.sh
```

Modify the following variables in the script as needed.

  | Variable  | Description                |
  |--------|-----------------|
  | MASTER_ADDR | IP address of the primary node in a multi-node setup  |
  | NODE_RANK | Node index of each machine in a multi-node setup    |
  | CKPT_SAVE_DIR | Path for saving weights during training  |
  | DATA_PATH | Path to the preprocessed data  |
  | TOKENIZER_PATH | Tokenizer directory  |
  | CKPT_LOAD_DIR | Path to the weights saved by weight conversion, which are used as the initially loaded weights  |

## LoRA Fine-Tuning

MindSpeed LLM provides a [script](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/examples/mcore/deepseek4_flash) for model training.

```sh
cd MindSpeed-LLM
bash examples/mcore/deepseek4_flash/lora_finetune_deepseek4_flash_4k_A3_ptd.sh
```

Modify the following variables in the script as needed.

  | Variable  | Description                |
  |--------|-----------------|
  | MASTER_ADDR | IP address of the primary node in a multi-node setup  |
  | NODE_RANK | Node index of each machine in a multi-node setup    |
  | CKPT_SAVE_DIR | Path for saving weights during training  |
  | DATA_PATH | Path to the preprocessed data  |
  | TOKENIZER_PATH | Tokenizer directory  |
  | CKPT_LOAD_DIR | Path to the weights saved by weight conversion, which are used as the initially loaded weights  |

Note: LoRA fine-tuning uses an instruction dataset. First run `examples/mcore/deepseek4_flash/data_convert_deepseek4_instruction.sh` to complete data preprocessing, and then enter the output prefix in `DATA_PATH`.

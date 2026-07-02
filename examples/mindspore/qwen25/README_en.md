# MindSpore Backend Support for Qwen25 Series Models

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Download link</th>
      <th>Sequence length</th>
      <th>Implementation</th>
      <th>Cluster</th>
      <th>Supported</th>
    </tr>
  </thead>
  <tbody>
      <tr>
      <td rowspan="7"><a href="https://huggingface.co/Qwen">Qwen25</a></td>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-7B/tree/main">7B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td>✅</td>
      </tr>
      <td>32K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td>✅</td>
      <tr>
      <td rowspan="2"><a href="https://huggingface.co/Qwen/Qwen2.5-32B/tree/main">32B</a></td>
      <td>4K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td>✅</td>
      </tr>
      <td>32K</td>
      <th>MCore</th>
      <td>4x8</td>
      <td>✅</td>
      <tr>
    </tr>
  </tbody>
</table>

## Step-by-Step Guide to Running Qwen25 Models on the MindSpore Backend

### Environment Configuration

For the installation procedure of the MindSpore backend of MindSpeed LLM, see [MindSpeed LLM Installation Guide](../../../docs/en/mindspore/install_guide.md).

### Weight Conversion

1. Download weights

   Use the 7B model as an example. Download the model weights and other configuration files from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B/tree/main). To continue pretraining, fine-tune, or run inference based on the open-source weights, also download the network model files.

2. Convert weights

   The script converts Hugging Face open-source weights to MCore weights for tasks such as training, inference, and evaluation. Use the script as follows. Modify the weight conversion script based on the required tensor parallelism (TP), pipeline parallelism (PP), and weight paths.

    ```sh
    cd MindSpeed-LLM
    bash examples/mindspore/qwen25/ckpt_convert_qwen25_hf2mcore.sh
    ```

   After you run the script, you expect to see log output similar to the following, which indicates that weight conversion succeeds.

    ```log
   successfully saved checkpoint from iteration 1 to ./model_weights/qwen2.5_mcore/
   INFO:root:Done!
    ```

**Note:**

- By default, the MindSpore backend converts weights on the device side. Large models may cause out-of-memory (OOM) exceptions. Therefore, you are advised to modify `convert_ckpt.py` and add the following code during package import to convert weights on the CPU side:

```python
import mindspore as ms
ms.set_context(device_target="CPU", pynative_synchronize=True)
import torch
torch.configs.set_pyboost(False)
```

- Model weights converted by the MindSpore backend cannot be used for training or inference on the Torch backend.

### Data Preprocessing

The MindSpore backend currently fully supports data preprocessing for multiple MindSpeed LLM task scenarios.

#### Pretraining

Use the Alpaca dataset as an example. For [data preprocessing](../../../docs/en/pytorch/tools/data_process_pretrain.md), configure only the data input and output paths and the tokenizer model path in the pretraining data preprocessing script `data_convert_qwen25_pretrain.sh`, and then start the script.

```sh
bash examples/mindspore/qwen25/data_convert_qwen25_pretrain.sh
```

The pretraining dataset processing results are as follows:

```log
./dataset/alpaca_text_document.bin
./dataset/alpaca_text_document.idx
```

During pretraining, pass `./dataset/alpaca_text_document` to the dataset path parameter `--data-path`.

#### Fine-Tuning

Use [Alpaca-style fine-tuning dataset processing](../../../docs/en/pytorch/tools/data_process_sft_alpaca_style.md) as an example. Configure only the data input and output paths and the tokenizer model path in the fine-tuning data preprocessing script `data_convert_qwen25_instruction.sh`, and then start the script.

```sh
bash examples/mindspore/qwen25/data_convert_qwen25_instruction.sh
```

The fine-tuning dataset processing results are as follows:

```log
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```

During fine-tuning, enter `./finetune_dataset/alpaca` as the dataset path.

### Training

#### Pretraining

Use the following command for pretraining:

```sh
# Use the 7B model as an example.
cd MindSpeed-LLM
bash examples/mindspore/qwen25/pretrain_qwen25_7b_32k_ms.sh
```

Modify the following variables in the script based on your actual environment:

  | Variable  | Description                                |
  |--------|-----------------------------------|
  | MASTER_ADDR | IP address of the master node in multi-node scenarios.                        |
  | NODE_RANK | Rank of each node in multi-node scenarios.                      |
  | CKPT_SAVE_DIR | Path for saving weights during training.                         |
  | DATA_PATH | Data path after data preprocessing.                       |
  | TOKENIZER_PATH | Qwen25 tokenizer directory.                |
  | CKPT_LOAD_DIR | Weight path saved after weight conversion for initial weight loading. If no initial weights are provided, the model is randomly initialized. |

#### Fine-Tuning

Fine-tuning uses a method similar to pretraining.

```sh
# Use full-parameter fine-tuning of the 7B model as an example.
cd MindSpeed-LLM
bash examples/mindspore/qwen25/tune_qwen25_7b_4k_full_ms.sh
```

As with pretraining, modify the preceding variables in the script based on your environment.

### Inference

Use the following command for inference:

```sh
# Use the 7B model as an example.
cd MindSpeed-LLM
bash examples/mindspore/qwen25/generate_qwen25_7b_ms.sh
```

Modify the following variables in the script based on your actual environment:

  | Variable  | Description                 |
  |--------|--------------------|
  | MASTER_ADDR | IP address of the master node in multi-node scenarios.         |
  | NODE_RANK | Rank of each node in multi-node scenarios.       |
  | CHECKPOINT | Path to the weights saved during training.          |
  | TOKENIZER_PATH | Qwen25 tokenizer directory. |

### Evaluation

Use the following command for evaluation:

```sh
# Use the 7B model as an example.
cd MindSpeed-LLM
bash examples/mindspore/qwen25/evaluate_qwen25_7b_ms.sh
```

Modify the following variables in the script based on your environment. For datasets, see [evaluation datasets](../../../docs/en/pytorch/training/evaluation/evaluation_datasets/mmlu_evaluation.md).

  | Variable  | Description                    |
  |--------|-----------------------|
  | MASTER_ADDR | IP address of the master node in multi-node scenarios.            |
  | NODE_RANK | Rank of each node in multi-node scenarios.          |
  | TOKENIZER_PATH | Qwen25 tokenizer directory.    |
  | CKPT_LOAD_DIR | Weight path saved after weight conversion, or weight path saved during training.         |
  | DATA_PATH | Dataset path used for evaluation. MMLU is currently recommended. |
  | TASK | Dataset used for evaluation. MMLU is currently recommended. |

The evaluation results on the first three MMLU subsets are as follows:

```log
INFO:mindspeed_llm.tasks.evaluation.eval_impl.mmlu_eval:mmlu acc = 321/387=0.8294573643410853
total: 100%|█████████████████████████████████████████████████████| 3/3 [06:16<00:00, 128.12s/it]INFO:main:
             subject   question_n   acc
0   abstract_algebra          100   0.720000
1          astronomy          152   0.927632
2            anatomy          135   0.800000
3              total          387   0.829457
INFO:main:MMLU Running Time:, 376.0990614891052
```

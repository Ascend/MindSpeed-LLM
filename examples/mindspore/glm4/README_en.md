# MindSpore Backend Support for GLM-4-9B Series Models

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
      <td rowspan="7"><a href="https://huggingface.co/THUDM/glm-4-9b">GLM-4-9B</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/glm-4-9b/tree/main">9B</a></td>
      <td> 4K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td>✅</td>
      </tr>
      <tr>
      <td> 32K</td>
      <th>MCore</th>
      <td>1x8</td>
      <td>✅</td>
      </tr>
  </tbody>
</table>

## Step-by-Step Guide to Running the GLM-4-9B Model on the MindSpore Backend

### Environment Configuration

For the installation steps for the MindSpore backend of MindSpeed LLM, see the [MindSpeed LLM installation guide](../../../docs/en/mindspore/install_guide.md).

### Weight Conversion

1. Download weights.

   Download the model weights and other configuration files from [Hugging Face](https://huggingface.co/THUDM/glm-4-9b/tree/main). To continue pretraining, fine-tuning, or inference with the open-source weights, also download the network model file.

2. Convert weights.

   The script converts Hugging Face open-source weights to MCore weights for tasks such as training, inference, and evaluation. Use it as follows. Modify the weight conversion script based on the required tensor parallelism (TP), pipeline parallelism (PP), partitioning strategies, and weight paths.

    ```sh
    cd MindSpeed-LLM
    bash examples/mindspore/glm4/ckpt_convert_glm4_hf2mcore.sh
    ```

   After you run the script, you expect to see log output similar to the following, which indicates that weight conversion succeeds.

    ```log
   successfully saved checkpoint from iteration 1 to ./model_weights/glm4_mcore/
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

The MindSpore backend currently fully supports data preprocessing for multiple MindSpeed LLM task scenarios. For the data preprocessing guide, see [data preprocessing](../../../docs/en/pytorch/tools/data_process_pretrain.md).

In a pretraining scenario, for example, you only need to configure the data input and output paths and the tokenizer model path in the pretraining data preprocessing script `data_convert_glm4_pretrain.sh`, and then start the script:

```sh
bash examples/mindspore/glm4/data_convert_glm4_pretrain.sh
```

The pretraining dataset processing result is as follows:

```log
./dataset/alpaca_text_document.bin
./dataset/alpaca_text_document.idx
```

### Training

#### Pretraining

Use the following command for pretraining:

```sh
cd MindSpeed-LLM
bash examples/mindspore/glm4/pretrain_glm4_8k_ms.sh
```

Modify the following variables in the script based on your actual environment:

  | Variable  | Description                                |
  |--------|-----------------------------------|
  | MASTER_ADDR | IP address of the master node in a multi-node scenario.                        |
  | NODE_RANK | Sequence number of the corresponding node in a multi-node scenario.                      |
  | CKPT_SAVE_DIR | Path to save weights during training.                         |
  | DATA_PATH | Data path after data preprocessing.                       |
  | TOKENIZER_PATH | GLM-4-9B tokenizer directory.                |
  | CKPT_LOAD_DIR | Path to the weights saved during weight conversion and used for initial weight loading. If no initial weights are available, weights are initialized randomly. |

#### Fine-Tuning

Fine-tuning works similarly to pretraining.

```sh
cd MindSpeed-LLM
bash examples/mindspore/glm4/tune_glm4_9b_32k_full_ms.sh
```

As with pretraining, modify the preceding variables in the script based on your actual environment.

### Inference

Use the following command for inference:

```sh
# Use the 9B model as an example.
cd MindSpeed-LLM
bash examples/mindspore/glm4/generate_glm4_9b_ms.sh
```

Modify the following variables in the script based on your actual environment:

  | Variable  | Description                 |
  |--------|--------------------|
  | MASTER_ADDR | IP address of the master node in a multi-node scenario.         |
  | NODE_RANK | Sequence number of the corresponding node in a multi-node scenario.       |
  | CHECKPOINT | Path to the weights saved during training.          |
  | TOKENIZER_PATH | GLM-4-9B tokenizer directory. |

### Evaluation

Use the following command for evaluation:

```sh
# Use the 9B model as an example.
cd MindSpeed-LLM
bash examples/mindspore/glm4/evaluate_glm4_9b_ms.sh
```

Modify the following variables in the script based on your actual environment:

  | Variable  | Description                    |
  |--------|-----------------------|
  | MASTER_ADDR | IP address of the master node in a multi-node scenario.            |
  | NODE_RANK | Sequence number of the corresponding node in a multi-node scenario.          |
  | TOKENIZER_PATH | GLM-4-9B tokenizer directory.    |
  | CKPT_LOAD_DIR | Path to the weights saved during weight conversion or training.         |
  |  DATA_PATH | Path to the dataset used for evaluation. Currently, we recommend MMLU. |
  | TASK  | Dataset used for evaluation. Currently, we recommend MMLU.   |

When you use MMLU, the evaluation result is as follows:

```log
INFO:mindspeed_llm.tasks.evaluation.eval_impl.mmlu_eval:mmlu acc = 321/387=0.8294573643410853
total: 100%|█████████████████████████████████████████████████████| 3/3 [06:16<00:00, 128.12s/it]INFO:main:
                         subject   question_n   acc
0                      sociology          201   0.915423
1   high_school_european_history          165   0.884848
2                      astronomy          152   0.763158
3                      nutrition          306   0.781046
...
57                         total        14042   0.747543
INFO:main:MMLU Running Time:, 4332.02222272872925
```

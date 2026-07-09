# 1 Environment Configuration

For the installation procedure of the MindSpore backend of MindSpeed LLM, see the [MindSpeed LLM Installation Guide](../../../docs/en/mindspore/install_guide.md).

# 2 Weight Conversion

## 2.1 Weight Download

Download the model weights and other configuration files from [Hugging Face (Qwen3-0.6B is used as an example)](https://huggingface.co/Qwen/Qwen3-0.6B/tree/main). If you need to continue pretraining, fine-tune, or run inference based on the open-source weights, also download the network model files.

## 2.2 Weight Conversion

The script converts Hugging Face open-source weights to MCore weights for tasks such as training, inference, and evaluation. Use the script as follows. Modify the weight conversion script based on the required tensor parallelism (TP), pipeline parallelism (PP), and weight paths:

```shell
cd MindSpeed-LLM
bash examples/mindspore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

After you run the script, you can expect log output similar to the following, which indicates that weight conversion succeeds:

```shell
successfully saved checkpoint from iteration 1 to ./model_weights/qwen3_mcore/
INFO:root:Done!
```

* Note: Model weights converted by the MindSpore backend cannot be used for training or inference on the Torch backend.

# 3 Data Preprocessing

The MindSpore backend currently fully supports data preprocessing for multiple MindSpeed LLM task scenarios. For the data preprocessing guide, see [Data Preprocessing](https://gitcode.com/ascend/MindSpeed-LLM/blob/master/docs/en/pytorch/tools/data_process_pretrain.md).

## 3.1 Processing Pretraining Data

Using the Alpaca dataset as an example, modify the `data_convert_qwen3_pretrain.sh` pretraining script.

Configure the data input and output paths and the tokenizer model path, and then run the script:

```shell
bash examples/mindspore/qwen3/data_convert_qwen3_pretrain.sh
```

The pretraining dataset processing results are as follows:

```shell
./dataset/alpaca_text_document.bin
./dataset/alpaca_text_document.idx
```

## 3.2 Processing Fine-Tuning Data

Using the Alpaca dataset as an example, modify the `data_convert_qwen3_instruction.sh` fine-tuning script.

Configure the data input and output paths and the tokenizer model path, and then run the script:

```shell
bash examples/mindspore/qwen3/data_convert_qwen3_instruction.sh
```

The fine-tuning dataset processing results are as follows:

```shell
./finetune_dataset/alpaca_packed_attention_mask_document.bin
./finetune_dataset/alpaca_packed_attention_mask_document.idx
./finetune_dataset/alpaca_packed_input_ids_document.bin
./finetune_dataset/alpaca_packed_input_ids_document.idx
./finetune_dataset/alpaca_packed_labels_document.bin
./finetune_dataset/alpaca_packed_labels_document.idx
```

# 4 Training

## 4.1 Pretraining

Modify the related parameters in the `pretrain_qwen3_0point6b_4K_ms.sh` script, and then run the script.

```shell
cd MindSpeed-LLM
bash examples/mindspore/qwen3/pretrain_qwen3_0point6b_4K_ms.sh
```

| Variable | Description |
| --- | --- |
| MASTER_ADDR | IP address of the master node in multi-node scenarios. |
| NODE_RANK | Sequence number of each node in multi-node scenarios. |
| DATA_PATH | Path of the preprocessed data. |
| TOKENIZER_PATH | Qwen3-0.6B tokenizer directory. |
| CKPT_LOAD_DIR | Directory for loading initial weights. If no initial weights exist, weights are randomly initialized. |
| CKPT_SAVE_DIR | Directory for saving weights during training. |
| TRAIN_ITERS | Number of training iterations. |

* Note: The 0.6B model is small, and one node is usually sufficient.

## 4.2 Fine-Tuning

Modify the related parameters in the `tune_qwen3_0point6b_4K_full_ms.sh` script, and then run the script.

```shell
cd MindSpeed-LLM
bash examples/mindspore/qwen3/tune_qwen3_0point6b_4K_full_ms.sh
```

| Variable | Description |
| --- | --- |
| MASTER_ADDR | IP address of the master node in multi-node scenarios. |
| NODE_RANK | Sequence number of each node in multi-node scenarios. |
| DATA_PATH | Path of the preprocessed data. |
| TOKENIZER_PATH | Qwen3-0.6B tokenizer directory. |
| CKPT_LOAD_DIR | Directory for loading initial weights. If no initial weights exist, weights are randomly initialized. |
| CKPT_SAVE_DIR | Directory for saving weights after training. |
| TRAIN_ITERS | Number of training iterations. |

* Note 1: The 0.6B model is small, and one node is usually sufficient.
* Note 2: For `CKPT_LOAD_DIR`, select the weights saved after pretraining.

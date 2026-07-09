# Edge-Cloud Collaborative Distributed Secure Training

## How to Use

Because the edge-cloud collaborative distributed training feature currently supports only the Qwen2.5 and Qwen3 series models, this document uses Qwen3-32B as an example (`PP=2`, 64 total hidden layers) to describe how to enable it. Perform the following steps:

1. Refer to [MindSpeed LLM Installation Guide](../../install_guide.md) to complete the environment setup.

    Before training starts, configure the Ascend NPU suite environment variables as follows:

    ```shell
    source /usr/local/Ascend/cann/set_env.sh     # Replace this with the actual Toolkit installation path
    source /usr/local/Ascend/nnal/atb/set_env.sh # Replace this with the actual nnal package installation path
    ```

2. Prepare the model weights and fine-tuning dataset.

    The complete Qwen3-32B model directory should contain the following files:

    ```shell
    .
    ├── README.md                    # Model documentation
    ├── config.json                  # Model architecture configuration file
    ├── generation_config.json       # Configuration for text generation
    ├── merges.txt                   # tokenizer merge rules file
    ├── model-00001-of-00017.safetensors  # Part 1 of the model weight files, 17 parts in total
    ├── model-00002-of-00017.safetensors  # Part 2 of the model weight files
    ├── ...
    ├── model-00016-of-00017.safetensors  # Part 16 of the model weight files
    ├── model-00017-of-00017.safetensors  # Part 17 of the model weight files
    ├── model.safetensors.index.json      # Weight shard index file that maps each parameter to its file
    ├── tokenizer.json              # Tokenizer in the Hugging Face format
    ├── tokenizer_config.json       # Tokenizer-related configuration
    └── vocab.json                  # Model vocabulary file
    ```

3. Convert the Hugging Face weights to the MCore format.

    The edge-cloud collaborative distributed training uses U-shaped partitioning. The first and last layer weights of the model must be stored separately. For detailed configuration, see [Qwen3 weight conversion script](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh).

    Using the Qwen3-32B model with `TP=8` and `PP=2` as an example, modify the related path parameters and model partitioning configuration:

    ```shell
    --target-tensor-parallel-size 8          # TP partition size
    --target-pipeline-parallel-size 2        # PP partition size
    --num-layer-list 16,32,16               # U-shaped partitioning: 16 layers in the first stage, 32 hidden layers in the middle stage, and 16 layers in the last stage
    --load-dir ./model_from_hf/qwen3_hf/     # Path to the original Hugging Face model weights
    --save-dir ./model_weights/qwen3_mcore/  # Path to save the Megatron weights
    ```

    Parameters:

    - `--num-layer-list`: Non-uniform PP partitioning. Pass the number of hidden layers for each pipeline stage as `L0,...,LPP`, where L0 and LPP indicate the number of hidden layers in the first and last stages. For example, when `PP=2`, the value `16,32,16` means 16 layers in the first stage, 32 hidden layers in the middle stage, and 16 layers in the last stage.

    After you verify that the paths are correct, run the weight conversion script:

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

4. Convert the MCore-format model to the VPP format.

    The edge-cloud collaborative distributed training workflow requires you to merge the first and last layer weights into the VPP format. Run the merge script `convert_ckpt_pp_vpp.py` as follows:

    ```shell
    python mindspeed_llm/tasks/layerwise_disaggregated_training/convert_ckpt_pp_vpp.py merge \
        --load-dir ./model_weights/qwen3_mcore/ \
        --save-dir-edge ./model_weights/qwen3_vpp_edge/ \
        --save-dir-cloud ./model_weights/qwen3_vpp_cloud/ \
        --merge-stages 0,1 \
        --middle-stages 1
    ```

    The parameters are as follows:

    | Parameter | Description | Required |
    | --------- | ----------- | -------- |
    | --load-dir | Path to the weight files in the MCore format. | Yes |
    | --save-dir-edge | Path to save the edge-side weight files. | Yes |
    | --save-dir-cloud | Path to save the cloud-side weight files. | Yes |
    | --merge-stages  | PP stage indexes for the first and last layers, in the `0,PP` format. | Yes |
    | --middle-stages | PP stage indexes for the middle layers, in the `1,...,PP-1` format. | Yes |

5. Perform data preprocessing.

    Using the Alpaca dataset as an example, perform data preprocessing. For detailed configuration, see [Qwen3 data preprocessing script](../../../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh):

    ```shell
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # Path to the original dataset
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf               # Path to the Hugging Face tokenizer
    --output-prefix ./finetune_dataset/alpaca                       # Save path
    ```

    After you finish setting the relevant parameters, run the data preprocessing script:

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

6. Start fine-tuning.

    Configure the fine-tuning script for the model. For detailed configuration, see [Qwen3-32B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh). Modify the related path parameters and model partitioning configuration:

    ```shell
    CKPT_LOAD_DIR="./model_weights/qwen3_vpp_edge/"  # Path to load the edge-side weights
    CKPT_LOAD_CLOUD_DIR="./model_weights/qwen3_vpp_cloud/"  # Path to load the cloud-side weights
    CKPT_SAVE_DIR="./ckpt/qwen3_finetune/"           # Path to save the weights after fine-tuning
    DATA_PATH="./finetune_dataset/alpaca"            # Dataset path
    TOKENIZER_PATH="./model_from_hf/qwen3_hf"        # Vocabulary path
    TP=8                                             # TP partition size
    PP=2                                             # PP partition size
    ```

    Add the following parameters to the training script to enable edge-cloud collaborative distributed training:

    ```shell
    --layerwise-disaggregated-training               # Enable edge-cloud collaborative distributed secure training
    --num-layer-list 16,32,16                        # Non-uniform PP partitioning, which must match the weight conversion settings
    --num-virtual-stages-per-pipeline-rank 2         # Number of virtual pipeline stages. Set this value to 2
    ```

    After you finish setting the relevant parameters, run the fine-tuning script:

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

## Usage Constraints

### Supported Models

- The following Qwen2.5 and Qwen3 LLM models are supported:

    | Model Type | Specific Models                   |
    | ---------- | --------------------------------- |
    | LLM        | qwen3-32B, qwen2.5-32B, qwen2.5-72B |

- MoE models are not supported.

### Other Constraints

- LoRA is not supported.
- Standard VPP parallelism is not supported: the `--num-virtual-stages-per-pipeline-rank` parameter must be set to `2` to enable joint first-and-last-layer deployment.

## Notes

- The parallel configuration of training parameters, such as TP and PP, must match the configuration used during weight conversion.
- The edge-cloud collaborative distributed training uses U-shaped partitioning. The first and last layers of the model are deployed on the edge side at the same time, and the original samples do not need to be uploaded to the cloud.
- Cross-domain collaborative training uses pipeline orchestration optimization and computation-communication overlap to achieve efficient training in edge-cloud cross-domain connection scenarios.

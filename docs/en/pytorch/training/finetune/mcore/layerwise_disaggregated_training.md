# Edge-Cloud Collaborative Distributed Secure Training

## How to Use

Because the edge-cloud collaborative distributed training feature currently supports only the Qwen2.5 and Qwen3 series models, this document uses Qwen3-32B as an example (`PP=3`, 64 total hidden layers) to describe how to enable it. Perform the following steps:

1. Refer to [MindSpeed LLM Installation Guide](../../install_guide.md) to complete the environment setup.

    Before training starts, configure the Ascend NPU suite environment variables as follows:

    ```shell
    source /usr/local/Ascend/cann/set_env.sh     # Replace this with the actual Toolkit installation path.
    source /usr/local/Ascend/nnal/atb/set_env.sh # Replace this with the actual nnal package installation path.
    ```

2. Prepare the model weights and fine-tuning dataset.

    The complete Qwen3-32B model directory should contain the following files:

    ```shell
    .
    ├── README.md                    # Model documentation.
    ├── config.json                  # Model architecture configuration file.
    ├── generation_config.json       # Configuration for text generation.
    ├── merges.txt                   # tokenizer merge rules file.
    ├── model-00001-of-00017.safetensors  # Part 1 of the model weight files, 17 parts in total.
    ├── model-00002-of-00017.safetensors  # Part 2 of the model weight files.
    ├── ...
    ├── model-00016-of-00017.safetensors  # Part 16 of the model weight files.
    ├── model-00017-of-00017.safetensors  # Part 17 of the model weight files.
    ├── model.safetensors.index.json      # Weight shard index file that maps each parameter to its file.
    ├── tokenizer.json              # Tokenizer in the Hugging Face format.
    ├── tokenizer_config.json       # Tokenizer-related configuration.
    └── vocab.json                  # Model vocabulary file.
    ```

3. Convert the Hugging Face weights to the MCore format.

    The edge-cloud collaborative distributed training uses U-shaped model partitioning to support joint deployment of the first and last layers. For detailed configuration, see [the Qwen3 weight conversion script](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh).

    Weight conversion precautions:

    - After you enable the edge-cloud feature, the number of edge-side devices can be smaller than the cloud-side TP size. In that case, the edge-side TP size equals the number of edge-side devices. When you convert the weights, use the respective TP size for the edge side and the cloud side.

    - When you convert the weights, first use a pipeline size of PP+1. The extra pipeline stage stores the model's last layer. Then use the first-and-last-layer merge script to output weights for joint deployment of the first and last layers, and restore the pipeline size to PP.

    Parameters descriptions:

    - `--num-layer-list`: Configure non-uniform PP partitioning. Pass the number of hidden layers for each pipeline stage as `L0,...,LPP`, where L0 and LPP indicate the number of hidden layers in the first and last stages. For example, when `PP=3`, the value `1,31,31,1` means one layer in the first stage, 31 hidden layers in each middle stage, and one layer in the last stage.

    Using one device on the edge side and 16 devices on the cloud side as an example, and with `PP=3`, edge-side `TP=1`, and cloud-side `TP=8`, the detailed weight conversion steps are as follows.

    Step 1: Convert the weights on the edge side with `TP=1` and `PP=3`. Modify the related path parameters and model partitioning configuration.

    ```shell
    --target-tensor-parallel-size 1          # TP partition size.
    --target-pipeline-parallel-size 4        # PP partition size.
    --num-layer-list 1,31,31,1               # U-shaped partitioning: one layer in the first stage, 31 hidden layers in each middle stage, and one layer in the last stage.
    --load-dir ./model_from_hf/qwen3_hf/     # Path to the original Hugging Face model weights.
    --save-dir ./model_weights/qwen3_mcore_tp1/  # Path to save the Megatron weights.
    ```

    After you verify that the paths are correct, run the weight conversion script:

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

    Step 2: Convert the weights on the cloud side with `TP=8` and `PP=3`. Modify the related path parameters and model partitioning configuration.

    ```shell
    --target-tensor-parallel-size 8          # TP partition size.
    --target-pipeline-parallel-size 4        # PP partition size.
    --num-layer-list 1,31,31,1               # U-shaped partitioning: one layer in the first stage, 31 hidden layers in each middle stage, and one layer in the last stage.
    --load-dir ./model_from_hf/qwen3_hf/     # Path to the original Hugging Face model weights.
    --save-dir ./model_weights/qwen3_mcore_tp8/  # Path to save the Megatron weights.
    ```

    After you verify that the paths are correct, run the weight conversion script:

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

    Step 3: Convert the MCore-format model to the VPP format by using the first-and-last-layer merge script.

    The edge-cloud collaborative distributed training workflow requires you to merge the first and last layer weights into the VPP format. Run the merge script `convert_ckpt_pp_vpp.py` as follows:

    ```shell
    python mindspeed_llm/tasks/posttrain/ldt_sft/convert_ckpt_pp_vpp.py merge \
        --load-dir-edge ./model_weights/qwen3_mcore_tp1/ \
        --load-dir-cloud ./model_weights/qwen3_mcore_tp8/ \
        --save-dir-edge ./mmpath/Qwen3_32B_vtp/vpp_edge/ \
        --save-dir-cloud ./mmpath/Qwen3_32B_vtp/vpp_cloud/ \
        --merge-stages 0,3 \
        --middle-stages 1,2
    ```

    The parameter descriptions are as follows:

    | Parameter | Description | Required |
    | --------- | ----------- | -------- |
    | --load-dir-edge | Path to the edge-side weight files in the MCore format. | Yes |
    | --load-dir-cloud | Path to the cloud-side weight files in the MCore format. | Yes |
    | --save-dir-edge | Path to save the edge-side weight files. | Yes |
    | --save-dir-cloud | Path to save the cloud-side weight files. | Yes |
    | --merge-stages  | PP stage indexes for the first and last layers, in the `0,PP` format. | Yes |
    | --middle-stages | PP stage indexes for the middle layers, in the `1,...,PP-1` format. | Yes |

4. Perform data preprocessing.

    Using the Alpaca dataset as an example, perform data preprocessing. For detailed configuration, see [the Qwen3 data preprocessing script](../../../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh):

    ```shell
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # Path to the original dataset.
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf               # Path to the Hugging Face tokenizer.
    --output-prefix ./finetune_dataset/alpaca                       # Save path.
    ```

    After you finish setting the relevant parameters, run the data preprocessing script:

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

5. Start fine-tuning.

    Configure the fine-tuning script for the model. For detailed configuration, see [the Qwen3-32B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh). Modify the related path parameters and model partitioning configuration:

    ```shell
    CKPT_LOAD_DIR="./model_weights/qwen3_vpp_edge/"  # Path to load the edge-side weights.
    CKPT_LOAD_CLOUD_DIR="./model_weights/qwen3_vpp_cloud/"  # Path to load the cloud-side weights.
    CKPT_SAVE_DIR="./ckpt/qwen3_finetune/"           # Path to save the weights after fine-tuning.
    DATA_PATH="./finetune_dataset/alpaca"            # Dataset path.
    TOKENIZER_PATH="./model_from_hf/qwen3_hf"        # Vocabulary path.
    TP=8                                             # TP partition size.
    PP=3                                             # PP partition size.
    ```

    Add the following parameters to the training script to enable edge-cloud collaborative distributed training:

    ```shell
    --layerwise-disaggregated-training               # Enable edge-cloud collaborative distributed secure training.
    --num-layer-list 1,31,31,1                       # Non-uniform PP partitioning, which must match the weight conversion settings.
    --num-virtual-stages-per-pipeline-rank 2         # Number of virtual pipeline stages. Set this value to 2.
    ```

    In the training script, the edge side and the cloud side must set `NPUS_PER_NODE` to the actual number of devices on the local compute node. Using one edge-side device as an example, configure the following:

    ```shell
    NPUS_PER_NODE=1
    ```

    After you finish setting the relevant parameters, run the fine-tuning script:

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

## Usage Constraints

### Supported Models

- The following Qwen2.5 and Qwen3 series LLMs are supported:

    | Model Type | Specific Models |
    | -------- | ---------------- |
    | LLM      | qwen3-32B, qwen2.5-32B, qwen2.5-72B |

- MoE models are not supported yet.

### Other Constraints

- LoRA is not supported yet.
- Standard VPP parallelism is not supported yet. The `--num-virtual-stages-per-pipeline-rank` argument must be `2` to enable joint deployment of the first and last layers.

## Notes

- The parallel configuration of training parameters, such as TP and PP, must match the configuration used during weight conversion.
- The edge-cloud collaborative distributed training uses U-shaped partitioning. The first and last layers of the model are deployed on the edge side at the same time, and the original samples do not need to be uploaded to the cloud.
- Cross-domain collaborative training uses pipeline orchestration optimization and computation-communication overlap to achieve efficient training in edge-cloud cross-domain connection scenarios.

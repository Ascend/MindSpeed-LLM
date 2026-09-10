# Edge-Cloud Collaborative Distributed Trusted Training

## How to Use

Because the edge-cloud collaborative distributed trusted training feature currently supports only the Qwen2.5 and Qwen3 series models, this document uses the Qwen3-32B model as an example (`PP=3`, with 64 hidden layers in total) to describe how to enable the feature. Perform the following steps:

### Prerequisites

1. Refer to [MindSpeed LLM Installation](../../install_guide.md) to complete the environment setup.

    Before training starts, configure the environment variables related to the Ascend NPU suite as follows:

    ```shell
    source /usr/local/Ascend/cann/set_env.sh     # Replace this with the actual Toolkit package installation path
    source /usr/local/Ascend/nnal/atb/set_env.sh # Replace this with the actual nnal package installation path
    ```

2. Prepare the model weights and the fine-tuning dataset.

    The complete Qwen3-32B model directory should contain the following files:

    ```shell
    .
    ├── README.md                    # Model documentation
    ├── config.json                  # Model architecture configuration file
    ├── generation_config.json       # Configuration for text generation
    ├── merges.txt                   # Tokenizer merge rules file
    ├── model-00001-of-00017.safetensors  # Part 1 of the model weight files (17 parts in total)
    ├── model-00002-of-00017.safetensors  # Part 2 of the model weight files
    ├── ...
    ├── model-00016-of-00017.safetensors  # Part 16 of the model weight files
    ├── model-00017-of-00017.safetensors  # Part 17 of the model weight files
    ├── model.safetensors.index.json      # Weight shard index file that maps each parameter to its file
    ├── tokenizer.json              # Tokenizer in the Hugging Face format
    ├── tokenizer_config.json       # Tokenizer-related configuration
    └── vocab.json                  # Model vocabulary file
    ```

3. Preprocess the data.

    Using the Alpaca dataset as an example, preprocess the data. For detailed configuration, see [the Qwen3 data preprocessing script](../../../../../../examples/mcore/qwen3/data_convert_qwen3_instruction.sh):

    ```shell
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet # Path to the original dataset
    --tokenizer-name-or-path ./model_from_hf/qwen3_hf               # Path to the Hugging Face tokenizer
    --output-prefix ./finetune_dataset/alpaca                       # Save path
    ```

    After you finish setting the relevant parameters, run the data preprocessing script:

    ```shell
    bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
    ```

### Model Fine-Tuning

Edge-cloud collaborative distributed trusted training supports four modes: symmetric TP and DP with joint deployment of the first and last layers, asymmetric TP, asymmetric DP, and asymmetric TP and DP. The model partitioning and fine-tuning scripts differ slightly among the modes. The following sections introduce the four modes separately.

#### Mode 1: Joint Deployment of the First and Last Layers, Symmetric TP and DP

1. Convert the weights, transforming the Hugging Face weights into the Megatron-Mcore format.

    Edge-cloud collaborative distributed training uses U-shaped model partitioning to satisfy the joint deployment of the first and last layers. For detailed configuration, see [the Qwen3 weight conversion script](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh).

    Weight conversion precautions:

    - When you convert the weights, first perform the conversion with a pipeline size of `PP+1`. The extra pipeline stage stores the last layer of the model. Then use the first-and-last-layer merge script to output weights for the joint deployment of the first and last layers, and restore the pipeline size to `PP`.

    Parameters:

    - `--num-layer-list`: Configures non-uniform PP partitioning. Pass the number of hidden layers for each pipeline stage as `L0,...,LPP`, where `L0` and `LPP` indicate the numbers of hidden layers in the first and last stages. For example, when `PP=3`, passing `1,31,31,1` means one layer in the first stage, 31 hidden layers in each middle stage, and one layer in the last stage.

    Using one device on the edge side and 16 devices on the cloud side as an example, and with `PP=3`, edge-side `TP=1`, and cloud-side `TP=8`, the detailed weight conversion steps are as follows.

    Step 1: Convert the weights on the edge side with `TP=1` and `PP=3`. Modify the related path parameters and model partitioning configuration.

    ```shell
    --target-tensor-parallel-size 1          # TP partition size
    --target-pipeline-parallel-size 4        # PP partition size, which is the actual PP+1 and is configured by the layers in num-layer-list
    --num-layer-list 1,31,31,1               # U-shaped partitioning: one layer in the first stage, 31+31 hidden layers in the middle stages, and one layer in the last stage
    --load-dir ./model_from_hf/qwen3_hf/     # Path to the original Hugging Face model weights
    --save-dir ./model_weights/qwen3_mcore_tp1/  # Path to save the Megatron weights
    ```

    After you verify that the paths are correct, run the weight conversion script:

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

    Step 2: Use the first-and-last-layer merge script to convert the model from the Megatron-Mcore format to the VPP format.

    Edge-cloud collaborative distributed training requires you to merge the first-and-last-layer weights into the VPP format. Run the weight conversion script `convert_ckpt_pp_vpp.py` to perform the operation:

    ```shell
    python mindspeed_llm/tasks/posttrain/ldt_sft/convert_ckpt_pp_vpp.py merge \
        --load-dir ./model_weights/qwen3_mcore_tp1/ \
        --save-dir-edge ./model_weights/qwen3_vpp_edge/ \
        --save-dir-cloud ./model_weights/qwen3_vpp_cloud/ \
        --merge-stages 0,3 \
        --middle-stages 1,2
    ```

    The parameters are as follows:

    | Parameter | Description | Required |
    | --------- | ----------- | -------- |
    | --load-dir | Path to load the weight files in the Megatron-Mcore format | Yes |
    | --save-dir-edge | Path to save the edge-side weight files | Yes |
    | --save-dir-cloud | Path to save the cloud-side weight files | Yes |
    | --merge-stages | PP stage indexes for the first and last layers, in the `0,PP` format | Yes |
    | --middle-stages | PP stage indexes for the middle layers, in the `1,...,PP-1` format | Yes |

2. Start fine-tuning.

    Configure the model fine-tuning script. For detailed configuration, see [the Qwen3-32B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh). Modify the related path parameters and model partitioning configuration:

    ```shell
    CKPT_LOAD_DIR="./model_weights/qwen3_vpp_edge/"  # Path to load the edge-side weights
    CKPT_LOAD_CLOUD_DIR="./model_weights/qwen3_vpp_cloud/"  # Path to load the cloud-side weights
    CKPT_SAVE_DIR="./ckpt/qwen3_finetune/"           # Path to save the weights after fine-tuning
    DATA_PATH="./finetune_dataset/alpaca"            # Dataset path
    TOKENIZER_PATH="./model_from_hf/qwen3_hf"        # Vocabulary path
    TP=8                                             # TP partition size
    PP=3                                             # PP partition size
    ```

    Add the following parameters to the training script to enable the edge-cloud collaborative distributed training feature:

    ```shell
    --layerwise-disaggregated-training               # Enable edge-cloud collaborative distributed trusted training
    --num-layer-list 1,31,31,1                       # Non-uniform PP partitioning, which must be consistent with the weight conversion settings
    --num-virtual-stages-per-pipeline-rank 2         # Number of virtual pipeline stages, which must be set to 2
    ```

    In the training script, the edge side and the cloud side must set `NPUS_PER_NODE` to the actual number of devices on the local compute node. Using one edge-side device as an example, configure the following:

    ```shell
    NPUS_PER_NODE=1
    ```

    After you finish setting the relevant parameters, run the fine-tuning script on the edge side and the cloud side separately:

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

#### Mode 2: Joint Deployment of the First and Last Layers, Asymmetric TP

1. Convert the weights, transforming the Hugging Face weights into the Megatron-Mcore format.

    Edge-cloud collaborative distributed training uses U-shaped model partitioning to satisfy the joint deployment of the first and last layers. For detailed configuration, see [the Qwen3 weight conversion script](../../../../../../examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh).

    Weight conversion precautions:

    - After you enable the edge-cloud feature, the number of edge-side devices can be smaller than the cloud-side TP size. In that case, the edge-side TP size equals the number of edge-side devices. When you convert the weights, the edge side and the cloud side each use their own TP size for the conversion.

    - When you convert the weights, first perform the conversion with a pipeline size of `PP+1`. The extra pipeline stage stores the last layer of the model. Then use the first-and-last-layer merge script to output weights for the joint deployment of the first and last layers, and restore the pipeline size to `PP`.

    Parameters:

    - `--num-layer-list`: Configures non-uniform PP partitioning. Pass the number of hidden layers for each pipeline stage as `L0,...,LPP`, where `L0` and `LPP` indicate the numbers of hidden layers in the first and last stages. For example, when `PP=3`, passing `1,31,31,1` means one layer in the first stage, 31 hidden layers in each middle stage, and one layer in the last stage.

    Using one device on the edge side and 16 devices on the cloud side as an example, and with `PP=3`, edge-side `TP=1`, and cloud-side `TP=8`, the detailed weight conversion steps are as follows.

    Step 1: Convert the weights on the edge side with `TP=1` and `PP=3`. Modify the related path parameters and model partitioning configuration.

    ```shell
    --target-tensor-parallel-size 1          # TP partition size
    --target-pipeline-parallel-size 4        # PP partition size
    --num-layer-list 1,31,31,1               # U-shaped partitioning: one layer in the first stage, 31+31 hidden layers in the middle stages, and one layer in the last stage
    --load-dir ./model_from_hf/qwen3_hf/     # Path to the original Hugging Face model weights
    --save-dir ./model_weights/qwen3_mcore_tp1/  # Path to save the Megatron weights
    ```

    After you verify that the paths are correct, run the weight conversion script:

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

    Step 2: Convert the weights on the cloud side with `TP=8` and `PP=3`. Modify the related path parameters and model partitioning configuration.

    ```shell
    --target-tensor-parallel-size 8          # TP partition size
    --target-pipeline-parallel-size 4        # PP partition size
    --num-layer-list 1,31,31,1               # U-shaped partitioning: one layer in the first stage, 31+31 hidden layers in the middle stages, and one layer in the last stage
    --load-dir ./model_from_hf/qwen3_hf/     # Path to the original Hugging Face model weights
    --save-dir ./model_weights/qwen3_mcore_tp8/  # Path to save the Megatron weights
    ```

    After you verify that the paths are correct, run the weight conversion script:

    ```shell
    bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
    ```

    Step 3: Use the first-and-last-layer merge script to convert the model from the Megatron-Mcore format to the VPP format.

    Edge-cloud collaborative distributed training requires you to merge the first-and-last-layer weights into the VPP format. Run the weight conversion script `convert_ckpt_pp_vpp.py` to perform the operation:

    ```shell
    python mindspeed_llm/tasks/posttrain/ldt_sft/convert_ckpt_pp_vpp.py merge \
        --load-dir-edge ./model_weights/qwen3_mcore_tp1/ \
        --load-dir-cloud ./model_weights/qwen3_mcore_tp8/ \
        --save-dir-edge ./model_weights/qwen3_vpp_edge/ \
        --save-dir-cloud ./model_weights/qwen3_vpp_cloud/ \
        --merge-stages 0,3 \
        --middle-stages 1,2
    ```

    The parameters are as follows:

    | Parameter | Description | Required |
    | --------- | ----------- | -------- |
    | --load-dir-edge | Path to load the edge-side weight files in the Megatron-Mcore format | Yes |
    | --load-dir-cloud | Path to load the cloud-side weight files in the Megatron-Mcore format | Yes |
    | --save-dir-edge | Path to save the edge-side weight files | Yes |
    | --save-dir-cloud | Path to save the cloud-side weight files | Yes |
    | --merge-stages | PP stage indexes for the first and last layers, in the `0,PP` format | Yes |
    | --middle-stages | PP stage indexes for the middle layers, in the `1,...,PP-1` format | Yes |

2. Start fine-tuning.

    Configure the model fine-tuning script. For detailed configuration, see [the Qwen3-32B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh). Modify the related path parameters and model partitioning configuration:

    ```shell
    CKPT_LOAD_DIR="./model_weights/qwen3_vpp_edge/"  # Path to load the edge-side weights
    CKPT_LOAD_CLOUD_DIR="./model_weights/qwen3_vpp_cloud/"  # Path to load the cloud-side weights
    CKPT_SAVE_DIR="./ckpt/qwen3_finetune/"           # Path to save the weights after fine-tuning
    DATA_PATH="./finetune_dataset/alpaca"            # Dataset path
    TOKENIZER_PATH="./model_from_hf/qwen3_hf"        # Vocabulary path
    TP=8                                             # TP partition size
    PP=3                                             # PP partition size
    ```

    **Note: In the asymmetric TP scenario, the edge-side and cloud-side TP sizes must be set to the same value. You cannot configure the edge-side TP size based on the actual number of devices.**

    Add the following parameters to the training script to enable the edge-cloud collaborative distributed training feature:

    ```shell
    --layerwise-disaggregated-training               # Enable edge-cloud collaborative distributed trusted training
    --num-layer-list 1,31,31,1                       # Non-uniform PP partitioning, which must be consistent with the weight conversion settings
    --num-virtual-stages-per-pipeline-rank 2         # Number of virtual pipeline stages, which must be set to 2
    ```

    In the training script, the edge side and the cloud side must set `NPUS_PER_NODE` to the actual number of devices on the local compute node. Using one edge-side device as an example, configure the following:

    ```shell
    NPUS_PER_NODE=1
    ```

    After you finish setting the relevant parameters, run the fine-tuning script on the edge side and the cloud side separately:

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

#### Mode 3: Joint Deployment of the First and Last Layers, Asymmetric DP

1. The weight conversion operations in Mode 3 are the same as those in Mode 1. Perform the weight conversion by following the instructions for Mode 1.

2. Start fine-tuning.

    Configure the model fine-tuning script. For detailed configuration, see [the Qwen3-32B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh). Modify the related path parameters and model partitioning configuration:

    ```shell
    WORLD_SIZE=40                                    # Total number of devices, including the edge side and the cloud side. For example, with one edge-side machine with 8 devices and four cloud-side machines with 8 devices each, WORLD_SIZE=40
    CKPT_LOAD_DIR="./model_weights/qwen3_vpp_edge/"  # Path to load the edge-side weights
    CKPT_LOAD_CLOUD_DIR="./model_weights/qwen3_vpp_cloud/"  # Path to load the cloud-side weights
    CKPT_SAVE_DIR="./ckpt/qwen3_finetune/"           # Path to save the weights after fine-tuning
    DATA_PATH="./finetune_dataset/alpaca"            # Dataset path
    TOKENIZER_PATH="./model_from_hf/qwen3_hf"        # Vocabulary path
    TP=8                                             # TP partition size
    PP=3                                             # PP partition size
    ```

    **Note: In the asymmetric DP scenario, you must set `WORLD_SIZE` to the number of edge-side devices plus the number of cloud-side devices. Do not use the default `WORLD_SIZE=$(($NPUS_PER_NODE*$NODES))` calculation.**

    Add the following parameters to the training script to enable the edge-cloud collaborative distributed training feature:

    ```shell
    --layerwise-disaggregated-training               # Enable edge-cloud collaborative distributed trusted training
    --num-layer-list 1,31,31,1                       # Non-uniform PP partitioning, which must be consistent with the weight conversion settings
    --num-virtual-stages-per-pipeline-rank 2         # Number of virtual pipeline stages, which must be set to 2
    ```

    In the training script, the edge side and the cloud side must set `NPUS_PER_NODE` to the actual number of devices on the local compute node. Using one edge-side device as an example, configure the following:

    ```shell
    NPUS_PER_NODE=1
    ```

    After you finish setting the relevant parameters, run the fine-tuning script on the edge side and the cloud side separately:

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

#### Mode 4: Joint Deployment of the First and Last Layers, Asymmetric TP and DP

1. The weight conversion operations in Mode 4 are the same as those in Mode 2. Perform the weight conversion by following the instructions for Mode 2.

2. Start fine-tuning.

    Configure the model fine-tuning script. For detailed configuration, see [the Qwen3-32B fine-tuning script](../../../../../../examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh). Modify the related path parameters and model partitioning configuration:

    ```shell
    WORLD_SIZE=33                                    # Total number of devices, including the edge side and the cloud side. For example, with one edge-side machine with 1 device and four cloud-side machines with 8 devices each, WORLD_SIZE=33
    CKPT_LOAD_DIR="./model_weights/qwen3_vpp_edge/"  # Path to load the edge-side weights
    CKPT_LOAD_CLOUD_DIR="./model_weights/qwen3_vpp_cloud/"  # Path to load the cloud-side weights
    CKPT_SAVE_DIR="./ckpt/qwen3_finetune/"           # Path to save the weights after fine-tuning
    DATA_PATH="./finetune_dataset/alpaca"            # Dataset path
    TOKENIZER_PATH="./model_from_hf/qwen3_hf"        # Vocabulary path
    TP=8                                             # TP partition size
    PP=3                                             # PP partition size
    ```

    **Note: In the asymmetric TP and DP scenario, the edge-side and cloud-side TP sizes must be set to the same value. You cannot configure the edge-side TP size based on the actual number of devices. In addition, you must set `WORLD_SIZE` to the number of edge-side devices plus the number of cloud-side devices. Do not use the default `WORLD_SIZE=$(($NPUS_PER_NODE*$NODES))` calculation.**

    Add the following parameters to the training script to enable the edge-cloud collaborative distributed training feature:

    ```shell
    --layerwise-disaggregated-training               # Enable edge-cloud collaborative distributed trusted training
    --num-layer-list 1,31,31,1                       # Non-uniform PP partitioning, which must be consistent with the weight conversion settings
    --num-virtual-stages-per-pipeline-rank 2         # Number of virtual pipeline stages, which must be set to 2
    ```

    In the training script, the edge side and the cloud side must set `NPUS_PER_NODE` to the actual number of devices on the local compute node. Using one edge-side device as an example, configure the following:

    ```shell
    NPUS_PER_NODE=1
    ```

    After you finish setting the relevant parameters, run the fine-tuning script on the edge side and the cloud side separately:

    ```shell
    bash examples/mcore/qwen3/tune_qwen3_32b_4K_full_ptd.sh
    ```

## Notes

- The parallel configuration of training parameters, such as TP and PP, must match the configuration used during weight conversion.
- Edge-cloud collaborative distributed training uses the U-shaped partitioning scheme. The first and last layers of the model are deployed on the edge side at the same time, and the original samples do not need to be uploaded to the cloud.
- Cross-domain collaborative training uses pipeline orchestration optimization and computation-communication overlap to achieve efficient training in edge-cloud cross-domain connection scenarios.

# LU-LoRA Fine-Tuning Introduction

Local-update LoRA (LU-LoRA) is a novel parameter-efficient fine-tuning (PEFT) method. Compared with full fine-tuning and PEFT methods such as LoRA, it significantly reduces memory usage and speeds up fine-tuning. This method combines local learning rules with the parameter-efficient LoRA method in an effective way. Figure 1a shows the conventional method that updates adapters through backpropagation. LU-LoRA proposes replacing backpropagation-based weight updates with local updates in some adapters. The overall scheme is shown in Figure 1b.

## How LU-LoRA Works

The main components of this method are the following:

1. Effective combination of backpropagation and local learning during fine-tuning
2. Special initialization for LoRA adapters updated with the Hebb rule
3. Different learning rates for LoRA adapters

![alt text](../../../figures/lu_lora_finetune/lu_lora_model.png)
*Figure 1: Fine-tuning scheme*

## Usage Instructions

MindSpeed LLM supports mixed low-parameter training with LU-LoRA for fine-tuning and other tasks. You enable it by adding LU-LoRA parameters to the baseline task. Here, we use a fine-tuning task as an example to explain how to use LU-LoRA in the baseline task.

### Data Preprocessing Example

For the naming and startup method of the MindSpeed LLM fine-tuning data preprocessing script, see the corresponding data preprocessing document for other baseline tasks:

```shell
bash examples/mcore/llama2/data_convert_llama2_instruction.sh
```

During data preprocessing, if `output-prefix` is `./finetune_dataset/llama-2-7b/alpaca`, use the following command:

```shell
python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf \
    --output-prefix ./finetune_dataset/llama-2-7b/alpaca \
    --workers 16 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2
```

### Hf2mcore Weight Conversion Example

The MindSpeed LLM LU-LoRA fine-tuning script can use normal MCore base weights for fine-tuning:

```shell
bash examples/mcore/llama2/ckpt_convert_llama2_hf2mcore.sh
```

In the hf2mcore weight conversion script, use the following command:

```shell
# Convert the weight format. Configure the required parallel settings and use --num-layers-per-virtual-pipeline-stage 5 together with --params-dtype bf16 as needed.
python convert_ckpt.py \
 --model-type GPT \
 --load-model-type hf \
 --save-model-type mg \
 --target-tensor-parallel-size 8 \
 --target-pipeline-parallel-size 1 \
 --load-dir ./model_from_hf/llama-2-7b-hf/ \
 --save-dir ./model_weights/llama2-mcore/ \
 --tokenizer-model ./model_from_hf/llama-2-7b-hf/tokenizer.model \
 --use-mcore-models \
 --model-type-hf llama2
```

### LU-LoRA Fine-Tuning Script Details

When you run the corresponding command for fine-tuning, `DATA_PATH` should stay consistent:

```shell
DATA_PATH="./finetune_dataset/llama-2-7b/alpaca" # Dataset path.
CKPT_LOAD_DIR="./model_weights/llama2-mcore/" # Weight path.
```

The naming and startup method of the MindSpeed LLM LU-LoRA fine-tuning script are as follows:

```shell
# Initialize environment variables.
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.
source /usr/local/Ascend/nnal/atb/set_env.sh # Replace this with the actual nnal package installation path.
# Start the job.
bash examples/mcore/llama2/tune_llama2_7b_lu_lora_ptd.sh
```

Parameters:

- **`--lu-lora-final-layer-index`**
  The index of the last layer trained with LU-LoRA. For example, if the value is 11, the first 12 layers of the model, from layer 0 to layer 11, use the LU-LoRA algorithm, and the remaining layers use the LoRA algorithm. Models with more layers should update more layers for local learning to reduce compute and memory consumption. However, covering too many layers with LU-LoRA may limit model generalization. You are advised to start with 37.5 percent of the layers to achieve a suitable memory reduction. Further hyperparameter tuning can improve the result.
- **`--lu-lora-lr-ratio`**
  The learning-rate ratio between the LU-LoRA B and LU-LoRA A adapters. The default recommended value is 24, but it may be lower according to the experimental results for a specific model.
- **`--lu-lora-lr`**
  The initial learning rate for the LU-LoRA layers. The default is 1.25e-6. The learning rate for LU-LoRA adapters can differ from the learning rate for LoRA adapters.

### Merging and Converting LU-LoRA Weights with the Base Weights

After LU-LoRA fine-tuning, the obtained LU-LoRA weights differ from the base weights and cannot be used directly for inference or continued training. You need to merge them with the base weights before you can use them. Because LU-LoRA is based on LoRA, you can use the LoRA conversion script. Add the following parameters to merge the trained LU-LoRA weights with the base weights and then convert them to MCore weights after merging:

```shell
 --lora-load ${CHECKPOINT_LORA} \
 --lora-r 16 \
 --lora-alpha 32 \
 --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
```

The following is an example command for converting LoRA weights to MCore weights:

```shell
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.

python convert_ckpt.py \
 --use-mcore-models \
 --model-type GPT \
 --load-model-type mg \
 --save-model-type mg \
 --load-dir ./model_weights/llama-2-7b-mcore \
 --lora-load ./ckpt/llama-7b-lora-mcore-tp1pp1 \
 --save-dir ./model_weights/llama2-7b-lora2mcore \
 --lora-r 16 \
 --lora-alpha 32 \
 --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
 --target-tensor-parallel-size 1 \
 --target-pipeline-parallel-size 1 \
 --model-type-hf llama2
```

The following is an example of a launch script similar to the one used for LoRA conversion:

```shell
# Start the job.
bash examples/mcore/llama2/ckpt_convert_llama2_mg2mg_lora.sh
```

#### Merging LU-LoRA Weights and Converting Them to Hugging Face Weights

If you want to merge LU-LoRA weights and convert them to the Hugging Face format, you can use the same LoRA conversion command:

```shell
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.

python convert_ckpt.py \
    --model-type GPT \
    --use-mcore-models \
    --load-model-type mg \
    --save-model-type hf \
    --load-dir ./model_weights/llama-2-7b-mcore/ \
    --lora-load ./ckpt/llama-7b-lora-mcore-tp1pp1 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama-2-7b-hf/ # Fill in the original Hugging Face model path. The new weights will be stored in ./model_from_hf/llama-2-7b-hf/mg2hg/
```

The example launch script is as follows:

```shell
# Start the job.
bash examples/mcore/llama2/ckpt_convert_llama2_mcore2hf_lora.sh
```

**Note:** The value of the `lora` parameters should match the parameter settings used during fine-tuning to ensure that the converted model has the same performance and compatibility.

### LU-LoRA Inference

The naming and startup method of the MindSpeed LLM inference script are as follows:

```shell
# Initialize environment variables.
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit installation path.
source /usr/local/Ascend/nnal/atb/set_env.sh # Replace this with the actual nnal package installation path.
```

Before you start, modify the model weight path and tokenizer path in the launch script according to the actual situation:

```shell
CHECKPOINT="./model_weights/llama-2-7b-mcore"
CHECKPOINT_LORA="./ckpt/llama-2-7b-lora/"
TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf/"

# Start the job.
bash examples/mcore/llama2/generate_llama2_7b_lora_ptd.sh
```

### LU-LoRA Fine-Tuning Weight Evaluation

The dedicated evaluation script for LU-LoRA fine-tuning weights is the same as the one used for LoRA.

## References

- [Going beyond classical LLM LoRA fine-tuning with Hebb learning: blazingly fast and accurate (European Conference on Artificial Intelligence (accepted), 2025)](https://ecai2025.org/accepted-papers/)
- [Implementation Challenges and Strategies for Hebbian Learning in Convolutional Neural Networks (Optical Memory and Neural Networks Journal, 2023)](https://dl.acm.org/doi/abs/10.3103/S1060992X23060048)

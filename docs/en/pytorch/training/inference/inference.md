# MindSpeed LLM Streaming Inference

## Streaming Inference Results

`ChatGLM3-6B-Base streaming inference results`

```shell
Instruction: "how are you?"
MindSpeed-LLM:   "I'm just a computer program, so I don't have feelings or physical sensations, \
              but I'm here to help you with any questions you might have. \
              Is there something specific you would like to know?"
HuggingFace: "I'm just a computer program, so I don't have feelings or physical sensations, \
              but I'm here to help you with any questions you might have. \
              Is there something specific you would like to know?"
```

`Llama-3.1-8B-Instruct streaming inference results`

```shell
Instruction: "how are you?"
MindSpeed-LLM:   "I hope you are doing well. I am writing to ask for your help with a project I am working on. \
              I am a student at [University Name] and I am doing a research project on [Topic]."
HuggingFace: "I hope you are doing well. I am writing to ask for your help with a project I am working on. \
              I am a student at [University Name] and I am doing a research project on [Topic]."
```

## Streaming Inference Example

### Initializing Environment Variables

```shell
source /usr/local/Ascend/cann/set_env.sh # Replace this with the actual Toolkit package path.
source /usr/local/Ascend/nnal/atb/set_env.sh # Replace this with the actual nnal package path.
```

### Launching the Script

Use the [streaming inference script](../../../../../examples/mcore/llama2/generate_llama2_13b_ptd.sh) in the Llama-2-7B model directory.

#### Setting Relevant Paths

`CHECKPOINT`: Points to the path where the converted weights are stored.

`TOKENIZER_PATH`: Specifies the folder that contains the model tokenizer.

`TOKENIZER_MODEL`: Specifies the tokenizer file path for the model, for example `tokenizer.model`.

Therefore, based on the preceding example, set the paths as follows:

```shell
CHECKPOINT="./model_weights/llama-2-13b-mcore/"
TOKENIZER_PATH="./model_from_hf/llama-2-13b-hf/"
TOKENIZER_MODEL="./model_from_hf/llama-2-13b-hf/tokenizer.model"
```

`Note`Apart from the path configuration, all other parameters in the inference script must match the training parameters, including the parallel partition settings for TP, PP, EP, and VPP, and custom model structure parameters such as `--noop-layers` and `--num-layer-list`. Otherwise, the model weights may fail to load.

#### Running the Script

```shell
bash examples/mcore/llama2/generate_llama2_13b_ptd.sh
```

#### Streaming Inference Script Parameters

`--task`

Specifies the generation strategy used during inference. It supports `greedy_search`, `greedy_search_with_sampling`, `beam_search`, `beam_search_with_sampling`, and the `chat` strategy.

`--stream`

Enables streaming output during inference.

`--max-new-tokens`

The maximum number of tokens to generate. It does not count the number of tokens in the input prompt.

`--max-length`

The maximum generated length. It equals the length of the input prompt plus `max_new_tokens`. If you set `max_new_tokens`, this parameter has no effect.

`--add-eos-token`

Sets the token ID used to generate the end-of-sequence token. Multiple end-of-sequence tokens are supported.

`--top-k`

The default value is `0`. If you set this to an integer greater than `1`, the system keeps the `k` vocabulary tokens with the highest probabilities for top-k sampling.

`--top-p`

The default value is `1.0`. If you set this to a floating-point number less than `1`, the system keeps only the smallest set of the most probable tokens whose cumulative probability is `top-p` or higher for sampling.

`--temperature`

The default value is `1.0`. This temperature coefficient is used in the sampling strategy.

`--num-beams`

Sets the beam size when you use beam search.

`--use-kv-cache`

Enables `kv-cache` during online inference to accelerate generation.

`--use-flash-attn`

Enables `prompt_flash_attention` and `incre_flash_attention` during online inference to accelerate generation. You must enable it together with `kv-cache`. Currently, it supports MHA, GQA, and Alibi.

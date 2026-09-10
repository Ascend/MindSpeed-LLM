# MindSpeed LLM FSDP2 Backend Model Adaptation Guide

This document uses gpt-oss as an example to describe how to integrate Hugging Face models into the FSDP2 training backend based on MindSpeed LLM, and covers the complete workflow of weight download, dataset download, model adaptation, YAML configuration, and training launch.

The FSDP2 training backend supports the following two adaptation paths:

| Path | Applicable Scenario | Complexity | Advantages |
| --- | --- | --- | --- |
| Native Transformers Adaptation | The model is already supported by the `transformers` library in the current environment and does not require changes to the model structure, operators, or `forward` logic. | Low | No new model code is required. The model is loaded automatically through `AutoModelForCausalLM`. |
| Custom Registry Adaptation | Secondary development based on the native Transformers implementation is needed. For example, injecting NPU-fused operators, expert parallelism, context parallelism, special attention, or MoE routing logic. | Medium | The model implementation can be directly controlled, making it suitable for performance optimization and hardware-friendly adaptation. |

gpt-oss in MindSpeed LLM uses the second path: first, migrate the model file from the native Transformers implementation, then adapt MoE expert parallelism and NPU grouped GEMM under `mindspeed_llm/fsdp2/models/gpt_oss/`.

## Obtaining Weights and Datasets

The following commands use `gpt-oss-20b` and Alpaca parquet data as examples. Adjust the model and data directories according to your actual machine disk paths.

### Downloading gpt-oss Weights

| Name | Link | Purpose |
| --- | --- | --- |
| gpt-oss-20b | [Hugging Face](https://huggingface.co/openai/gpt-oss-20b)/[ModelScope](https://modelscope.cn/models/unsloth/gpt-oss-20b-BF16/) | The 20B weights and tokenizer used in the examples of this document. |
| gpt-oss-120b | [Hugging Face](https://huggingface.co/openai/gpt-oss-120b)/[ModelScope](https://modelscope.cn/models/unsloth/gpt-oss-120b-BF16) | A larger model that can be adapted to the training configuration in the same way. |

Choose one of the following two download methods:

- Download using Git LFS:

  ```bash
  mkdir -p /home/data
  cd /home/data
  git lfs install
  git clone https://huggingface.co/openai/gpt-oss-20b gpt-oss-20b-hf
  ```

- Download using the Hugging Face CLI:

  ```bash
  hf download openai/gpt-oss-20b \
    --local-dir /home/data/gpt-oss-20b-hf
  ```

After the download is complete, the model directory should contain the `config.json`, tokenizer files, and safetensors weight files.

### Downloading the Example Dataset

| Name | Link | Purpose |
| --- | --- | --- |
| Alpaca parquet file | [Alpaca train-00000 parquet (Hugging Face)](https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet) | Can be downloaded directly to the local machine as example data. |

Download the parquet file:

```bash
mkdir -p /home/data/alpaca
wget -O /home/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
```

## Model Adaptation

### Path 1: Native Transformers Adaptation

If the current `transformers` version already supports the target model, such as gpt-oss or Qwen3, you can start FSDP2 training directly using the Hugging Face model directory without modifying the model source code.

1. Check the model directory.

    The model directory must contain at least the `config.json`, tokenizer files, and weight files. An example `config.json` for gpt-oss:

    ```json
    {
      // The model class name used by Hugging Face when loading the model
      "architectures": [
        "GptOssForCausalLM"
      ],
      // Whether the attention linear layer has a bias; attention_dropout is the attention dropout probability
      "attention_bias": true,
      "attention_dropout": 0.0,
      // End-of-document and padding token IDs
      "eos_token_id": 200002,
      "pad_token_id": 199999,
      // Number of experts each token is routed to; num_experts_per_tok is a synonym
      "experts_per_token": 4,
      "num_experts_per_tok": 4,
      // Dimension of a single attention head
      "head_dim": 64,
      "hidden_act": "silu",
      // Hidden layer dimension and MLP intermediate layer dimension
      "hidden_size": 2880,
      "intermediate_size": 2880,
      // Original context length, usually corresponding to the pre-RoPE-extension length
      "initial_context_length": 4096,
      "initializer_range": 0.02,
      // Maximum position length of the model
      "max_position_embeddings": 131072,
      // Model type identifier; gpt-oss uses gpt_oss
      "model_type": "gpt_oss",
      // Number of attention query heads and KV heads
      "num_attention_heads": 64,
      "num_key_value_heads": 8,
      // Number of Transformer layers
      "num_hidden_layers": 24,
      // Total number of local experts
      "num_local_experts": 32,
      // Whether to output router logits; can usually be kept as false for the main training pipeline
      "output_router_logits": false,
      // RMSNorm epsilon
      "rms_norm_eps": 1e-05,
      // RoPE extension configuration; the example uses YaRN
      "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
        "truncate": false
      },
      // RoPE frequency base
      "rope_theta": 150000,
      // MoE router auxiliary loss coefficient
      "router_aux_loss_coef": 0.9,
      // Sliding window attention window size
      "sliding_window": 128,
      // SwiGLU activation clip ceiling
      "swiglu_limit": 7.0,
      // Whether to share the input embedding and the output head weights
      "tie_word_embeddings": false,
      // Default dtype for weights; the gpt-oss example uses bfloat16
      "torch_dtype": "bfloat16",
      // Transformers version when this configuration was exported
      "transformers_version": "4.56.0.dev0",
      // Whether to enable KV cache during inference
      "use_cache": true,
      // Tokenizer vocabulary size
      "vocab_size": 201088
    }
    ```

2. Reuse the task YAML for startup.

    Native Transformers adaptation does not require adding new model source code. After preparing the weights, dataset, and `config.json`, you can reuse the task scripts and YAML configuration from [Configuring Task Scripts and YAML Files](#configuring-task-scripts-and-yaml-files) to start training. For the existing gpt-oss configuration structure in MindSpeed LLM, refer to `examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml`. If you use the Native Transformers path, focus on adjusting `model.model_name_or_path`, the data path, parallelism settings, and training parameters.

    If you later want to use NPU-fused operators, expert parallelism customization, or replace the MoE computation logic of gpt-oss, refer to [Path 2: Custom Registry Adaptation for gpt-oss](#path-2-custom-registry-adaptation-for-gpt-oss).

### Path 2: Custom Registry Adaptation for gpt-oss

gpt-oss is integrated into MindSpeed LLM through the custom registry approach. You are advised to start from the native Transformers implementation, migrate the model file to the FSDP2 model directory, and then carry out further development. This preserves the Hugging Face weight naming and model structure as much as possible, reducing weight loading discrepancies.

1. Place the model source code.

    Obtain the gpt-oss model file from the native Transformers implementation:

    ```text
    transformers/models/gpt_oss/modeling_gpt_oss.py
    transformers/models/gpt_oss/configuration_gpt_oss.py
    ```

    Place it in the following directory within MindSpeed LLM:

    ```text
    mindspeed_llm/fsdp2/models/
    └── gpt_oss/
        ├── __init__.py
        └── modeling_gpt_oss.py
    ```

    MindSpeed LLM already has `mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py`. The FSDP2 adaptation logic for gpt-oss can be maintained in this file.

2. Module optimization.

    gpt-oss is an MoE model, and the adaptation focus is on the MLP/Expert layers. The current implementation retains the native expert computation path while adding `GptOssFusedExperts`. When `moe_grouped_gemm` is enabled or the `fused` expert dispatch is used, NPU grouped GEMM is invoked.

    ```python
    class GptOssMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.router = GptOssTopKRouter(config)
            args = get_args()
            if args.moe_grouped_gemm or args.ep_dispatcher == "fused":
                self.experts = GptOssFusedExperts(config)
            else:
                self.experts = GptOssExperts(config)

        def forward(self, hidden_states):
            router_scores, router_indices = self.router(hidden_states)
            routed_out = self.experts(hidden_states, router_indices, router_scores)
            return routed_out, router_scores
    ```

    The following locations typically need to be carefully reviewed when adapting gpt-oss:

    | Adaptation Point | Description |
    | --- | --- |
    | Model class | `GptOssForCausalLM` must be exported for registration with `ModelRegistry`. |
    | Attention | If FlashAttention, long-sequence CP, or NPU-fused operators are needed, the corresponding `forward` must be replaced. |
    | MLP/Expert | MoE models typically need to adapt expert parallelism, token dispatch, and grouped GEMM. |
    | Norm/RoPE | If fused RMSNorm or fused RoPE is to be used, the corresponding operators must be connected in the model implementation. |
    | Weight names | Keep aligned with the key names in the Hugging Face weight file to avoid missing or unexpected keys during weight loading. |

3. Register the model class.

    Modify `mindspeed_llm/fsdp2/models/model_registry.py` to import the gpt-oss model class and add it to `_REGISTRY`. The current code example in MindSpeed LLM is as follows:

    ```python
    class ModelRegistry:
        from mindspeed_llm.fsdp2.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
        from mindspeed_llm.fsdp2.models.step35.modeling_step3p5 import Step3p5ForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3.qwen3 import Qwen3ForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3.qwen3_moe import Qwen3MoEForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3_next.qwen3_next import Qwen3NextForCausalLM
        from mindspeed_llm.fsdp2.models.mamba3.modeling_mamba3 import Mamba3ForCausalLM
        from mindspeed_llm.fsdp2.models.minimax_m27.modeling_minimax_m2 import MiniMaxM2ForCausalLM

        _REGISTRY = {
            "gpt_oss": GptOssForCausalLM,
            "step35": Step3p5ForCausalLM,
            "qwen3": Qwen3ForCausalLM,
            "qwen3_moe": Qwen3MoEForCausalLM,
            "qwen3_next": Qwen3NextForCausalLM,
            "mamba3": Mamba3ForCausalLM,
            "minimax_m27": MiniMaxM2ForCausalLM,
        }
    ```

4. Update the parameter enumeration.

    The optional values for `model_id` are defined in `ModelArguments` in `mindspeed_llm/fsdp2/utils/arguments.py`. gpt-oss must be included in the `Literal[...]` type. The current code example in MindSpeed LLM is as follows:

    ```python
    model_id: Optional[Literal[
        "gpt_oss",
        "qwen3",
        "qwen3_moe",
        "qwen3_next",
        "step35",
        "mamba3",
        "minimax_m27",
    ]] = field(default=None)
    ```

    MindSpeed LLM has already completed the registration and parameter enumeration configuration for gpt-oss.

## Configuring Task Scripts and YAML Files

MindSpeed LLM has already provided an FSDP2 pretraining example for gpt-oss:

```text
examples/fsdp2/gpt_oss/
├── pretrain_gpt_oss_20b_4k_fsdp2_A3.sh
└── pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml
```

### Modifying the Pretraining YAML File

Modify the model and data paths in the configuration file `examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml`:

The example below uses "Path 2: Custom Registry Adaptation". Therefore, `model.model_id: gpt_oss` must be configured. If "Path 1: Native Transformers Adaptation" is used, do not configure `model_id`. The framework will load the model automatically via `AutoModelForCausalLM` based on `config.json`.

```yaml
model:
  model_id: gpt_oss                               # Enable the gpt-oss custom implementation registered in ModelRegistry
  model_name_or_path: /home/data/gpt-oss-20b-hf/  # Hugging Face weight directory
  trust_remote_code: false                        # Usually set to false when a local model implementation already exists
  train_from_scratch: false                       # false means load existing weights
  tokenizer_name_or_path: null                    # Tokenizer path; null means model_name_or_path is used by default

data:
  dataset:
    file_name: /home/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet  # Raw parquet data path
  template: gpt                              # Data template; the gpt-oss example uses gpt
  cutoff_len: 4096                           # Maximum token length per sample
  max_samples: 100000                        # Maximum number of samples to read; can be reduced for debugging
  overwrite_cache: true                      # Whether to overwrite the data cache
  preprocessing_num_workers: 1               # Number of concurrent data preprocessing workers
  data_manager_type: mg                      # The pretraining example uses mg data management

parallel:
  fsdp_size: 16                              # FSDP sharding size, usually matching the number of training cards
  fsdp_modules:
    - model.layers.{*}                       # Transformer layers wrapped by FSDP per layer
    - model.embed_tokens                     # Embedding layer participates in FSDP
    - lm_head                                # Output head participates in FSDP
  tp_size: 1                                 # Tensor parallelism size
  ep_size: 4                                 # Expert parallelism size
  ep_modules:
    - model.layers.{*}.mlp.experts           # Expert module path for expert parallelism
  ep_fsdp_size: 4                            # FSDP sharding size within the expert module
  ep_fsdp_modules:
    - model.layers.{*}.mlp.experts           # FSDP wrapping path within the expert module
  ep_dispatcher: eager                       # Expert token dispatch method
  recompute: true                            # Whether to enable recomputation to save memory
  recompute_modules:
    - model.layers.{*}                       # Modules with recomputation enabled
  cp_size: 1                                 # Context parallelism size
  cp_type: ulysses                           # Context parallelism type

training:
  stage: pt                                  # Training stage; pt stands for pretraining
  per_device_train_batch_size: 1             # Per-device batch size
  gradient_accumulation_steps: 1             # Gradient accumulation steps
  dataloader_num_workers: 1                  # Number of dataloader workers
  disable_shuffling: 1                       # Whether to disable data shuffling
  seed: 42                                   # Random seed
  output_dir: ./output                       # Checkpoint and output directory
  optimizer: adamw                           # Optimizer type
  lr: 1.25e-06                               # Learning rate
  weight_decay: 0.0                          # Weight decay
  max_grad_norm: 1.0                         # Gradient clipping threshold
  lr_scheduler_type: cosine                  # Learning rate scheduler type
  max_steps: 2000                            # Maximum training steps
  save_steps: 500                            # Checkpoint saving interval
  logging_steps: 1                           # Log printing interval

optimization:
  use_fused_rmsnorm: true                    # Use fused RMSNorm
  moe_grouped_gemm: true                     # Use grouped GEMM expert computation path
  use_fused_rotary_pos_emb: true             # Use fused RoPE
```

Common parameter options:

| Configuration | Common Values | Description |
| --- | --- | --- |
| `model.model_id` | `gpt_oss`, `qwen3`, `qwen3_moe`, `qwen3_next`, `step35`, `mamba3`, `minimax_m27` | Specifies the custom registered model. Only needed for Path 2. Do not configure this item when using Path 1 Native Transformers Adaptation. |
| `model.model_name_or_path` | Local Hugging Face weight directory | Required. Points to the directory containing `config.json`, tokenizer, and weight files. |
| `model.trust_remote_code` | `true`/`false` | Set to `true` when the model depends on custom code on the Hub; usually `false` when using the built-in model implementation in MindSpeed LLM. |
| `model.train_from_scratch` | `true`/`false` | `true` means random initialization based on config. `false` means load existing weights. |
| `model.tokenizer_name_or_path` | `null` or tokenizer directory | `null` means `model_name_or_path` is used by default. |
| `data.dataset.file_name` | Local parquet, json, or jsonl path | Raw data path. The example uses Alpaca parquet. |
| `data.template` | `gpt`, and so on | Specifies the sample concatenation template. The gpt-oss example uses `gpt`. |
| `data.data_manager_type` | `mg`/`lf` | `mg` for Megatron-style pretraining data; `lf` for LlamaFactory-style SFT data. |
| `training.stage` | `pt`/`sft` | `pt` stands for pretraining. `sft` stands for supervised fine-tuning. |
| `parallel.fsdp_size` | Positive integer | FSDP sharding size, usually must match the actual number of training devices, `ep_size`, and other parallelism settings. |
| `parallel.fsdp_modules` | List of module paths | Modules wrapped by FSDP. Common patterns are `model.layers.{*}`, `model.embed_tokens`, `lm_head`. |
| `parallel.tp_size` | Positive integer | Tensor parallelism size. Set to `1` when TP is not used. |
| `parallel.ep_size` | Positive integer | Expert parallelism size, configured for MoE models based on the number of experts and devices. |
| `parallel.ep_modules` | List of module paths | Expert module paths for expert parallelism. For gpt-oss, it is `model.layers.{*}.mlp.experts`. |
| `parallel.ep_dispatcher` | `eager`/`fused`/`mc2` | Expert token dispatch method. `fused` triggers the fused expert path. |
| `parallel.recompute` | `true`/`false` | Whether to enable recomputation to save memory. |
| `parallel.cp_size` | Positive integer | Context parallelism size. Set to `1` when not enabled. |
| `parallel.cp_type` | `ulysses`/`ring` | Context parallelism type. |
| `training.optimizer` | `adamw`/`muon` | Optimizer type. |
| `training.lr_scheduler_type` | `cosine`/`linear`/`constant` | Learning rate scheduling strategy. |
| `optimization.moe_grouped_gemm` | `true`/`false` | When enabled, invokes the grouped GEMM path in `GptOssFusedExperts`. |
| `optimization.use_fused_rmsnorm` | `true`/`false` | Whether to use fused RMSNorm. |
| `optimization.use_fused_rotary_pos_emb` | `true`/`false` | Whether to use fused RoPE. |

### Modifying the Startup Script

Verify that the machine configuration in the pretraining script `examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.sh` matches the actual environment:

```bash
NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=1
NODE_RANK=0
```

For multi-node training, modify `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` according to the actual cluster.

You are advised to override parameters that vary significantly between machines or tasks, such as model weight paths, data paths, model parallelism, and debugging steps, in the shell script to avoid frequent modifications to the same YAML. Dot-separated parameters override the same-name fields in the YAML, for example:

```bash
DISTRIBUTED_ARGS="--nproc_per_node 16 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6499"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
  examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml \
  --model.model_name_or_path /home/data/gpt-oss-20b-hf/ \
  --data.dataset.file_name /home/data/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
  --parallel.fsdp_size 16 \
  --parallel.ep_size 4 \
  --parallel.ep_fsdp_size 4 \
  --training.output_dir ./output/gpt_oss_20b \
  --training.max_steps 2000 \
  | tee logs/pretrain_gpt_oss_20b_4k_${TIMESTAMP}.log
```

If using "Path 2: Custom Registry Adaptation", you can keep `model.model_id: gpt_oss` in the YAML, or add the parameter `--model.model_id gpt_oss` in the shell script. If using "Path 1: Native Transformers Adaptation", do not configure `model_id` in either the YAML or the shell script.

## Starting gpt-oss Training

The FSDP2 example script first loads the common environment variables:

```bash
source examples/fsdp2/env_config.sh
```

This sets:

```bash
export TRAINING_BACKEND=mindspeed_fsdp
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

If your Ascend environment requires manually loading the toolkit, execute the following before launching:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Start single-node 16-device pretraining:

```bash
bash examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.sh
```

The script is equivalent to executing:

```bash
torchrun \
  --nproc_per_node 16 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 6499 \
  train_fsdp2.py \
  examples/fsdp2/gpt_oss/pretrain_gpt_oss_20b_4k_fsdp2_A3.yaml
```

The logs will be written to `logs/pretrain_gpt_oss_20b_4k_<timestamp>.log`.

## Adaptation Checklist

After completing the gpt-oss adaptation, you are advised to check in the following order:

1. `GptOssForCausalLM` exists in `mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py`.
2. `mindspeed_llm/fsdp2/models/model_registry.py` has imported `GptOssForCausalLM` and added `"gpt_oss"` to `_REGISTRY`.
3. The `ModelArguments.model_id` enumeration in `mindspeed_llm/fsdp2/utils/arguments.py` includes `"gpt_oss"`.
4. When using "Path 2: Custom Registry Adaptation", explicitly set `model.model_id: gpt_oss` in the YAML. When using "Path 1: Native Transformers Adaptation", do not configure `model_id`.
5. The `model.model_name_or_path` in the YAML file points to the local Hugging Face weight directory.
6. For MoE models, correctly configure `parallel.ep_modules`, `parallel.ep_size`, and `parallel.ep_fsdp_modules`.
7. When using fused operators, confirm that the switches in `optimization` are consistent with the model code path. For example, `moe_grouped_gemm: true` triggers `GptOssFusedExperts`.
8. Before launching, first verify the pipeline with smaller `max_steps` and `max_samples`, then scale up the training.

## Frequently Asked Questions

- How to choose between native Transformers adaptation and custom registry adaptation?

  If you only need to verify the training pipeline and do not need to modify the gpt-oss model source code, use Native Transformers Adaptation. If you need NPU-fused operators, expert parallelism customization, or modification of the MoE computation logic, use Custom Registry Adaptation.

- Why is the model not found after setting `model_id: gpt_oss`?

  There are usually three reasons:

  1. `"gpt_oss"` is not registered in `mindspeed_llm/fsdp2/models/model_registry.py`.
  2. `"gpt_oss"` is not included in the `Literal[...]` in `mindspeed_llm/fsdp2/utils/arguments.py`, causing the parameter parsing stage to fail.
  3. The `model_id` in the YAML does not exactly match the key in `_REGISTRY`.

- How to handle missing key or unexpected key errors during weight loading?

  First check whether the parameter names in `mindspeed_llm/fsdp2/models/gpt_oss/modeling_gpt_oss.py` are consistent with the Hugging Face weights. When developing further from the native `modeling_gpt_oss.py`, you are advised to keep the module hierarchy and parameter names stable, and only replace the computation logic inside the modules that need optimization.

- How to configure the expert parallelism module for gpt-oss?

  The example YAML uses:

  ```yaml
  parallel:
    ep_modules:
      - model.layers.{*}.mlp.experts
    ep_fsdp_modules:
      - model.layers.{*}.mlp.experts
  ```

  This path must match the module naming in the model implementation. In the MLP of gpt-oss, the expert module is attached to `self.experts`. Therefore, the path is `model.layers.{*}.mlp.experts`.

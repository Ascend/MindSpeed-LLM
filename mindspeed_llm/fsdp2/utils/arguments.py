"""
Includes ModelArguments/DataArguments/ParallelArguments/TrainingArguments classes and parses the argument class using the command line inputs and yaml configuration.
"""
import argparse
from collections import defaultdict
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from inspect import isclass
import json
import os
import sys
import types
from typing import Optional, List, Union, Any, Callable, Dict, Literal, TypeVar, get_type_hints, get_origin, get_args
import yaml
from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Model-related parameters: path, initialization method, etc.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_id: Optional[Literal["gpt_oss", "qwen3", "qwen3_moe", "step35"]] = field(
        default=None,
        metadata={"help": "Model type. New model needs to be registered in the class ModelRegistry of mindspeed_llm/fsdp2/models/model_registry.py"}
    )
    init_model_with_meta_device: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize the model using the meta device."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files."}
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={
            "help": "If True, initialize the model from config (random weights) instead of loading pretrained weights."}
    )
    # Specify tokenizer path if different from model path
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    add_tokens: Optional[str] = field(
        default=None,
        metadata={
            "help": "Non-special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    add_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    new_special_tokens_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to YAML config with special token descriptions for semantic initialization. "
                "If set, this takes precedence over add_special_tokens. "
                "YAML format: {'<token>': 'description text', ...}"
            )
        },
    )
    init_special_tokens: Literal["noise_init", "desc_init", "desc_init_w_noise"] = field(
        default="noise_init",
        metadata={
            "help": (
                "Initialization method for new special tokens: "
                "'noise_init' (default, random noise around mean), "
                "'desc_init' (semantic initialization from descriptions), "
                "'desc_init_w_noise' (semantic + random noise). "
                "Note: 'desc_init' methods require new_special_tokens_config."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    om_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Modelers Hub."},
    )
    quant_recipe: Literal["mxfp8"] = field(
        default=None,
        metadata={"help": "Quantization strategy"},
    )
    quant_format: Literal["E4M3", "E5M2", "HIF8"] = field(
        default=None,
        metadata={"help": "FP8 format: 'E4M3', 'E5M2', or 'HIF8'."},
    )
    quant_block_size: int = field(
        default=32,
        metadata={"help": "Block size for MXFP8 quantization, default 32."},
    )
    quant_apply_modules: List[str] = field(
        default_factory=lambda: ['model.layers.{*}'],
        metadata={
            "help":
                "List of model module patterns to apply MXFP8 quantization."
                "Example: 'model.layers.{*}'applies quantization to all transformer layers."},
    )
    quant_ignored_modules: List[str] = field(
        default_factory=lambda: ['lm_head'],
        metadata={"help": "List of module patterns to exclude from MXFP8 quantization. "},
    )
    converters: List[str] = field(
        default_factory=lambda: ["quantize.linear.mx"],
        metadata={
            "help":
                "This field specifies the quantization converters to use. "
                "It's a list of strings where each string represents a specific quantization implementation."
                "Default uses 'quantize.linear.mx' for mxfp8 quantization."},
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("`model_name_or_path` must be specified.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset: Optional[Union[Dict[str, Any], str]] = field(
        default=None,
        metadata={"help": "Train dataset: config dict or comma-separated dataset names."}
    )
    eval_dataset: Optional[Union[Dict[str, Any], str]] = field(
        default=None,
        metadata={"help": "Eval dataset: config dict or comma-separated eval dataset names."}
    )
    dataset_dir: str = field(
        default="./configs/FSDP2/data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the validation set, should be an integer or a float in range `[0,1)`."},
    )
    eval_on_each_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to evaluate on each dataset separately."},
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples."},
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    enable_thinking: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={"help": "Whether or not to use a shared file system for the datasets."},
    )

    data_manager_type: Literal["lf", "mg"] = field(
        default="lf",
        metadata={"help": "Data Manager type for building the different data manager"},
    )
    #megatron dataset args
    split: str = field(
        default="100,0,0",
        metadata={"help": "Comma-separated list of proportions for training, validation, and test split."}
    )
    create_attention_mask_in_dataloader: Optional[bool] = field(
        default=False,
        metadata={"help": "If set, do create attention_masks in dataloader."}
    )     
    no_shared_storage: Optional[bool] = field(
        default=False,
        metadata={"help": "if no shared storage, set it."}
    )
    dataloader_type: Literal["single", "cyclic", "external"] = field(
        default="single",
        metadata={
            "help": ("Single pass vs multiple pass data loader")
        },
    )
    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        if isinstance(self.dataset, dict) and not self.dataset:
            self.dataset = None
        if isinstance(self.eval_dataset, dict) and not self.eval_dataset:
            self.eval_dataset = None

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError(f"val_size={self.val_size} but dataset=None (dataset must be specified when val_size>0).")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError(f"val_size={self.val_size} and eval_dataset={self.eval_dataset} cannot be set together.")

        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError(f"interleave_probs={self.interleave_probs} is not supported for mix_strategy={self.mix_strategy}.")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
                raise ValueError(f"len(dataset)={len(self.dataset)} != len(interleave_probs)={len(self.interleave_probs)}.")

            if self.eval_dataset is not None and len(self.eval_dataset) != len(self.interleave_probs):
                raise ValueError(f"len(eval_dataset)={len(self.eval_dataset)} != len(interleave_probs)={len(self.interleave_probs)}.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError(f"val_size={self.val_size} must be integer when streaming=True.")

        if self.streaming and self.max_samples is not None:
            raise ValueError(f"streaming=True and max_samples={self.max_samples} are incompatible.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError(f"mask_history={self.mask_history} and train_on_prompt={self.train_on_prompt} cannot be True together.")

        if self.neat_packing:
            self.packing = True

        if self.packing:
            self.cutoff_len -= 1  # avoid pad_to_multiple_of, needs improve

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParallelArguments:
    """
    MindSpeed FSDP backend parallel strategy parameters (FSDP2, TP, EP)
    """
    tp_size: int = field(
        default=1,
        metadata={"help": "Tensor Parallel size. (Cols/Rows splitting)"}
    )
    fsdp_size: int = field(
        default=1,
        metadata={"help": "Fully Sharded Data Parallel size. (Sharding parameters)"}
    )
    recompute: bool = field(
        default=False,
        metadata={"help": "Whether to enable Gradient Checkpointing (Activation Recomputation)."}
    )
    # Expert Parallel (MoE)
    ep_size: int = field(
        default=1,
        metadata={"help": "Expert Parallel size for MoE models."}
    )
    ep_fsdp_size: int = field(
        default=1,
        metadata={"help": "FSDP size inside Expert Parallel groups."}
    )
    data_parallel_mode: Literal["ddp", "fsdp1", "fsdp2"] = field(
        default="ddp",
        metadata={"help": "Data parallel mode."},
    )
    cp_size: int = field(
        default=1,
        metadata={"help": "context parallel size."}
    )

    cp_type: Literal["ulysses"] = field(
        default="ulysses",
        metadata={"help": "Use context parallel algo."},
    )
    fsdp_modules: List[str] = field(
        default_factory=lambda:['model.layers.{*}', 'model.embed_tokens', 'lm_head'],
        metadata={"help": "Model structure of layers with Fully Sharded Data Parallel."},
    )
    ignored_modules: List[str] = field(
        default=None,
        metadata={"help": "Model structure of layers with not Fully Sharded Data Parallel."},
    )
    reshard_after_forward: bool = field(
        default=True,
        metadata={"help": "Whether to reshard parameters after forward pass (for main FSDP module)"},
    )
    shard_placement_fn: Optional[str] = field(
        default=None,
        metadata={"help": "Custom shard placement function for main FSDP module"}
    )
    tp_colwise: List[str] = field(
        default_factory=lambda:['*.q_proj', '*.k_proj', '*.v_proj', '*.gate_proj', '*.up_proj'],
        metadata={"help": "Model structure of layers with Tensor Parallel(Cols splitting)."},
    )
    tp_rowwise: List[str] = field(
        default_factory=lambda:['*.o_proj', '*.down_proj'],
        metadata={"help": "Model structure of layers with Tensor Parallel(Rows splitting)."},
    )
    ep_modules: List[str] = field(
        default_factory=lambda:['model.layers.{*}.mlp.experts'],
        metadata={"help": "Model structure of layers with Expert Parallel."},
    )
    ep_fsdp_modules: List[str] = field(
        default_factory=lambda:['model.layers.{*}.mlp.experts'],
        metadata={"help": "Model structure of layers with FSDP inside Expert Parallel groups."},
    )
    ep_dispatcher: Literal["eager", "fused", "mc2"] = field(
        default="eager",
        metadata={
            "help": "Dispatcher strategy for Expert Parallel (MoE). "
                    "Options: 'eager' (immediate token dispatch to experts, default), "
                    "'fused' (fused routing & expert computation for higher throughput), "
                    "'mc2' (mixed compression dispatch to reduce cross-card communication cost). Defaults to 'eager'."
        },
    )
    recompute_modules: List[str] = field(
        default_factory=lambda:['model.layers.{*}'],
        metadata={"help": "Model structure of layers with Gradient Checkpointing (Activation Recomputation)."},
    )
    param_dtype: Literal["bf16", "fp16", "fp32"] = field(
        default="bf16",
        metadata={"help": "Data type for FSDP parameter storage. Defaults to 'bf16'"}
    )
    reduce_dtype: Literal["bf16", "fp16", "fp32"] = field(
        default="fp32",
        metadata={
            "help": "Data type for FSDP gradient reduction . Using 'fp32' ensures numerical stability. Defaults to 'fp32'."}
    )
    num_to_forward_prefetch: int = field(
        default=1,
        metadata={
            "help": "Number of modules to prefetch during FSDP forward pass (optimizes pipeline efficiency). Defaults to 1."}
    )
    num_to_backward_prefetch: int = field(
        default=1,
        metadata={
            "help": "Number of modules to prefetch during FSDP backward pass (optimizes pipeline efficiency). Defaults to 1."}
    )

    def __post_init__(self):
        if self.fsdp_modules is None:
            raise ValueError(
                "Parameter 'fsdp_modules' cannot be None! Please provide a non-empty list of module paths (e.g. ['model.layers.{*}'])."
            )
        if len(self.fsdp_modules) == 0:
            raise ValueError(
                "Parameter 'fsdp_modules' cannot be an empty list! Please provide at least one module path (e.g. ['model.layers.{*}'])."
            )


@dataclass
class TrainingArguments:
    """
    Training hyperparameters: corresponding to requirements of Trainer and Optimizer/Scheduler Factory
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    # --- Optimization ---
    optimizer: Literal["adamw"] = field(
        default="adamw",
        metadata={"help": "Optimizer. Default to adamw."},
    )
    lr: float = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.95,
        metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for AdamW optimizer."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping."}
    )

    # --- Scheduling ---
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. (cosine, linear, constant)"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    min_lr: float = field(
        default=1e-6,
        metadata={"help": "Minimum learning rate for cosine scheduler."}
    )

    # --- Training Loop Control ---
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Overrides num_train_epochs."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )

    # --- IO & Logging ---
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every X updates steps."}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={"help": "The list of integrations to report the results and logs to (e.g. 'wandb', 'tensorboard')."}
    )
    stage: Literal["pt", "sft"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    #megatron train args
    calculate_per_token_loss: bool = field(
        default=False,
        metadata={"help": "Scale cross entropy loss by the number of non-padded tokens in the global batch, versus the default behavior of assuming all tokens are non-padded"}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )

    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of batches loaded in advance by each worker. "
                "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
            )
        },
    )

    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )

    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        },
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per device accelerator core/CPU for training."}
    )
    use_fused_rmsnorm: bool = field(
        default=False,
        metadata={"help": "Use fused rmsnorm."}
    )
    moe_grouped_gemm: bool = field(
        default=False,
        metadata={"help": "When there are multiple experts per rank, launch multiple local GEMM kernels in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine."}
    )
    use_fused_rotary_pos_emb: bool = field(
        default=False,
        metadata={"help": "Use fused rotary-pos-emb."}
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "use FlashAttention implementation of attention."}
    )
    save_only_model: bool = field(
        default=False, metadata={"help": "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."}
    )
    save_async: bool = field(
        default=False, metadata={"help": "Whether to save checkpoint asynchronously."},
    )
    save_epochs: int = field(
        default=1, metadata={"help": "Number of epochs between two checkpoint saves."},
    )
    save_hf_weights: bool = field(
        default=True, metadata={"help": "Save the huggingface format weights to the last checkpoint dir."},
    )
    # --- Profiling (NPU) ---
    profile: bool = field(
        default=False,
        metadata={"help": "Enable NPU profiling using torch_npu.profiler."}
    )
    profile_step_start: int = field(
        default=0,
        metadata={"help": "Start profiling at this global step (inclusive)."}
    )
    profile_step_end: int = field(
        default=-1,
        metadata={"help": "Stop profiling before this global step (exclusive). If -1, profile until end."}
    )
    profile_ranks: List[int] = field(
        default_factory=lambda: [-1],
        metadata={"help": "List of ranks to enable profiling on. Use [-1] to profile all ranks."}
    )
    profile_level: str = field(
        default="level0",
        metadata={"help": "Profiling level: 'level_none', 'level0', 'level1', 'level2'."}
    )
    profile_export_type: str = field(
        default="text",
        metadata={"help": "Export type: 'text' or 'db'."}
    )
    profile_data_simplification: bool = field(
        default=False,
        metadata={"help": "Use data simplification mode in profiler."}
    )
    profile_with_cpu: bool = field(
        default=False,
        metadata={"help": "Record CPU activities in profiler."}
    )
    profile_with_stack: bool = field(
        default=False,
        metadata={"help": "Record call stack in profiler."}
    )
    profile_with_memory: bool = field(
        default=False,
        metadata={"help": "Profile memory allocation and usage."}
    )
    profile_record_shapes: bool = field(
        default=False,
        metadata={"help": "Record tensor shapes in profiler."}
    )
    profile_save_path: str = field(
        default="./profile",
        metadata={"help": "Directory to save profiling traces (TensorBoard format)."}
    )
    def __post_init__(self):  # Path parameter validation
        if self.output_dir is None:
            raise ValueError("`output_dir` must be specified.")
        if self.profile:
            if self.profile_step_start < 0:
                raise ValueError("`profile_step_start` must be >= 0")
            if self.profile_step_end != -1 and self.profile_step_end <= self.profile_step_start:
                raise ValueError("`profile_step_end` must be > profile_step_start or -1")


def _string_to_bool(value: Union[bool, str]) -> bool:
    """Convert string to boolean value"""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"Truthy value expected: got {value} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
    )


def _convert_str_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert string values in dictionary"""
    for key, value in input_dict.items():
        if isinstance(value, dict):
            input_dict[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            if value.lower() in ("true", "false"):  # check for bool
                input_dict[key] = value.lower() == "true"
            elif value.isdigit():  # check for digit
                input_dict[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                input_dict[key] = float(value)

    return input_dict


def _make_choice_type_function(choices: List[Any]) -> Callable[[str], Any]:
    """Build mapping from string to choices"""
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def fsdp2_parse_args(rootclass: TypeVar) -> TypeVar:
    """
    Parses the root argument class from CLI or YAML input.
    Supports complex types like List[Dict] and Dict via JSON serialization.
    """
    # 1: Create ArgumentParser and register all fields
    parser = _create_argument_parser(rootclass)

    # 2: Parse command-line args
    cmd_args = sys.argv[1:]
    if not cmd_args:
        raise ValueError("No arguments provided.")

    # 3: Load YAML config if the first arg is a .yaml/.yml file
    yaml_config = _load_yaml_if_provided(cmd_args)

    # 4: Merge YAML into CLI args (CLI has higher priority)
    final_cmd_args = _merge_yaml_into_cmd_args(cmd_args, yaml_config)

    # 5: Parse arguments
    args, remaining_args = parser.parse_known_args(final_cmd_args)
    if remaining_args:
        logger.info_rank0(f"Some specified arguments are not used by the ArgumentParser: {remaining_args}")

    # 6: Post-process fields that require JSON deserialization
    parse_result = _postprocess_json_fields(args, parser.dict_fields)

    # 7: Build dataclass instances
    return _build_dataclass_instances(rootclass, parse_result)


# =============================================================================
#  Create ArgumentParser from dataclass structure
# =============================================================================
def _create_argument_parser(rootclass):
    """Dynamically creates an ArgumentParser based on the dataclass schema."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.dict_fields = set()  # Tracks fields needing JSON deserialization

    for subclass in fields(rootclass):
        base = subclass.name
        subclass_type = subclass.default_factory

        try:
            type_hints = get_type_hints(subclass_type)
        except Exception as e:
            raise RuntimeError(f"Type resolution failed for {subclass_type}: {e}") from e

        for attr in fields(subclass_type):
            if not attr.init:
                continue

            attr_name = attr.name
            attr_type = type_hints[attr_name]
            origin_type = get_origin(attr_type)
            parser_kwargs = attr.metadata.copy()

            # Resolve Optional[T] → T
            effective_type, effective_origin = _resolve_optional_type(attr_type, origin_type)

            # Dispatch by type
            if effective_origin is Union or (hasattr(types, "UnionType") and isinstance(effective_origin, types.UnionType)):
                # For Union[Dict, str], we treat it as a dict field (since str is simple)
                # But actually, we'll handle it in post-processing
                _handle_dict(parser_kwargs, base, attr_name, attr, parser)
                parser.add_argument(f"--{base}.{attr_name}", **parser_kwargs)
                continue
            if effective_origin is Literal or (isinstance(effective_type, type) and issubclass(effective_type, Enum)):
                _handle_literal_or_enum(parser_kwargs, effective_type, effective_origin, attr)

            elif effective_type is bool or (effective_origin is Union and bool in get_args(effective_type)):
                _handle_bool(parser_kwargs, attr)

            elif effective_origin is list:
                item_type = get_args(effective_type)[0]
                item_origin = get_origin(item_type)
                if item_origin is dict or item_type is dict:
                    _handle_list_of_dict(parser_kwargs, base, attr_name, attr, parser)
                else:
                    _handle_list_of_scalar(parser_kwargs, item_type, attr)

            elif effective_origin is dict or effective_type is dict:
                _handle_dict(parser_kwargs, base, attr_name, attr, parser)

            else:
                _handle_scalar(parser_kwargs, effective_type, attr)

            parser.add_argument(f"--{base}.{attr_name}", **parser_kwargs)

    return parser


def _resolve_optional_type(attr_type, origin_type):
    """Extract inner type(s) from Union types, handling Optional[Union[A, B]]."""
    if origin_type is Union or (hasattr(types, "UnionType") and isinstance(origin_type, types.UnionType)):
        args = get_args(attr_type)
        non_none_types = [t for t in args if t is not type(None)]
        
        if len(non_none_types) == 0:
            raise RuntimeError(f"Union type contains only None: {attr_type}")
        elif len(non_none_types) == 1:
            # Standard Optional[T]
            effective_type = non_none_types[0]
            effective_origin = get_origin(effective_type)
            return effective_type, effective_origin
        else:
            # Multi-type Union like Union[Dict, str]
            # Return the original union as effective_type
            return attr_type, origin_type
    return attr_type, origin_type


def _handle_literal_or_enum(parser_kwargs, effective_type, effective_origin, attr):
    """Configure parser for Literal or Enum types."""
    choices = effective_type.__args__ if effective_origin is Literal else [x.value for x in effective_type]
    parser_kwargs["choices"] = choices
    parser_kwargs["type"] = _make_choice_type_function(choices)
    if attr.default is not MISSING:
        parser_kwargs["default"] = attr.default
    else:
        parser_kwargs["required"] = True


def _handle_bool(parser_kwargs, attr):
    """Configure parser for bool or Optional[bool]."""
    parser_kwargs["type"] = _string_to_bool
    if attr.default is not MISSING and attr.default is not None:
        parser_kwargs["default"] = attr.default
        parser_kwargs["nargs"] = "?"
        parser_kwargs["const"] = True
    else:
        parser_kwargs["default"] = False
        parser_kwargs["nargs"] = "?"


def _handle_list_of_dict(parser_kwargs, base, attr_name, attr, parser):
    """Handle List[Dict]: each item is passed as a JSON string."""
    parser_kwargs["type"] = str
    parser_kwargs["nargs"] = "+"
    parser.dict_fields.add(f"{base}.{attr_name}")
    if attr.default is not MISSING:
        parser_kwargs["default"] = [] if attr.default is None else attr.default
    elif attr.default_factory is not MISSING:
        parser_kwargs["default"] = attr.default_factory()
    else:
        parser_kwargs["required"] = True


def _handle_list_of_scalar(parser_kwargs, item_type, attr):
    """Handle List[str], List[int], etc."""
    parser_kwargs["type"] = item_type
    parser_kwargs["nargs"] = "+"
    if attr.default_factory is not MISSING:
        parser_kwargs["default"] = attr.default_factory()
    elif attr.default is not MISSING:
        parser_kwargs["default"] = attr.default
    else:
        parser_kwargs["required"] = True


def _handle_dict(parser_kwargs, base, attr_name, attr, parser):
    """Handle Dict: passed as a single JSON string."""
    parser_kwargs["type"] = str
    parser_kwargs["nargs"] = None
    parser.dict_fields.add(f"{base}.{attr_name}")
    if attr.default_factory is not MISSING:
        parser_kwargs["default"] = str(attr.default_factory())
    elif attr.default is not MISSING:
        parser_kwargs["default"] = str(attr.default) if attr.default is not None else "{}"
    else:
        parser_kwargs["required"] = True


def _handle_scalar(parser_kwargs, effective_type, attr):
    """Handle scalar types: int, float, str, etc."""
    parser_kwargs["type"] = effective_type
    if attr.default is not MISSING:
        parser_kwargs["default"] = attr.default
    elif attr.default_factory is not MISSING:
        parser_kwargs["default"] = attr.default_factory()
    else:
        parser_kwargs["required"] = True


# =============================================================================
#  Load YAML config
# =============================================================================
def _load_yaml_if_provided(cmd_args):
    """Load YAML config if the first argument is a .yaml or .yml file."""
    if cmd_args and (cmd_args[0].endswith(".yaml") or cmd_args[0].endswith(".yml")):
        input_path = cmd_args[0]
        with open(os.path.abspath(input_path), encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# =============================================================================
#  Merge YAML into CLI args
# =============================================================================
def _merge_yaml_into_cmd_args(cmd_args, yaml_config):
    """Convert YAML config to CLI-style arguments (lower priority than explicit CLI)."""
    working_args = cmd_args[1:] if yaml_config else cmd_args[:]
    cmd_args_string = "=".join(working_args)
    result = working_args[:]

    for base, arg_dict in yaml_config.items():
        if not isinstance(arg_dict, dict):
            continue
        for arg_name, arg_value in arg_dict.items():
            if arg_value is None:
                continue
            if f"--{base}.{arg_name}=" in cmd_args_string:
                continue  # Skip if overridden by CLI

            result.append(f"--{base}.{arg_name}")
            if isinstance(arg_value, list):
                result.extend(str(item) for item in arg_value)
            elif isinstance(arg_value, dict):
                result.append(json.dumps(arg_value, ensure_ascii=False))
            else:
                result.append(str(arg_value))

    return result


# =============================================================================
#  Post-process JSON fields
# =============================================================================
def _postprocess_json_fields(args, dict_fields):
    """Deserialize JSON strings back to dict/list for marked fields."""
    parse_result = defaultdict(dict)
    for key, value in vars(args).items():
        base, name = key.split(".", maxsplit=1)
        full_key = f"{base}.{name}"

        if full_key in dict_fields:
            if value is None:
                parsed_value = None
            elif isinstance(value, list):
                try:
                    parsed_value = [json.loads(item) for item in value]
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse list of dict for {full_key}: {e}") from e
            elif isinstance(value, str):
                stripped = value.strip()
                if stripped == "":
                    parsed_value = {} 
                elif stripped.startswith("{") and stripped.endswith("}"):
                    try:
                        parsed_value = json.loads(stripped)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON for dict field {full_key}: {value}, error: {e}") from e
                else:
                    parsed_value = value
            else:
                raise ValueError(f"Unexpected type for dict field {full_key}: {type(value)}")
            parse_result[base][name] = parsed_value
        else:
            parse_result[base][name] = value

    return parse_result


# =============================================================================
#  Build dataclass instances
# =============================================================================
def _build_dataclass_instances(rootclass, parse_result):
    """Construct dataclass instances from parsed arguments."""
    data_classes = {}
    for subclass in fields(rootclass):
        base = subclass.name
        subclass_type = subclass.default_factory
        data_classes[base] = subclass_type(**parse_result.get(base, {}))

    return rootclass(**data_classes)
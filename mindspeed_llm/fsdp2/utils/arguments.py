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
from typing import Optional, List, Union, Any, Callable, Dict, Literal, TypeVar, get_type_hints
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

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("`model_name_or_path` must be specified.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses to use for data loading."}
    )
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data",
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

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

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

    # Reserved for ParallelArguments Parameter Validation
    def __post_init__(self):
        pass


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

    def __post_init__(self):  # Path parameter validation
        if self.output_dir is None:
            raise ValueError("`output_dir` must be specified.")


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
    Parses the root argument class using the CLI inputs or yaml inputs. For example:

    Input YAML file content:
        model:
          model_path: /path/llm
        parallel:
          fsdp_size: 8
          recompute: true
        train:
          lr: 1e-6

    This function converts the YAML parameters into CLI argument format as follows:
       --model.model_path /path/llm --parallel.fsdp_size 8 --parallel.recompute true --train.lr 1e-6

    After parsing, return the instantiated rootclass. Access parameters via:
       rootclass.model.model_path (returns /path/llm), rootclass.train.lr (returns 1e-6).
    """
    # ======================== Step 1: Initialize Parser & Sub-dataclass Mapping ========================
    # Initialize command line argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    base_to_subclass = {}
    dict_fields = set()
    # Generate parsing rules, iterate through fields of root dataclass
    for subclass in fields(rootclass):
        base = subclass.name  # Sub-dataclass name: model/data/parallel/train
        base_to_subclass[base] = subclass.default_factory  # Sub-dataclass type: ModelArguments/DataArguments/ParallelArguments/TrainingArguments
        try:
            type_hints: Dict[str, type] = get_type_hints(subclass.default_factory)
        except Exception as e:
            raise RuntimeError(f"Type resolution failed for {subclass.default_factory}.") from e

        # ======================== Step 2: Generate CLI Parsing Rules for Sub-dataclass Fields ========================
        for attr in fields(subclass.default_factory):  # Iterate through fields of sub-dataclass, add CLI arguments for each field
            if not attr.init:
                continue
            # Get field type
            attr_type = type_hints[attr.name]
            origin_type = getattr(attr_type, "__origin__", attr_type)
            if isinstance(attr_type, str):
                raise RuntimeError(f"Cannot resolve string-based type annotation for field '{attr.name}' in sub-dataclass '{base}'. The problematic type annotation is: {attr.type}.")
            # Handle Optional type
            if origin_type is Union or (hasattr(types, "UnionType") and isinstance(origin_type, types.UnionType)):
                if len(attr_type.__args__) != 2 or type(None) not in attr_type.__args__:  # only allows Optional[X]
                    raise RuntimeError(f"Invalid Union type for field '{attr.name}' in sub-dataclass '{base}'. Only Optional[X] (Union[X, None]) is supported.")

                if bool not in attr_type.__args__:  # except for `Union[bool, NoneType]`
                    attr_type = (
                        attr_type.__args__[0] if isinstance(None, attr_type.__args__[1]) else attr_type.__args__[1]
                    )
                    origin_type = getattr(attr_type, "__origin__", attr_type)

            parser_kwargs = attr.metadata.copy()
            # Handle Literal/Enum type
            if origin_type is Literal or (isinstance(attr_type, type) and issubclass(attr_type, Enum)):
                if origin_type is Literal:
                    parser_kwargs["choices"] = attr_type.__args__
                else:
                    parser_kwargs["choices"] = [x.value for x in attr_type]

                parser_kwargs["type"] = _make_choice_type_function(parser_kwargs["choices"])

                if attr.default is not MISSING:
                    parser_kwargs["default"] = attr.default
                else:
                    parser_kwargs["required"] = True

            # Handle Bool type
            elif attr_type is bool or attr_type == Optional[bool]:
                parser_kwargs["type"] = _string_to_bool
                if attr_type is bool or (attr.default is not None and attr.default is not MISSING):
                    parser_kwargs["default"] = False if attr.default is MISSING else attr.default
                    parser_kwargs["nargs"] = "?"
                    parser_kwargs["const"] = True

            # Handle list type
            elif isclass(origin_type) and issubclass(origin_type, list):
                parser_kwargs["type"] = attr_type.__args__[0]
                parser_kwargs["nargs"] = "+"
                if attr.default_factory is not MISSING:
                    parser_kwargs["default"] = attr.default_factory()
                elif attr.default is MISSING:
                    parser_kwargs["required"] = True

            # Handle Dict type
            elif isclass(origin_type) and issubclass(origin_type, dict):
                parser_kwargs["type"] = str  # parse dict inputs with json string
                dict_fields.add(f"{base}.{attr.name}")
                if attr.default_factory is not MISSING:
                    parser_kwargs["default"] = str(attr.default_factory())
                elif attr.default is MISSING:
                    parser_kwargs["required"] = True

            else:
                parser_kwargs["type"] = attr_type
                if attr.default is not MISSING:
                    parser_kwargs["default"] = attr.default
                elif attr.default_factory is not MISSING:
                    parser_kwargs["default"] = attr.default_factory()
                else:
                    parser_kwargs["required"] = True
            # Generate CLI arguments in format --model.model_path
            parser.add_argument(f"--{base}.{attr.name}", **parser_kwargs)

    # ======================== Step 3: Merge YAML & CLI Arguments and Parse ========================
    cmd_args = sys.argv[1:]
    cmd_args_string = "=".join(cmd_args)  # use `=` to mark the end of arg name
    input_data = {}
    # Identify and load yaml file
    if cmd_args[0].endswith(".yaml") or cmd_args[0].endswith(".yml"):
        input_path = cmd_args.pop(0)
        with open(os.path.abspath(input_path), encoding="utf-8") as f:
            input_data: Dict[str, Dict[str, Any]] = yaml.safe_load(f)

    # Convert yaml parameters to CLI argument format, lower priority than explicit CLI arguments
    for base, arg_dict in input_data.items():  # Iterate through base classes in yaml
        for arg_name, arg_value in arg_dict.items():  # Iterate through each parameter under base class
            if f"--{base}.{arg_name}=" not in cmd_args_string:  # Check if parameter is explicitly specified in CLI, skip if specified
                cmd_args.append(f"--{base}.{arg_name}")
                # Process according to parameter value type
                if isinstance(arg_value, str):
                    cmd_args.append(arg_value)
                elif isinstance(arg_value, list):
                    cmd_args.extend(str(x) for x in arg_value)
                else:
                    cmd_args.append(json.dumps(arg_value))
    # Parser parses merged CLI arguments
    args, unknown_args = parser.parse_known_args(cmd_args)

    if unknown_args:
        logger.warn_rank0(f"Some specified arguments are not used by the ArgumentParser: {unknown_args}")  # unknown_args empty means all parameters are parsed

    # ======================== Step 4: Convert Parsed Results & Instantiate Dataclasses ========================
    # Split parameters
    parse_result = defaultdict(dict)
    for key, value in vars(args).items():
        if key in dict_fields:
            if isinstance(value, str) and value.startswith("{"):
                value = _convert_str_dict(json.loads(value))
            else:
                raise ValueError(f"Expect a json string for dict argument, but got {value}")

        base, name = key.split(".", maxsplit=1)  # model.config_path
        parse_result[base][name] = value

    # instantiate sub-dataclasses
    data_classes = {}
    for base, subclass_type in base_to_subclass.items():
        data_classes[base] = subclass_type(**parse_result.get(base, {}))

    # instantiate root class
    return rootclass(**data_classes)

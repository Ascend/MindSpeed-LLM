import os
import torch
import torch.distributed as dist
from typing import Any, Type
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)

# ==============================================================================
# [Mcore Imports] Dependencies for the Old Scheme
# ==============================================================================
try:
    from mindspeed_llm.fsdp2.models.fsdp2_model import FSDP2Model
    from mindspeed_llm.fsdp2 import ModelRegistry
except ImportError:
    # Graceful fallback if mcore dependencies are missing in a pure MindSpeed FSDP environment
    pass

# ==============================================================================
# Dependencies for the New Scheme (MindSpeed FSDP)
# ==============================================================================
try:
    from mindspeed.fsdp.mindspeed_parallel_engine import MindSpeedParallelEngine
    from mindspeed.fsdp.parallel_engine_config import (
        ParallelEngineConfig,
        FSDPPlanConfig,
        TPPlanConfig,
        EPPlanConfig
    )
except ImportError:
    pass


# ==============================================================================
# ModelFactory (New Scheme)
# ==============================================================================
class ModelFactory:
    """
    Responsible for building HuggingFace native models and wrapping them 
    as MindSpeed FSDP instances based on parallelization arguments.
    """

    @staticmethod
    def create(model_args, parallel_args):
        """
        Creates a MindSpeed FSDP wrapped model.
        
        Args:
            model_args: Contains model_name_or_path, trust_remote_code, train_from_scratch, etc.
            parallel_args: Contains tp_size, fsdp_size, recompute, ep_size, etc.
        """
        # 1. Setup Device
        # Ensure NPU is being used
        if torch.npu.is_available():
            # Respect LOCAL_RANK if set, otherwise default to 0
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"npu:{local_rank}")
            torch.npu.set_device(device)
        else:
            device = torch.device("cpu")

        # 2. Load HF Config
        logger.info_rank0(f"> Loading AutoConfig from {model_args.model_name_or_path}...")
        hf_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )

        # 3. Load HF Model
        # Decide loading method based on whether training from scratch or fine-tuning.
        # Typically: SFT uses `from_pretrained`, Pretrain (from scratch) uses `from_config`.
        train_from_scratch = getattr(model_args, 'train_from_scratch', False)

        if train_from_scratch:
            logger.info_rank0(f"> Initializing model from config (Random Weights) for Pretraining...")
            model = AutoModelForCausalLM.from_config(
                hf_config,
                trust_remote_code=True,
                torch_dtype=torch.float32 # Use FP32 for mixed precision training
            )
        else:
            logger.info_rank0(f"> Loading pretrained weights from {model_args.model_name_or_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=hf_config,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Use FP32 for mixed precision training
                low_cpu_mem_usage=True,  # Reduce memory peak usage during loading
                device_map="cpu"         # Load to CPU first; MindSpeed FSDP handles sharding and moving
            )

        # 4. Build MindSpeed FSDP Configuration
        # Dynamically calculate Data Parallel (DP) Size
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Guard against division by zero if args are not set correctly
        tp_size = getattr(parallel_args, 'tp_size', 1)
        fsdp_size = getattr(parallel_args, 'fsdp_size', 1)
        dp_size = world_size // (tp_size * fsdp_size)

        parallel_config = ModelFactory._build_parallel_config(parallel_args, dp_size)

        # 5. Wrap & Move
        logger.info_rank0(f"> Wrapping model with MindSpeed FSDP (TP={tp_size}, FSDP={fsdp_size})...")

        # MindSpeed FSDP will shard and wrap the CPU model based on the config.
        # The wrapped model automatically handles forward/backward communication.
        model = MindSpeedParallelEngine(config=parallel_config, model=model)

        # Finally move the model to NPU
        model = model.to(device)

        return model

    @staticmethod
    def _build_parallel_config(parallel_args, dp_size) -> 'ParallelEngineConfig':
        """
        Builds the Config based on parallel_args and hardcoded layer name rules.
        Note: The wildcards here (e.g., 'model.layers.{*}') are suitable for standard structures like Llama/Qwen.
        If using other non-standard models, these strings might need adjustment.
        """
        # --- 1. FSDP Plan ---
        # Requirement: Apply FSDP to transformer layers
        fsdp_plan = FSDPPlanConfig(
            ignored_modules=[],
            apply_modules= {
                'model.layers.{*}': {'reshard_after_forward': True, 'shard_placement_fn': None},
                'model.embed_tokens': {'reshard_after_forward': True},
                'lm_head': {'reshard_after_forward': True},
            },
            param_dtype='bf16',
            reduce_dtype='fp32',
            num_to_forward_prefetch=1,
            num_to_backward_prefetch=1
        )

        # --- 2. Tensor Parallel Plan ---
        # Requirement: Column Parallel for Q/K/V/Gate/Up, Row Parallel for O/Down
        tp_plan = TPPlanConfig(
            colwise_parallel=['*.q_proj', '*.k_proj', '*.v_proj', '*.gate_proj', '*.up_proj'],
            rowwise_parallel=['*.o_proj', '*.down_proj']
        )

        # --- 3. Expert Parallel Plan ---
        # For Mixture-of-Experts (MoE) models
        ep_size = getattr(parallel_args, 'ep_size', 1)
        ep_fsdp_size = getattr(parallel_args, 'ep_fsdp_size', 1)

        ep_plan = EPPlanConfig(
            apply_modules=['model.layers.{*}.mlp.experts'],
            apply_efsdp_modules=['model.layers.{*}.mlp.experts'],
            dispatcher='eager',
        )

        # --- 4. Recompute Plan ---
        # Activation Checkpointing
        recompute_plan = ['model.layers.{*}'] if parallel_args.recompute else []

        # --- 5. Assemble Config ---
        # Get parallel sizes safely
        tp_size = getattr(parallel_args, 'tp_size', 1)
        fsdp_size = getattr(parallel_args, 'fsdp_size', 1)

        config = ParallelEngineConfig(
            # Parallelism parameters
            data_parallel_size=dp_size,

            fully_shard_parallel_size=fsdp_size,
            fsdp_plan=fsdp_plan,

            tensor_parallel_size=tp_size,
            tp_plan=tp_plan,

            # Expert Parallelism
            expert_parallel_size=ep_size,
            expert_fully_shard_parallel_size=ep_fsdp_size,
            expert_data_parallel_size=dp_size,  # Usually EP data parallel size matches global or has specific logic
            ep_plan=ep_plan,

            # Recomputation
            recompute=parallel_args.recompute,
            recompute_plan=recompute_plan
        )

        return config


# ==============================================================================
# McoreModelFactory (Old Scheme)
# Formerly FSDP2ModelFactory
# ==============================================================================
class McoreModelFactory:
    """
    [Mcore] Factory responsible for resolving HuggingFace classes and creating
    the FSDP2-ready FSDP2Model wrapper.
    """

    @staticmethod
    def create(config: Any) -> 'FSDP2Model':
        """
        Static Factory Method.
        Args:
            config: Configuration object containing 'init_from_hf_path' and 'model_id'.
        """
        hf_path = config.init_from_hf_path
        transformer_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)

        # 1. Strategy: Determine which HF class to use
        model_cls = McoreModelFactory._resolve_model_class(config, transformer_config)
        
        # Apply model-specific patches (e.g., for NPU compatibility)
        if hasattr(model_cls, 'register_patches'):
            model_cls.register_patches(config)

        # 2. Composition: Inject configuration and class into the Wrapper
        model = FSDP2Model(
            config=config,
            transformer_config=transformer_config,
            model_cls=model_cls
        )

        return model

    @staticmethod
    def _resolve_model_class(config: Any, transformer_config: PretrainedConfig) -> Type[Any]:
        """
        Resolves the specific model class from the registry based on 'model_id'.
        """
        # Explicit mapping via config (Lookup in Registry)
        model_id = getattr(config, "model_id", None)
        if model_id:
            cls = ModelRegistry.get_model_class(model_id)
            if cls:
                return cls

        raise ValueError(f"Could not resolve model class for model_id='{model_id}'")


# ==============================================================================
# [Facade] AutoModelFactory
# Unified entry point used by AutoTrainer
# ==============================================================================
class AutoModelFactory:
    """
    Unified Factory for creating models.
    Dispatches to ModelFactory or McoreModelFactory based on the environment.
    """

    @staticmethod
    def create(*args, **kwargs):
        """
        Factory method that forwards arguments to the specific implementation.
        
        Dispatch Logic:
            - If TRAINING_BACKEND == 'mindspeed_fsdp': calls ModelFactory.create
            - Otherwise: calls McoreModelFactory.create
        """
        backend = os.environ.get("TRAINING_BACKEND", "mcore").lower()

        if backend == "mindspeed_fsdp":
            # MindSpeed FSDP implementation expects (model_args, parallel_args)
            return ModelFactory.create(*args, **kwargs)

        else:
            # Mcore implementation expects a single 'config' object
            return McoreModelFactory.create(*args, **kwargs)
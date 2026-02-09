import os
import torch
import torch.distributed as dist
from typing import Any, Type
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

from mindspeed_llm.fsdp2.models.model_registry import ModelRegistry
from mindspeed_llm.fsdp2.distributed.mindspeed_parallel_engine import MindSpeedParallelEngine
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import (
    ParallelEngineConfig,
    FSDPPlanConfig,
    TPPlanConfig,
    EPPlanConfig,
    CPPlanConfig,
    QuantizeConfig
)

from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)

# ==============================================================================
# [Mcore Imports] Dependencies for the Old Scheme
# ==============================================================================
try:
    from mindspeed_llm.fsdp2.models.fsdp2_model import FSDP2Model
except ImportError:
    # Graceful fallback if mcore dependencies are missing in a pure MindSpeed FSDP environment
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
        trust_remote_code = model_args.trust_remote_code
        hf_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=trust_remote_code
        )

        # 3. Load HF Model
        # Decide loading method based on whether training from scratch or fine-tuning.
        # Typically: SFT uses `from_pretrained`, Pretrain (from scratch) uses `from_config`.
        # if model_id is configured, load model according to model_id.
        if getattr(model_args, 'model_id', None):
            logger.info_rank0(f"> Using factory mode with model_id: {model_args.model_id}")

            model_cls = ModelRegistry.get_model_class(model_args.model_id)

            logger.info_rank0(f"> Loading model {model_cls.__name__} from pretrained path...")
            model = model_cls.from_pretrained(
                model_args.model_name_or_path,
                config=hf_config,
                low_cpu_mem_usage=True,
                device_map="cpu",
                dtype=torch.float32
            )
            logger.info_rank0("> Load model successfully")

        else:
            if getattr(model_args, 'train_from_scratch', False):
                logger.info_rank0(f"> Initializing model from config (Random Weights) for Pretraining...")
                model = AutoModelForCausalLM.from_config(
                    hf_config,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.float32 # Use FP32 for mixed precision training
                )
            else:
                logger.info_rank0(f"> Loading pretrained weights from {model_args.model_name_or_path}...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=hf_config,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch.float32,  # Use FP32 for mixed precision training
                    low_cpu_mem_usage=True,  # Reduce memory peak usage during loading
                    device_map="cpu"  # Load to CPU first; MindSpeed FSDP handles sharding and moving
                )

        # 4. Build MindSpeed FSDP Configuration
        # Dynamically calculate Data Parallel (DP) Size
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Guard against division by zero if args are not set correctly
        tp_size = parallel_args.tp_size
        fsdp_size = parallel_args.fsdp_size
        cp_size = parallel_args.cp_size
        dp_size = world_size // (tp_size * fsdp_size * cp_size)

        parallel_config = ModelFactory._build_parallel_config(model_args, parallel_args, dp_size)

        # 5. Wrap & Move
        logger.info_rank0(f"> Wrapping model with MindSpeed FSDP (TP={tp_size},CP={cp_size}, FSDP={fsdp_size})...")

        # MindSpeed FSDP will shard and wrap the CPU model based on the config.
        # The wrapped model automatically handles forward/backward communication.
        model = MindSpeedParallelEngine(config=parallel_config, model=model)

        # Finally move the model to NPU
        model = model.to(device)

        return model

    @staticmethod
    def _build_parallel_config(model_args, parallel_args, dp_size) -> 'ParallelEngineConfig':
        """
        Builds the Config based on parallel_args and hardcoded layer name rules.
        Note: The wildcards here (e.g., 'model.layers.{*}') are suitable for standard structures like Llama/Qwen.
        If using other non-standard models, these strings might need adjustment.
        """
        # --- 1. FSDP Plan ---
        # Requirement: Apply FSDP to transformer layers
        apply_modules = {
            parallel_args.fsdp_modules[0]: {'reshard_after_forward': parallel_args.reshard_after_forward,
                                            'shard_placement_fn': parallel_args.shard_placement_fn},
        }
        for modules in parallel_args.fsdp_modules[1:]:
            apply_modules[modules] = {'reshard_after_forward': parallel_args.reshard_after_forward,}
        fsdp_plan = FSDPPlanConfig(
            ignored_modules=parallel_args.ignored_modules if parallel_args.ignored_modules else [],
            apply_modules= apply_modules,
            param_dtype=parallel_args.param_dtype,
            reduce_dtype=parallel_args.reduce_dtype,
            num_to_forward_prefetch=parallel_args.num_to_forward_prefetch,
            num_to_backward_prefetch=parallel_args.num_to_backward_prefetch
        )

        # --- 2. Tensor Parallel Plan ---
        # Requirement: Column Parallel for Q/K/V/Gate/Up, Row Parallel for O/Down
        tp_plan = TPPlanConfig(
            colwise_parallel=parallel_args.tp_colwise,
            rowwise_parallel=parallel_args.tp_rowwise
        )

        # --- 3. Expert Parallel Plan ---
        # For Mixture-of-Experts (MoE) models
        ep_size = parallel_args.ep_size
        ep_fsdp_size = parallel_args.ep_fsdp_size

        ep_plan = EPPlanConfig(
            apply_modules=parallel_args.ep_modules,
            apply_efsdp_modules=parallel_args.ep_fsdp_modules,
            dispatcher=parallel_args.ep_dispatcher,
        )


        cp_plan = CPPlanConfig(
            context_parallel_type=parallel_args.cp_type,
            is_pack=getattr(model_args, "pack", False)
        )

        # --- 4. Recompute Plan ---
        # Activation Checkpointing
        recompute_plan = parallel_args.recompute_modules if parallel_args.recompute else []

        # --- 5. Quantization Config ---
        quantization_plan = QuantizeConfig(
            quant_format=model_args.quant_format,
            quant_recipe=model_args.quant_recipe,
            block_size=model_args.quant_block_size,
            quant_apply_modules=model_args.quant_apply_modules,
            quant_ignored_modules=model_args.quant_ignored_modules,
            converters=model_args.converters
        )

        # --- 6. Assemble Config ---
        # Get parallel sizes safely
        tp_size = parallel_args.tp_size
        fsdp_size = parallel_args.fsdp_size

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

            # Context Parallelism
            context_parallel_size=parallel_args.cp_size,
            context_parallel_type=parallel_args.cp_type,
            cp_plan=cp_plan,

            # Recomputation
            recompute=parallel_args.recompute,
            recompute_plan=recompute_plan,

            # Quantization
            quantization_plan = quantization_plan
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
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import os
import glob
from contextlib import contextmanager
from typing import Optional, Dict, Tuple, Set

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.utils.device import get_device_type

logger = get_logger(__name__)


# ==============================================================================
# Context Managers (from VeOmni module_utils.py)
# ==============================================================================
@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device.
    Adapted from VeOmni/accelerate.
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = (
                param
                if param.device == torch.device("meta")
                else param_cls(module._parameters[name].to("meta"), **kwargs)
            )
    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


# ==============================================================================
# Utility Functions
# ==============================================================================
def reset_hf_initialized_flag(module: nn.Module) -> None:
    """Reset HuggingFace's _is_hf_initialized flag."""
    if hasattr(module, "_is_hf_initialized"):
        setattr(module, "_is_hf_initialized", False)
    for child in module.children():
        reset_hf_initialized_flag(child)


def _find_submodule(module: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Find the leaf module according to the name."""
    pieces = name.split(".")
    for piece in pieces[:-1]:
        module = getattr(module, piece)
    return module, pieces[-1]


# ==============================================================================
# Model Loader Class
# ==============================================================================
class ModelLoader:
    """Load model on CPU or meta device."""
    
    def __init__(self, model_args, init_device: str = "cpu"):
        self.model_args = model_args
        self.init_device = init_device
        self.trust_remote_code = getattr(model_args, 'trust_remote_code', False)
        self.model_path = model_args.model_name_or_path
        self.train_from_scratch = getattr(model_args, 'train_from_scratch', False)
        self.hf_config = None
    
    def load_config(self) -> AutoConfig:
        """Load HuggingFace model config."""
        logger.info_rank0(f"> Loading config from {self.model_path}...")
        self.hf_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        return self.hf_config
    
    def create_model(self, model_cls=None) -> Tuple[nn.Module, Optional[str]]:
        """Create model based on init_device."""
        if self.hf_config is None:
            self.load_config()
        
        if self.init_device == "meta":
            return self._create_on_meta(model_cls)
        else:
            return self._create_on_cpu(model_cls)
    
    def _create_on_cpu(self, model_cls=None) -> Tuple[nn.Module, None]:
        """Create and load model on CPU."""
        if model_cls is not None:
            logger.info_rank0(f"> Loading {model_cls.__name__} on CPU...")
            model = model_cls.from_pretrained(
                self.model_path,
                config=self.hf_config,
                low_cpu_mem_usage=True,
                device_map="cpu",
                torch_dtype=torch.float32
            )
        elif self.train_from_scratch:
            logger.info_rank0("> Creating model with random weights on CPU...")
            model = AutoModelForCausalLM.from_config(
                self.hf_config,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float32
            )
        else:
            logger.info_rank0(f"> Loading pretrained model on CPU from {self.model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.hf_config,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
        
        return model, None
    
    def _create_on_meta(self, model_cls=None) -> Tuple[nn.Module, Optional[str]]:
        """Create empty model on meta device."""
        weights_path = None if self.train_from_scratch else self.model_path

        if model_cls is not None:
            logger.info_rank0(f"> Creating empty {model_cls.__name__} on meta device...")
            with init_empty_weights(), no_init_weights():
                if hasattr(model_cls, '_from_config'):
                    model = model_cls._from_config(self.hf_config)
                else:
                    model = model_cls.from_config(self.hf_config)
        elif self.train_from_scratch:
            logger.info_rank0("> Creating empty model on meta device for random init...")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(
                    self.hf_config,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float32
                )
        else:
            logger.info_rank0(f"> Creating empty model on meta device (weights: {self.model_path})...")
            with init_empty_weights(), no_init_weights():
                model = AutoModelForCausalLM.from_config(
                    self.hf_config,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float32
                )
        
        logger.info_rank0(f"> Model created on meta device. Weights path: {weights_path}")
        return model, weights_path


# ==============================================================================
# Weight Loader Class (Adapted from VeOmni load_model_weights)
# ==============================================================================
class WeightLoader:
    """
    Load weights into FSDP-wrapped model.
    Adapted from VeOmni's load_model_weights function.
    """
    
    @staticmethod
    def load(
        model: nn.Module, 
        weights_path: Optional[str], 
        device: Optional[str] = None
    ) -> None:
        """Load or initialize weights after FSDP wrapping."""
        if device is None:
            device = get_device_type()
        
        if weights_path is None:
            WeightLoader._init_random(model, device)
        else:
            WeightLoader._load_pretrained(model, weights_path, device)
    
    @staticmethod
    def _init_random(model: nn.Module, device: str) -> None:
        """Initialize model with random weights."""
        logger.info_rank0(f"> Initializing random weights on {device}...")
        
        model.to_empty(device=device)
        model = model.float()
        reset_hf_initialized_flag(model)
        
        if hasattr(model, 'init_weights'):
            model.init_weights()
            
        logger.info_rank0("> Random initialization done")
    
    @staticmethod
    @torch.no_grad()
    def _load_pretrained(model: nn.Module, weights_path: str, device: str) -> None:
        """
        Load pretrained weights - adapted from VeOmni's load_model_weights.
        """
        from torch.distributed.tensor import distribute_tensor
        
        logger.info_rank0(f"> Loading pretrained weights from {weights_path}...")
        
        # Step 1: Save buffers before to_empty (VeOmni pattern)
        buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
        parameter_names_to_load = {name for name, _ in model.named_parameters()}
        
        logger.info_rank0(f"> Saved {len(buffer_dict)} buffers, {len(parameter_names_to_load)} parameters to load")
        
        # Step 2: Materialize model to device
        model.to_empty(device=device)
        model = model.float()
        logger.info_rank0(f"> Model materialized to {device}")
        
        # Step 3: Load state dict and dispatch parameters
        state_dict_files = WeightLoader._get_state_dict_files(weights_path)
        
        for state_dict_file in state_dict_files:
            for name, tensor in WeightLoader._iterate_state_dict(state_dict_file):
                if name in buffer_dict:
                    # Update buffer in buffer_dict
                    buffer_dict[name] = tensor.clone()
                elif name in parameter_names_to_load:
                    parameter_names_to_load.remove(name)
                    WeightLoader._dispatch_parameter(model, name, tensor, distribute_tensor)
                else:
                    logger.debug(f"> Unexpected key in state dict: {name}")
        
        # Step 4: Post-process (restore buffers, handle missing params)
        WeightLoader._post_process(model, buffer_dict, parameter_names_to_load, distribute_tensor)
        
        logger.info_rank0("> Pretrained weights loaded successfully")
    
    @staticmethod
    def _get_state_dict_files(weights_path: str):
        """Get list of state dict files."""
        # Check for safetensors index
        index_file = os.path.join(weights_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json
            with open(index_file, 'r') as f:
                index = json.load(f)
            files = set(index["weight_map"].values())
            return [os.path.join(weights_path, f) for f in sorted(files)]
        
        # Check for single safetensors file
        single_safetensor = os.path.join(weights_path, "model.safetensors")
        if os.path.exists(single_safetensor):
            return [single_safetensor]
        
        # Check for multiple safetensors files
        safetensor_files = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
        if safetensor_files:
            return safetensor_files
        
        # Check for pytorch files
        pytorch_files = sorted(glob.glob(os.path.join(weights_path, "*.bin")))
        if pytorch_files:
            return pytorch_files
        
        pytorch_files = sorted(glob.glob(os.path.join(weights_path, "*.pt")))
        if pytorch_files:
            return pytorch_files
        
        raise FileNotFoundError(f"No weight files found in {weights_path}")
    
    @staticmethod
    def _iterate_state_dict(filepath: str):
        """Iterate over state dict file, yielding (key, tensor) pairs."""
        if filepath.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)
        else:
            state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
            for key, tensor in state_dict.items():
                yield key, tensor
    
    @staticmethod
    def _dispatch_parameter(
        model: nn.Module,
        name: str,
        tensor: torch.Tensor,
        dtensor_factory
    ) -> None:
        """
        Assign parameter to model - adapted from VeOmni's _dispatch_parameter.
        """
        module, local_name = _find_submodule(model, name)
        orig_tensor = dict(module.named_parameters(recurse=False))[local_name].data
        
        # Convert tensor to match original dtype and device
        tensor = tensor.to(orig_tensor)
        
        if hasattr(orig_tensor, "device_mesh"):
            # DTensor parameter
            device_mesh = orig_tensor.device_mesh
            placements = orig_tensor.placements
            dict(module.named_parameters(recurse=False))[local_name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
        else:
            # Regular parameter
            dict(module.named_parameters(recurse=False))[local_name].data.copy_(tensor)
    
    @staticmethod
    def _dispatch_buffer(
        model: nn.Module,
        name: str,
        buffer: torch.Tensor,
        dtensor_factory
    ) -> None:
        """
        Assign buffer to model - adapted from VeOmni's _dispatch_buffer.
        """
        module, local_name = _find_submodule(model, name)
        orig_buffer = dict(module.named_buffers(recurse=False))[local_name]
        
        if hasattr(orig_buffer, "device_mesh"):
            device_mesh = orig_buffer.device_mesh
            placements = orig_buffer.placements
            module.register_buffer(local_name, dtensor_factory(
                buffer.to(dtype=orig_buffer.dtype), 
                device_mesh, 
                placements
            ))
        else:
            dict(module.named_buffers(recurse=False))[local_name].copy_(buffer.to(device=orig_buffer.device, dtype=orig_buffer.dtype))
    
    @staticmethod
    def _init_parameter(model: nn.Module, name: str) -> None:
        """
        Initialize missing parameter - adapted from VeOmni's _init_parameter.
        """
        pieces = name.split(".")
        init_func = None
        module = model
        
        for piece in pieces[:-1]:
            if hasattr(module, "_init_weights"):
                init_func = getattr(module, "_init_weights", None)
            module = getattr(module, piece)
        
        if init_func is not None:
            module.apply(init_func)
        else:
            logger.warning(f"> Cannot find _init_weights for {name}, skipping initialization")
    
    @staticmethod
    def _post_process(
        model: nn.Module,
        buffer_dict: Dict[str, torch.Tensor],
        parameter_names_left: Set[str],
        dtensor_factory
    ) -> None:
        """
        Post-process after weight loading - adapted from VeOmni's post_process_after_weight_loading.
        """
        # Restore buffers
        for name, buffer in buffer_dict.items():
            try:
                WeightLoader._dispatch_buffer(model, name, buffer, dtensor_factory)
            except Exception as e:
                logger.warning(f"> Failed to restore buffer {name}: {e}")
        
        logger.info_rank0(f"> Restored {len(buffer_dict)} buffers")
        
        # Initialize missing parameters
        if parameter_names_left:
            logger.info_rank0(f"> Missing {parameter_names_left} parameters, initializing them...")
            for name in parameter_names_left:
                try:
                    WeightLoader._init_parameter(model, name)
                except Exception as e:
                    logger.warning(f"> Failed to initialize {name}: {e}")
        
        # Tie embeddings if needed
        if getattr(model.config, "tie_word_embeddings", True):
            try:
                input_embeddings = model.get_input_embeddings()
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is not None and input_embeddings is not None:
                    output_embeddings.register_parameter(
                        "weight",
                        input_embeddings.weight,
                        )
                    logger.info_rank0("> Tied input/output embeddings")
            except Exception as e:
                logger.warning(f"> Failed to tie embeddings: {e}")
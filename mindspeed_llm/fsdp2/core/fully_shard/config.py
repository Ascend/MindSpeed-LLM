import dataclasses
import yaml
from typing import Optional, Iterable, Union, Dict, Any

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, MixedPrecisionPolicy

from mindspeed.utils import _get_dtype

@dataclasses.dataclass
class Fsdp2Config:
    """
    Configuration class for FSDP2 settings.
    Handles loading from YAML and converting to torch.distributed.fsdp arguments.
    """
    sharding_size: Optional[int] = None
    sub_modules_to_wrap: Optional[Iterable[str]] = None
    reshard_after_forward: Union[bool, int] = True
    
    # mp_policy settings
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True

    # offload settings
    offload_to_cpu: bool = False
    pin_memory: bool = True  # effective only when offload_to_cpu is True

    # prefetch settings
    num_to_forward_prefetch: Optional[int] = 0
    num_to_backward_prefetch: Optional[int] = 0

    ignored_modules: Optional[Iterable[str]] = None

    recompute_modules: Optional[Iterable[str]] = None
    use_reentrant: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format required by fully_shard."""
        mp_policy = self._mp_policy()
        
        if self.offload_to_cpu:
            offload_policy = CPUOffloadPolicy(pin_memory=self.pin_memory)
        else:
            offload_policy = OffloadPolicy()  # means no offloading

        kwargs = {
            "mp_policy": mp_policy,
            "reshard_after_forward": self.reshard_after_forward,
            "offload_policy": offload_policy,
        }
        return kwargs

    @classmethod
    def load_from_yaml(cls, yml_file: str) -> 'Fsdp2Config':
        """Factory method to create config instance from a YAML file."""
        with open(yml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Filter keys to match dataclass fields
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        return cls(**kwargs)

    def _mp_policy(self) -> MixedPrecisionPolicy:
        """Construct the MixedPrecisionPolicy object."""
        param_dtype = _get_dtype(self.param_dtype) if self.param_dtype else None
        reduce_dtype = _get_dtype(self.reduce_dtype) if self.reduce_dtype else None
        output_dtype = _get_dtype(self.output_dtype) if self.output_dtype else None
        
        return MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            output_dtype=output_dtype,
            cast_forward_inputs=self.cast_forward_inputs
        )
"""
Base module for FLOPS/MFU calculation: abstract interface, fallback estimator, utilities.
"""
from abc import ABC, abstractmethod
from typing import List
import torch

from transformers import PretrainedConfig

from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------
# BaseFlopsEstimator
# --------------------------
class BaseFlopsEstimator(ABC):
    """
    Abstract base class for model-specific FLOPS estimation.
    All model FLOPS calculators must implement this interface.
    """
    def __init__(self, config: PretrainedConfig):
        self.config = config

    @abstractmethod
    def calculate_achieved_flops(
        self, 
        tokens_sum: int, 
        batch_seqlens: List[int], 
        delta_time: float
    ) -> float:
        """
        Calculate achieved FLOPS for one batch in TFLOPS.
        
        Args:
            tokens_sum: total number of tokens in the batch
            batch_seqlens: list of sequence lengths for each sample
            delta_time: time spent processing the batch (seconds)
        
        Returns:
            achieved TFLOPS for this batch
        """
        pass


# --------------------------
# UnknownModelFlopsEstimator
# --------------------------
class UnknownModelFlopsEstimator(BaseFlopsEstimator):
    """
    Fallback estimator for unsupported model types.
    Returns 0.0 and logs a warning.
    """
    # Class variable to track if warning has been printed
    _warning_printed = False
    
    def calculate_achieved_flops(
        self, 
        tokens_sum: int, 
        batch_seqlens: List[int], 
        delta_time: float
    ) -> float:
        if not UnknownModelFlopsEstimator._warning_printed:
            logger.warn_rank0(
                f"Model type '{self.config.model_type}' is not supported for MFU calculation. "
                f"Supported models are registered in flops_factory.py."
            )
            UnknownModelFlopsEstimator._warning_printed = True
        return 0.0


# --------------------------
# Common utilities
# --------------------------
def unit_convert(number: float, level: str) -> float:
    """
    Convert numerical values between metric units (B/K/M/G/T/P) from base unit "B"
    
    Args:
        number (float): Original value in "B" unit to be converted
        level (str): Target unit (B/K/M/G/T/P)
    
    Returns:
        float: Converted value in target unit (return original number if invalid input)
    """
    units = ["B", "K", "M", "G", "T", "P"]
    # Basic edge case handling (keep consistent with original logic)
    if number <= 0 or level not in units:
        return number
    target_idx = units.index(level)
    conversion_factor = 1000 ** target_idx  # 1000^0=1 (B), 1000^1=1000 (K), 1000^2=1000000 (M)...
    return number / conversion_factor


def get_device_flops(unit: str = "T") -> float:
    """
    Get theoretical peak FLOPS of the current device with unit conversion

    Args:
        unit (str): Target unit for FLOPS value (B/K/M/G/T/P), default: "T"
    
    Returns:
        float: Peak FLOPS value in the specified unit
    """
    DEVICE_FLOPS_MAP = [
        ("H100", 989e12),
        ("H800", 989e12),
        ("A100", 312e12),
        ("A800", 312e12),
        ("L40", 181.05e12),
        ("L20", 119.5e12),
        ("H20", 148e12),
        ("910B", 354e12),
        ("B200", 2250e12),
    ]

    device_name = torch.accelerator.get_device_name()
    flops = float("inf")
    for key, val in DEVICE_FLOPS_MAP:
        if key in device_name:
            flops = val
            break
    return unit_convert(flops, unit)
from typing import Any
import torch


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = torch.npu.is_available()
def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    elif IS_NPU_AVAILABLE:
        device = "npu"
    else:
        device = "cpu"

    return device
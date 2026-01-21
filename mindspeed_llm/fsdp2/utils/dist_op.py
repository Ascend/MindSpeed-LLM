import torch
import torch.distributed as dist
from typing import Union, Tuple, List

def all_reduce(
    inputs: Union[float, torch.Tensor, Tuple, List], 
    group: dist.ProcessGroup = None, 
    average: bool = True
) -> Union[float, Tuple[float, ...]]:
    """
    Performs an All-Reduce operation on input scalars or tensors (averaging by default).
    
    Args:
        inputs: A single scalar, Tensor, or a tuple/list of them (e.g., (loss, grad_norm)).
        group: The distributed process group (ProcessGroup).
        average: Whether to calculate the average after reduction (Sum / GroupSize).
    
    Returns:
        The aggregated Python scalar or tuple.
    """
    # 1. Format Standardization: Check if input is a sequence (tuple or list)
    is_sequence = isinstance(inputs, (tuple, list))
    if not is_sequence:
        inputs = [inputs]

    # 2. Prepare Tensors: Ensure all data is on the NPU
    packed_tensors = []
    # Get the current NPU device to prevent tensors from defaulting to CPU
    device = torch.device(f"npu:{torch.npu.current_device()}")
    
    for item in inputs:
        if isinstance(item, torch.Tensor):
            # If it is a Tensor, detach it to avoid affecting the computation graph and move to NPU
            t = item.detach().clone().to(device)
        else:
            # If it is a Python scalar (float/int), convert it to a Tensor
            t = torch.tensor(item, device=device, dtype=torch.float32)
        packed_tensors.append(t)

    # 3. Execute All-Reduce
    # If no group is specified, default to the global world group
    if group is None:
        group_size = dist.get_world_size()
    else:
        group_size = dist.get_world_size(group)

    # For efficiency, we could stack tensors for a single communication call and then unbind.
    # However, assuming inputs are few (usually just loss and grad_norm), we loop for code clarity.
    for t in packed_tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        if average:
            t /= group_size

    # 4. Convert back to Python scalars
    results = [t.item() for t in packed_tensors]

    # 5. Restore original structure and return
    if is_sequence:
        return tuple(results)
    else:
        return results[0]
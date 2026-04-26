import torch
import torch_npu


_SWIGLU_LIMIT = None


def get_swiglu_limit():
    from megatron.training.global_vars import get_args
    args = get_args()
    return getattr(args, 'swiglu_limit', 0)


def fused_swiglu_with_limit(x, limit=0):
    # use limit from global vars
    global _SWIGLU_LIMIT
    if _SWIGLU_LIMIT is None:
        _SWIGLU_LIMIT = get_swiglu_limit()
    else:
        limit = _SWIGLU_LIMIT
    if limit > 0:
        y_1, y_2 = torch.chunk(x.float(), 2, dim=0)
        y_1 = y_1.clamp(min=None, max=limit)
        y_2 = y_2.clamp(min=-limit, max=limit)
        return torch_npu.npu_swiglu(torch.cat([y_1, y_2], dim=0).to(x.dtype), dim=-1)
    return torch_npu.npu_swiglu(x, dim=-1)

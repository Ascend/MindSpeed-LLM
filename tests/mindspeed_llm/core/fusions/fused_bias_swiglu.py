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
        y_1, y_2 = torch.chunk(x, 2, dim=-1)
        y_1 = y_1.clamp(min=None, max=limit)
        y_2 = y_2.clamp(min=-limit, max=limit)
        return torch_npu.npu_swiglu(torch.cat([y_1, y_2], dim=-1).to(x.dtype), dim=-1)
    return torch_npu.npu_swiglu(x, dim=-1)


def triton_swiglu_limit(x, limit=None):
    # use limit from global vars
    global _SWIGLU_LIMIT
    if limit is None:
        if _SWIGLU_LIMIT is None:
            _SWIGLU_LIMIT = get_swiglu_limit()
        limit = _SWIGLU_LIMIT
    if limit > 0:
        from mindspeed_llm.ops.triton.triton_swiglu_with_limit import triton_swiglu_with_limit

        if x.ndim == 2:
            return triton_swiglu_with_limit(x.unsqueeze(1), limit).squeeze(1)
        return triton_swiglu_with_limit(x, limit)
    return torch_npu.npu_swiglu(x, dim=-1)

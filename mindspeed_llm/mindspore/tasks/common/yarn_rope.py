import math
import torch


def yarn_linear_ramp_mask(min_, max_, dim, device):
    if min_ == max_:
        max_ += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_) / (max_ - min_)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func
import numpy as np
import torch


def yarn_linear_ramp_mask(min_, max_, dim):
    if min_ == max_:
        max_ += 0.001  # Prevent singularity

    linear_func = (np.arange(dim, dtype=np.float32) - min_) / (max_ - min_)
    ramp_func = np.clip(linear_func, 0, 1)
    return torch.tensor(ramp_func)
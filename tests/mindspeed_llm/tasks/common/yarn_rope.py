import math

import torch


class YarnRotaryPositionEmbedding():
    """Yarn Rotary Embedding Methods"""

    @staticmethod
    def yarn_find_correction_dim(
            num_rotations, dim, base=10000, max_position_embeddings=2048
    ):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
                2 * math.log(base)
        )

    @staticmethod
    def yarn_find_correction_range(
            low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
    ):
        low = math.floor(
            YarnRotaryPositionEmbedding.yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            YarnRotaryPositionEmbedding.yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    @staticmethod
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def yarn_linear_ramp_mask(min_, max_, dim):
        if min_ == max_:
            max_ += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
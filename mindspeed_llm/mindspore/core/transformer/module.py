#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from mindspeed.mindspore.core.utils import cached_modules


def set_is_first_microbatch(self):
    """
    Sets the is_first_microbatch flag if it exists. When this flag is set, TE modules will update their fp8 parameter
    cache.
    """
    for m in cached_modules(self):
        if hasattr(m, "is_first_microbatch"):
            m.is_first_microbatch = True

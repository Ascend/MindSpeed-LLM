# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.transformer.transformer_config import TransformerConfig


def fp8_context_wrapper(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Wraps the fp8_context_wrapper function."""

    def wrapper_fn(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with get_fp8_context(config, layer_no, is_init):
                return fn(*args, **kwargs)

        return wrapper

    return wrapper_fn
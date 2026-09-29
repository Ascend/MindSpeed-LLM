# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
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

import inspect
from functools import wraps


def write_item_serialization_format_wrapper(write_item):
    """Supply the serialization format required by newer PyTorch versions.

    Megatron calls ``_write_item`` with the legacy PyTorch signature. Return the
    original function unchanged when that signature is still in use.
    """
    write_item_signature = inspect.signature(write_item)
    if 'serialization_format' not in write_item_signature.parameters:
        return write_item

    @wraps(write_item)
    def wrapped_write_item(*args, **kwargs):
        # bind_partial detects the argument whether it was passed by position or keyword.
        if 'serialization_format' not in write_item_signature.bind_partial(*args, **kwargs).arguments:
            from torch.distributed.checkpoint.filesystem import SerializationFormat

            kwargs['serialization_format'] = SerializationFormat.TORCH_SAVE
        return write_item(*args, **kwargs)

    return wrapped_write_item

# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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

"""General utilities."""

import inspect
import sys

from mindspeed.patch_utils import Patch, MindSpeedPatchesManager


def apply_patch(original_func_name, new_func):
    split_name = original_func_name.rsplit('.', 1)
    if len(split_name) == 1:
        orig_module_name, orig_func_name = original_func_name, None
    else:
        orig_module_name, orig_func_name = split_name

    orig_module, orig_func = Patch.parse_path(orig_module_name, orig_func_name, False)
    final_patch_func = new_func
    if orig_func_name is not None:
        setattr(orig_module, orig_func_name, final_patch_func)
    for _, value in sys.modules.copy().items():
        if orig_func_name is not None and hasattr(value, orig_func_name) \
                and id(getattr(value, orig_func_name)) == id(orig_func):
            setattr(value, orig_func_name, final_patch_func)


def clear_wrapper_v2(original_func_name, target_func):
    '''update the pt wrapper patch with mindspore wrapper'''
    reset_patch_v2(original_func_name)
    # orig_func is the original megatron method
    orig_func = inspect.unwrap(target_func)
    # patch with orig_func, which is equivalent to restore this patch to the original megatron method
    apply_patch(original_func_name, orig_func)


def reset_patch_v2(original_func_name):
    '''clear the wrapper info in Patch object'''
    target_patch = MindSpeedPatchesManager.patches_info[original_func_name]
    target_patch.wrappers = []

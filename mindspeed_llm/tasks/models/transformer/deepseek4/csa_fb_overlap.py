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


def csa_fb_overlap_attention_forward_wrapper(forward_func):
    if getattr(forward_func, '_csa_fb_overlap_wrapped', False):
        return forward_func

    @wraps(forward_func)
    def wrapper(*args, **kwargs):
        hidden_states = forward_func(*args, **kwargs)
        layer = args[0]
        self_attention = getattr(layer, 'self_attention', None)
        if getattr(layer, 'defer_attention_recompute_for_mhc_post', False):
            if hasattr(self_attention, 'discard_csa_attention_intermediate_outputs'):
                self_attention.discard_csa_attention_intermediate_outputs()
            return hidden_states
        if hasattr(self_attention, 'discard_csa_attention_output'):
            self_attention.discard_csa_attention_output(hidden_states)
        return hidden_states

    wrapper._csa_fb_overlap_wrapped = True
    return wrapper

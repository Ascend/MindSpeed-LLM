# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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


GROUPED_MLP_LORA_PARAMETER_NAMES = (
    "weight1_lora_a",
    "weight1_lora_b",
    "weight2_lora_a",
    "weight2_lora_b",
)


def enable_grouped_mlp_lora_parameters(model, grouped_mlp_cls):
    """Re-enable custom grouped-expert LoRA parameters after PEFT wrapping.

    PEFT 0.21 determines adapter parameters from PEFT tuner layer types and
    freezes parameters belonging to other module types. LoraParallelGroupedMLP
    is implemented outside PEFT, so its LoRA parameters must be restored after
    ``get_peft_model`` while its base expert weights remain frozen.

    Returns the number of trainable parameter elements restored.
    """
    enabled_numel = 0
    for module in model.modules():
        if not isinstance(module, grouped_mlp_cls):
            continue
        for parameter_name in GROUPED_MLP_LORA_PARAMETER_NAMES:
            parameter = getattr(module, parameter_name)
            parameter.requires_grad_(True)
            enabled_numel += parameter.numel()
    return enabled_numel

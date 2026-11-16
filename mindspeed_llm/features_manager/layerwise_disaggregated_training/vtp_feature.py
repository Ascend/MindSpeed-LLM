# coding=utf-8
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

from argparse import Namespace
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class VTPFeature(MindSpeedFeature):
    """Virtual Tensor Parallelism (VTP) feature.

    Enables different PP stages to have different TP sizes in LDT SFT mode.
    VTP sizes are auto-detected from per-node GPU topology after distributed init.
    Requires --layerwise-disaggregated-training to be enabled.
    --tensor-model-parallel-size should be set to the maximum TP size across nodes.
    """

    def __init__(self):
        super().__init__(feature_name="virtual-tp", optimization_level=0)

    def pre_validate_args(self, args: Namespace):
        """Inflate world_size if LDT enabled and world_size not divisible by TP*PP*CP.

        VTP sizes are auto-detected after distributed init via all_gather.
        At this stage we only ensure Megatron validation passes by inflating
        world_size to TP*PP*CP (DP=1 minimum valid value).
        """
        ldt = getattr(args, 'layerwise_disaggregated_training', False)
        if not ldt:
            return
        world_size = getattr(args, 'world_size', None)
        if world_size is None:
            return
        tp = args.tensor_model_parallel_size
        pp = args.pipeline_model_parallel_size
        cp = getattr(args, 'context_parallel_size', 1) or 1
        if world_size % (tp * pp * cp) == 0:
            return  # Already valid, no inflation needed
        args._vtp_orig_world_size = world_size
        args.world_size = tp * pp * cp  # DP=1 minimum valid value

    def post_validate_args(self, args: Namespace):
        """Restore real world_size after Megatron validation."""
        orig = getattr(args, '_vtp_orig_world_size', None)
        if orig is not None:
            args.world_size = orig
            del args._vtp_orig_world_size

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        ldt = getattr(args, 'layerwise_disaggregated_training', False)
        if not ldt:
            return

        from mindspeed_llm.core.layerwise_disaggregated_training.utils import vtp_all_gather_into_tensor_wrapper

        patch_manager.register_patch('torch.distributed.all_gather_into_tensor', vtp_all_gather_into_tensor_wrapper)

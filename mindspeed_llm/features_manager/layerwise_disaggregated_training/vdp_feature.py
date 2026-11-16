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


class VirtualDPFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__(feature_name="virtual-dp", optimization_level=0)

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        if getattr(args, "layerwise_disaggregated_training", None):
            from mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel import (
                finish_grad_sync,
                register_grad_ready,
            )
            from mindspeed_llm.core.layerwise_disaggregated_training import parallel_state
            from mindspeed_llm.core.layerwise_disaggregated_training.utils import (
                ldt_reduce_max_stat_across_model_parallel_group,
                ldt_logical_and_across_model_parallel_group,
                ldt_get_grad_norm_fp32,
                ldt_vdp_barrier_wrapper,
            )

            patch_manager.register_patch(
                "megatron.core.distributed.distributed_data_parallel.finish_grad_sync", finish_grad_sync
            )
            patch_manager.register_patch(
                "megatron.core.distributed.distributed_data_parallel.register_grad_ready", register_grad_ready
            )
            patch_manager.register_patch(
                "megatron.training.utils.reduce_max_stat_across_model_parallel_group",
                ldt_reduce_max_stat_across_model_parallel_group,
            )
            patch_manager.register_patch(
                "megatron.training.utils.logical_and_across_model_parallel_group",
                ldt_logical_and_across_model_parallel_group,
            )
            patch_manager.register_patch("megatron.core.parallel_state.create_group", parallel_state.create_group)
            patch_manager.register_patch(
                "megatron.core.optimizer.clip_grads.get_grad_norm_fp32", ldt_get_grad_norm_fp32
            )
            patch_manager.register_patch("torch.distributed.barrier", ldt_vdp_barrier_wrapper)

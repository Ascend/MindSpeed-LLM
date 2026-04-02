# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import warnings

import torch

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class QATQuantEngineFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('qat-quant-engine', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--qat-scheme', type=str, default=None,
                           help='Quantization method to use')

    def register_patches(self, pm: MindSpeedPatchesManager, args):
        if getattr(args, 'qat_scheme', None) == "w4a16-mxfp4":
            if getattr(args, "transformer_impl", "transformer_engine") == "local":
                use_optimized_linear = (
                    getattr(args, "gradient_accumulation_fusion", False) or
                    getattr(args, "async_tensor_model_parallel_allreduce", False) or
                    getattr(args, "sequence_parallel", False)
                )
                if not use_optimized_linear:
                    warnings.warn(
                        f"w4a16-mxfp4 quantization requires at least one of the following optimizations "
                        f"to be enabled to use the optimized linear layer: "
                        f"--gradient-accumulation-fusion, --async-tensor-model-parallel-allreduce, "
                        f"--sequence-parallel. "
                    )
                else:
                    from mindspeed.core.qat.layers import linear_with_grad_accumulation_and_async_w4a16_forward
                    pm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                                    linear_with_grad_accumulation_and_async_w4a16_forward)
                    from mindspeed.core.qat.layers import linear_with_grad_accumulation_and_async_w4a16_backward
                    pm.register_patch('megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                                    linear_with_grad_accumulation_and_async_w4a16_backward)
            else:
                warnings.warn(f"w4a16-mxfp4 just not support TE implement")
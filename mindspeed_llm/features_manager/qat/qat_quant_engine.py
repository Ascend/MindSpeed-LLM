# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import warnings


from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class QATQuantEngineFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('qat-quant-engine', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument(
            '--qat-scheme',
            type=str,
            default=None,
            choices=['w4a16-mxfp4', 'w4a16-mxfp4-moe-only', 'w4a4-mxfp4', 'w4a8-moe-only'],
            help='Set the QAT quantization method',
        )
        group.add_argument(
            '--qat-quant-block-size',
            type=int,
            default=None,
            choices=[32, 128],
            help='Set the QAT quantization block size (effective only when --qat-scheme=w4a8-moe-only)',
        )

    def register_patches(self, pm: MindSpeedPatchesManager, args):
        scheme = getattr(args, 'qat_scheme', None)
        if scheme == "w4a8-moe-only":
            if getattr(args, "gemm_gradient_accumulation_fusion", False):
                warnings.warn("gemm_gradient_accumulation_fusion is forced to False under 'w4a8-moe-only' scheme.")
        if scheme != "w4a8-moe-only":
            if getattr(args, 'qat_quant_block_size', None) is not None:
                warnings.warn(
                    f"--qat-quant-block-size is only effective when --qat-scheme='w4a8-moe-only', "
                    f"but current scheme is '{scheme}'. The parameter will be ignored."
                )
        if scheme == "w4a16-mxfp4":
            if getattr(args, "transformer_impl", "transformer_engine") == "local":
                use_optimized_linear = (
                    getattr(args, "gradient_accumulation_fusion", False)
                    or getattr(args, "async_tensor_model_parallel_allreduce", False)
                    or getattr(args, "sequence_parallel", False)
                )
                if not use_optimized_linear:
                    warnings.warn(
                        "w4a16-mxfp4 quantization requires at least one of the following optimizations "
                        "to be enabled to use the optimized linear layer: "
                        "--gradient-accumulation-fusion, --async-tensor-model-parallel-allreduce, "
                        "--sequence-parallel. "
                    )
                else:
                    from mindspeed.core.qat.layers import linear_with_grad_accumulation_and_async_w4a16_forward

                    pm.register_patch(
                        'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                        linear_with_grad_accumulation_and_async_w4a16_forward,
                    )
                    from mindspeed.core.qat.layers import linear_with_grad_accumulation_and_async_w4a16_backward

                    pm.register_patch(
                        'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                        linear_with_grad_accumulation_and_async_w4a16_backward,
                    )
            else:
                warnings.warn("w4a16-mxfp4 just not support TE implement")
        elif scheme == "w4a4-mxfp4":
            if getattr(args, "transformer_impl", "transformer_engine") == "local":
                use_optimized_linear = (
                    getattr(args, "gradient_accumulation_fusion", False)
                    or getattr(args, "async_tensor_model_parallel_allreduce", False)
                    or getattr(args, "sequence_parallel", False)
                )
                if not use_optimized_linear:
                    warnings.warn(
                        "w4a4-mxfp4 quantization requires at least one of the following optimizations "
                        "to be enabled to use the optimized linear layer: "
                        "--gradient-accumulation-fusion, --async-tensor-model-parallel-allreduce, "
                        "--sequence-parallel. "
                    )
                else:
                    from mindspeed.core.qat.w4a4_layers import linear_with_grad_accumulation_and_async_w4a4_forward

                    pm.register_patch(
                        'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.forward',
                        linear_with_grad_accumulation_and_async_w4a4_forward,
                    )
                    from mindspeed.core.qat.w4a4_layers import linear_with_grad_accumulation_and_async_w4a4_backward

                    pm.register_patch(
                        'megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication.backward',
                        linear_with_grad_accumulation_and_async_w4a4_backward,
                    )
            else:
                warnings.warn("w4a4-mxfp4 just not support TE implement")

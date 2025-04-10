#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from megatron.training.utils import print_rank_0
try:
    from mindspeed.core.pipeline_parallel.fb_overlap import (
        linear_backward_wgrad_detach,
        group_mlp_forward_detach,
        transformer_layer_forward_backward_overlaping,
        forward_backward_pipelining_with_interleaving
    )
    from mindspeed.core.pipeline_parallel.fb_overlap.adaptor import _make_param_hook
    from mindspeed_llm.core.pipeline_parallel.dualpipe.gpt_model import gpt_model_forward_backward_overlaping
    from mindspeed_llm.core.pipeline_parallel.dualpipe.MTP_overlap import forward_overlap
except ImportError:
    pass

from mindspeed_llm.tasks.models.transformer.multi_token_predication import MultiTokenPredication


def dualpipe_register_patches(MegatronAdaptation):
    args = MegatronAdaptation.get_args()
    MegatronAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_param_hook',
                                _make_param_hook)

    MultiTokenPredication.forward = forward_overlap
    MegatronAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel.forward',
                                gpt_model_forward_backward_overlaping)
    MegatronAdaptation.register('megatron.core.transformer.transformer_layer.TransformerLayer.forward',
                                transformer_layer_forward_backward_overlaping)
    MegatronAdaptation.register('mindspeed.core.transformer.transformer_block.NoopTransformerLayer.forward',
                                transformer_layer_forward_backward_overlaping)
    MegatronAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward_detach)

    if args.schedules_method == 'dualpipev':
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import \
            forward_backward_pipelining_with_cutinhalf
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_chunks import (
            get_model, dualpipev_fp16forward, get_num_layers_to_build, train_step, _allreduce_embedding_grads_wrapper
        )
        MegatronAdaptation.register('megatron.training.training.get_model', get_model)
        MegatronAdaptation.register('megatron.training.training.train_step', train_step)
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                                    forward_backward_pipelining_with_cutinhalf)
        MegatronAdaptation.register('megatron.legacy.model.module.Float16Module.forward', dualpipev_fp16forward)
        MegatronAdaptation.register('megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)
        MegatronAdaptation.register('megatron.training.utils.print_rank_last', print_rank_0)
        MegatronAdaptation.register('megatron.core.distributed.finalize_model_grads._allreduce_embedding_grads',
                                    _allreduce_embedding_grads_wrapper)
    else:
        MegatronAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                                    forward_backward_pipelining_with_interleaving)



# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from mindspeed.features_manager.pipeline_parallel.dualpipev_feature import DualpipeVFeature as MSDualpipeVFeature


class DualpipeVFeature(MSDualpipeVFeature):
    def register_patches(self, patch_manager, args):
        from megatron.training.utils import print_rank_0
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import forward_backward_pipelining_with_cutinhalf
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_chunks import (
            get_model, dualpipev_fp16forward, get_num_layers_to_build, train_step,
            _allreduce_embedding_grads_wrapper, evaluate, get_transformer_layer_offset, pretrain
        )
        from mindspeed.core.pipeline_parallel.dualpipev.mtp_utils import (setup_embeddings_and_output_layer_with_mtp,
                                                                         dualpipev_get_mtp_num_layers_to_build)

        if args.schedules_method == "dualpipev":

            patch_manager.register_patch(
                'megatron.training.training.get_model', get_model)
            patch_manager.register_patch(
                'megatron.training.training.train_step', train_step)
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving',
                                         forward_backward_pipelining_with_cutinhalf)
            patch_manager.register_patch(
                'megatron.core.transformer.module.Float16Module.forward', dualpipev_fp16forward)
            patch_manager.register_patch(
                'megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)
            patch_manager.register_patch(
                'megatron.training.utils.print_rank_last', print_rank_0)
            patch_manager.register_patch(
                'megatron.core.distributed.finalize_model_grads._allreduce_embedding_grads', _allreduce_embedding_grads_wrapper)
            patch_manager.register_patch('megatron.training.training.evaluate', evaluate)
            patch_manager.register_patch('megatron.core.transformer.transformer_layer.get_transformer_layer_offset', get_transformer_layer_offset)
            # Use existing patch: megatron.training.training.pretrain
            # Use existing patch: megatron.training.utils.get_batch_on_this_tp_rank
            patch_manager.register_patch("megatron.core.models.common.language_module.language_module.LanguageModule.setup_embeddings_and_output_layer",
                                        setup_embeddings_and_output_layer_with_mtp)
            patch_manager.register_patch("megatron.core.transformer.multi_token_prediction.get_mtp_num_layers_to_build",
                                        dualpipev_get_mtp_num_layers_to_build)
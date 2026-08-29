# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from argparse import ArgumentParser
from mindspeed.features_manager.tensor_parallel.tp_2d import TP2dFeature as MSTP2dFeature


class TP2dFeature(MSTP2dFeature):
    def register_patches(self, patch_manager, args):
        if getattr(args, self.feature_name, None):
            from mindspeed.core.tensor_parallel.tp_2d.norm_factory_2d import get_norm_tp_2d
            patch_manager.register_patch('megatron.legacy.model.utils.get_norm', get_norm_tp_2d)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_allreduce_layernorm_grads_wrapper
            patch_manager.register_patch('megatron.core.distributed.finalize_model_grads._allreduce_layernorm_grads',
                                 mindspeed_allreduce_layernorm_grads_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_mlp_init_wrapper
            patch_manager.register_patch('megatron.core.transformer.mlp.MLP.__init__', mindspeed_mlp_init_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_language_model_embedding_forward_wrapper
            patch_manager.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding.forward',
                                 mindspeed_language_model_embedding_forward_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_get_tensor_shapes_wrapper
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_tensor_shapes',
                                mindspeed_get_tensor_shapes_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_forward_backward_pipelining_with_interleaving_tp2d
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving',
                                mindspeed_forward_backward_pipelining_with_interleaving_tp2d)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_transformer_block_forward_wrapper
            patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward', mindspeed_transformer_block_forward_wrapper)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_transformer_config_post_init
            patch_manager.register_patch('megatron.core.transformer.transformer_config.TransformerConfig.__post_init__',
                                mindspeed_transformer_config_post_init)

            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_initialize_model_parallel_wrapper
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel', mindspeed_initialize_model_parallel_wrapper)
            
            from mindspeed.core.tensor_parallel.tp_2d.adaptor import MindSpeedRotaryEmbedding2D
            patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding',
                                    MindSpeedRotaryEmbedding2D)
            
            from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_self_attention_init_wrapper
            patch_manager.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', mindspeed_self_attention_init_wrapper)

            from mindspeed_llm.core.tensor_parallel.tp_2d.parallel_linear_2d import parallell_linear_2D_init_wrapper
            patch_manager.register_patch(
                "mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d.ParallelLinear2D.__init__",
                parallell_linear_2D_init_wrapper)

        self.more_patches_for_tp2d(patch_manager, args)

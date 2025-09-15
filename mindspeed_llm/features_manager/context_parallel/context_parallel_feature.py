# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser
from mindspeed.features_manager.context_parallel.context_parallel_feature import ContextParallelFeature as MindspeedContextParallelFeature


class ContextParallelFeature(MindspeedContextParallelFeature):

    def __init__(self):
        super().__init__()

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--context-parallel-algo', type=str, default='megatron_cp_algo',
                           choices=['megatron_cp_algo', 'hybrid_cp_algo'],
                           help='context parallel algorithm')

        # ring context parallel
        group.add_argument('--cp-window-size', type=int, default=1)
        group.add_argument('--attention-mask-type', type=str, default='causal',
                           choices=['causal', 'general'], help='context parallel attention mask type')
        group.add_argument('--use-cp-send-recv-overlap', action='store_true',
                           help='use this flag to enable cp send-recv-overlap.')
        group.add_argument("--use-fused-ring-attention-update", action='store_true',
                           help="Use fused ring attention update.")
        group.add_argument("--megatron-cp-in-bnsd", action='store_true',
                           help="Megatron CP in bnsd.")


    def validate_args(self, args):
        super().validate_args(args)
        if args.context_parallel_size > 1:
            if args.position_embedding_type == 'alibi':
                raise AssertionError("Context parallel does not support alibi")
            if args.use_kv_cache:
                raise AssertionError("Context parallel does not support use_kv_cache")
            if args.sliding_window is not None and args.seq_length > args.sliding_window:
                raise AssertionError("Context parallel does not support sliding_windows")


    def register_patches(self, patch_manager, args):
        if int(getattr(args, 'context_parallel_size', 1)) > 1 and not getattr(args, 'reset_attention_mask', None):
            from mindspeed.core.context_parallel.model_parallel_utils import initialize_model_parallel_cp_wrapper, \
                destroy_model_parallel_cp_wrapper, get_context_parallel_group_for_send_recv_overlap
            from mindspeed.core.context_parallel.rotary_pos_embedding_utils import get_pos_emb_on_this_cp_rank
            from mindspeed_llm.core.context_parallel.adaptor import CPDotProductAttention
            from mindspeed_llm.core.context_parallel.adaptor import attention_init_wrapper
            from mindspeed_llm.core.context_parallel.get_batch_utils import get_batch_on_this_cp_rank
            patch_manager.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                                         initialize_model_parallel_cp_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                                         destroy_model_parallel_cp_wrapper)
            patch_manager.register_patch('megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                                         get_context_parallel_group_for_send_recv_overlap)
            patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.get_pos_emb_on_this_cp_rank',
                                         get_pos_emb_on_this_cp_rank)
            patch_manager.register_patch('megatron.training.utils.get_batch_on_this_cp_rank', get_batch_on_this_cp_rank)
            patch_manager.register_patch('megatron.core.transformer.attention.Attention.__init__',
                                         attention_init_wrapper)
            patch_manager.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention',
                                         CPDotProductAttention)
            patch_manager.register_patch('megatron.core.extensions.transformer_engine.TEDotProductAttention',
                                         CPDotProductAttention)

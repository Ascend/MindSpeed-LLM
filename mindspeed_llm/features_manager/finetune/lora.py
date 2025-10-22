# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class LoraFeature(MindSpeedFeature):

    def __init__(self):
        super(LoraFeature, self).__init__(feature_name="lora", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                            help='Lora target modules.')
        group.add_argument('--lora-load', type=str, default=None,
                            help='Directory containing a lora model checkpoint.')
        group.add_argument('--lora-r', type=int, default=16,
                            help='Lora r.')
        group.add_argument('--lora-alpha', type=int, default=32,
                            help='Lora alpha.')
        group.add_argument('--lora-modules-to-save', nargs='+', type=str, default=None,
                            help='Lora modules to save.')
        group.add_argument('--lora-register-forward-hook', nargs='+', type=str, default=['word_embeddings', 'input_layernorm'],
                            help='Lora register forward hook.')
        group.add_argument('--lora-fusion', action='store_true',
                            help='use fusion to accelerate lora.')
        group.add_argument('--lora-ckpt-filter', action='store_true', default=False,
                            help='Enable only saving lora checkpoint.')
        group.add_argument('--qlora', action='store_true', default=False,
                            help='Enable QLoRA for fine-tuning with reduced memory usage.')
        group.add_argument('--qlora-save-dequantize', action='store_true', default=False,
                            help='Dequantize weights to original precision when saving in QLoRA tuning.')

    def register_patches(self, patch_manager, args):
        # for qlora
        from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_qlora
        if is_enable_qlora(args):
            from mindspeed_llm.tasks.posttrain.lora.qlora import get_model
            from mindspeed_llm.tasks.posttrain.lora.qlora import (parallel_linear_init_wrapper,
                                                                  linear_with_frozen_weight_forward,
                                                                  linear_with_frozen_weight_backward,
                                                                  parallel_linear_save_to_state_dict_wrapper,
                                                                  parallel_linear_load_from_state_dict_wrapper,
                                                                  groupedmlp_load_from_state_dict_wrapper,
                                                                  grouped_gemm_util_ops_gmm,
                                                                  moe_layer_overlap_all2all_gmm_op_wrapper)
            patch_manager.register_patch('megatron.training.training.get_model', 
                                          get_model)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__',
                                          parallel_linear_init_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear.__init__',
                                          parallel_linear_init_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithFrozenWeight.forward',
                                          linear_with_frozen_weight_forward)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.LinearWithFrozenWeight.backward',
                                          linear_with_frozen_weight_backward)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear._save_to_state_dict',
                                          parallel_linear_save_to_state_dict_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear._save_to_state_dict',
                                          parallel_linear_save_to_state_dict_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear._load_from_state_dict',
                                          parallel_linear_load_from_state_dict_wrapper)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear._load_from_state_dict',
                                          parallel_linear_load_from_state_dict_wrapper)
            patch_manager.register_patch('megatron.core.transformer.moe.experts.GroupedMLP._load_from_state_dict',
                                          groupedmlp_load_from_state_dict_wrapper)
            patch_manager.register_patch('mindspeed.core.transformer.moe.grouped_gemm_util.Ops.gmm',
                                          grouped_gemm_util_ops_gmm)
            patch_manager.register_patch('mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_all2all.gmm_op',
                                          moe_layer_overlap_all2all_gmm_op_wrapper)
        else:
            from mindspeed_llm.training.training import get_model_wrapper
            patch_manager.register_patch('megatron.training.training.get_model', 
                                          get_model_wrapper)

        from mindspeed_llm.core.distributed.finalize_model_grads import _allreduce_word_embedding_grads
        from mindspeed_llm.training.utils import unwrap_model_wrapper
        from mindspeed_llm.training.checkpointing import _load_base_checkpoint_wrapper, save_checkpoint_wrapper
        from mindspeed_llm.core.transformer.moe.moe_layer import lora_moe_layer_init
        patch_manager.register_patch('megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads',
                                      _allreduce_word_embedding_grads)
        # fix unwrap PerfModel 
        patch_manager.register_patch('megatron.training.checkpointing.unwrap_model',
                                      unwrap_model_wrapper)
        patch_manager.register_patch('megatron.training.training.unwrap_model',
                                      unwrap_model_wrapper)
        patch_manager.register_patch('megatron.training.checkpointing._load_base_checkpoint',
                                      _load_base_checkpoint_wrapper)
        patch_manager.register_patch('megatron.training.checkpointing.save_checkpoint',
                                      save_checkpoint_wrapper)
        if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
            patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
                                          lora_moe_layer_init)

    def validate_args(self, args):
        has_valid_lora_target = hasattr(args, 'lora_target_modules') and args.lora_target_modules

        if args.num_experts and (has_valid_lora_target and args.moe_token_dispatcher_type != "alltoall_seq"):
            raise AssertionError('Lora and Qlora in the moe only enable the alltoall_seq.')
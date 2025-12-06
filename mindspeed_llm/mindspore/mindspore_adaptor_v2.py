# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from mindspeed.patch_utils import MindSpeedPatchesManager
from mindspeed_llm.mindspore.utils import clear_wrapper_v2


class MindSporeAdaptation(MindSpeedPatchesManager):
    @staticmethod
    def register_patch(orig_func_name, new_func=None, force_patch=True, create_dummy=False, check_patch=False):
        if check_patch:
            new_func = MindSporeAdaptation.wrap_print_new_func(new_func)
        MindSpeedPatchesManager.register_patch(orig_func_name, new_func, force_patch, create_dummy)

    @staticmethod
    def wrap_print_new_func(new_func):
        from functools import wraps

        # wrap the new func with info print
        def make_patch(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"Stepping into MindSpore patch: {func.__name__}")
                return func(*args, **kwargs)

            return wrapper

        # wrap new_func before handing it off to MindSporeAdaptation.register
        new_func_with_print = make_patch(new_func)
        return new_func_with_print


def mindspore_adaptation(patch_manager, args):
    _patch_third_party_libraries()
    _patch_model_and_embedding()
    _patch_tensor_parallel_and_pipeline()
    _patch_moe_and_communication(args)
    _patch_optimizer_and_training(args)
    _patch_fused_operators(args)
    _patch_fsdp()

    # Optional patches (remain in main for control)
    if args.moe_fb_overlap:
        patch_moe_fb_overlap()

    if args.swap_optimizer:
        patch_swap_optimizer()

    if args.enable_seq1f1b:
        patch_seq1f1b(args)


def _patch_third_party_libraries():
    from mindspeed.mindspore.third_party.safetensors.torch import save_file, load_file
    MindSporeAdaptation.register_patch('safetensors.torch.save_file', save_file)
    MindSporeAdaptation.register_patch('safetensors.torch.load_file', load_file)
    MindSporeAdaptation.patches_info["safetensors.torch.save_file"].apply_patch()

    from mindspeed.mindspore.third_party.accelerate.extract import extract_model_from_parallel
    MindSporeAdaptation.register_patch('accelerate.utils.extract_model_from_parallel', extract_model_from_parallel)

    from mindspeed.mindspore.third_party.transformers.configuration_utils import dict_torch_dtype_to_str
    from mindspeed.mindspore.third_party.transformers.modeling_utils import (
        load_state_dict, _load_state_dict_into_meta_model, safe_open, get_parameter_dtype, get_parameter_or_buffer
    )
    from mindspeed.mindspore.third_party.transformers.integrations.deepspeed import is_deepspeed_available
    MindSporeAdaptation.register_patch(
        'transformers.configuration_utils.PretrainedConfig.dict_torch_dtype_to_str', dict_torch_dtype_to_str)
    MindSporeAdaptation.register_patch('transformers.modeling_utils.load_state_dict', load_state_dict)
    MindSporeAdaptation.register_patch(
        'transformers.modeling_utils._load_state_dict_into_meta_model', _load_state_dict_into_meta_model)
    MindSporeAdaptation.register_patch('transformers.modeling_utils.safe_open', safe_open)
    MindSporeAdaptation.register_patch('transformers.modeling_utils.get_parameter_dtype', get_parameter_dtype)
    MindSporeAdaptation.register_patch('transformers.integrations.deepspeed.is_deepspeed_available',
                                       is_deepspeed_available)
    MindSporeAdaptation.register_patch('transformers.modeling_utils.PreTrainedModel.get_parameter_or_buffer',
                                        get_parameter_or_buffer)

def _patch_model_and_embedding():
    from mindspeed.mindspore.core.models.common.embeddings.rotary_pos_embedding import local_rotate_half
    MindSporeAdaptation.register_patch('megatron.core.models.common.embeddings._rotate_half', local_rotate_half)

    from .tasks.common.yarn_rope import yarn_linear_ramp_mask
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.tasks.common.yarn_rope.YarnRotaryPositionEmbedding.yarn_linear_ramp_mask',
        yarn_linear_ramp_mask)

    from mindspeed_llm.mindspore.core.models.common.embeddings.rotary_pos_embedding import apply_llama3_scaling
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.core.models.common.embeddings.rotary_pos_embedding.apply_llama3_scaling', apply_llama3_scaling)


def _patch_tensor_parallel_and_pipeline():
    from mindspeed.mindspore.core.tensor_parallel.mappings import all_to_all_forward
    MindSporeAdaptation.register_patch('megatron.core.tensor_parallel.mappings._AllToAll.forward', all_to_all_forward)

    from mindspeed.mindspore.core.tensor_parallel.random import local_set_cuda_rng_state
    MindSporeAdaptation.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state',
                                       local_set_cuda_rng_state)

    from mindspeed.mindspore.core.pipeline_parallel.schedules import deallocate_output_tensor_
    MindSporeAdaptation.register_patch('megatron.core.pipeline_parallel.schedules.deallocate_output_tensor',
                                       deallocate_output_tensor_)

    from mindspeed.mindspore.core.timers import _get_global_min_max_time
    MindSporeAdaptation.register_patch('megatron.core.timers.Timers._get_global_min_max_time', _get_global_min_max_time)

    from ..mindspore.training.utils import get_batch_on_this_tp_rank
    MindSporeAdaptation.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)

    from ..mindspore.core.tensor_parallel.layers import vocab_embedding_init_func
    MindSporeAdaptation.register_patch('megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__',
                                       vocab_embedding_init_func)
    from ..mindspore.core.tensor_parallel.random import fork
    MindSporeAdaptation.register_patch('megatron.core.tensor_parallel.random.CudaRNGStatesTracker.fork', fork)

    from ..mindspore.core.tensor_parallel.data import broadcast_data
    MindSporeAdaptation.register_patch('megatron.core.tensor_parallel.data.broadcast_data', broadcast_data)

def _patch_moe_and_communication(args):
    from mindspeed.mindspore.core.transformer.moe.comm_utils import async_all_to_all
    MindSporeAdaptation.register_patch('mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils.async_all_to_all',
                                       async_all_to_all)
    MindSporeAdaptation.register_patch('mindspeed.core.transformer.moe.comm_utils.async_all_to_all', async_all_to_all)

    from mindspeed.mindspore.core.transformer.moe.token_dispatcher import preprocess
    MindSporeAdaptation.register_patch('mindspeed.core.transformer.moe.token_dispatcher.preprocess', preprocess)

    from mindspeed.mindspore.core.transformer.moe.legacy_a2a_token_dispatcher import moealltoallseqtokendispatcher_init
    MindSporeAdaptation.register_patch(
        'megatron.core.transformer.moe.legacy_a2a_token_dispatcher.MoEAlltoAllSEQTokenDispatcher.__init__',
        moealltoallseqtokendispatcher_init)
    
    from mindspeed.mindspore.core.transformer.moe.moe_feature.overlap.token_dispatcher import token_permutation
    MindSporeAdaptation.register_patch(
        'mindspeed.core.transformer.moe.moe_feature.overlap.token_dispatcher.MoEAlltoAllSeqOverLapDispatcher.token_permutation', 
        token_permutation)

    if hasattr(args, 'moe_grouped_gemm') and args.moe_grouped_gemm:
        from ..mindspore.core.transformer.moe.legacy_a2a_token_dispatcher import ascend_gmm_preprocess
        MindSporeAdaptation.register_patch(
            'megatron.core.transformer.moe.legacy_a2a_token_dispatcher.MoEAlltoAllSEQTokenDispatcher.preprocess',
            ascend_gmm_preprocess)

def _patch_optimizer_and_training(args):
    # Cross Entropy
    from ..mindspore.core.tensor_parallel.cross_entropy import calculate_predicted_logits, \
        prepare_gradient_calculation_operands
    MindSporeAdaptation.register_patch(
        'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
        calculate_predicted_logits)
    MindSporeAdaptation.register_patch(
        'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.prepare_gradient_calculation_operands',
        prepare_gradient_calculation_operands)

    # Checkpoint & Model Registration
    from mindspeed_llm.mindspore.tasks.checkpoint.models import register_functions, get_modules_from_pretrained
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.tasks.checkpoint.models.ModelBase._ModelBase__register_functions', register_functions)
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.tasks.checkpoint.models.HuggingfaceModel.get_modules_from_pretrained',
        get_modules_from_pretrained)

    from mindspeed_llm.mindspore.core.datasets.blended_megatron_dataset_builder import need_to_build_dataset
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.core.datasets.blended_megatron_dataset_builder.need_to_build_dataset', need_to_build_dataset)

    # share memory
    if args.enable_share_memory:
        from ..mindspore.tasks.dataset.shared_memory_manager import SharedMemoryManager
        MindSporeAdaptation.register_patch(
            'mindspeed_llm.tasks.dataset.shared_memory_manager.SharedMemoryManager', SharedMemoryManager)
        from ..mindspore.training.utils import _compute_actual_seq_len
        MindSporeAdaptation.register_patch(
            'mindspeed_llm.training.utils._compute_actual_seq_len', _compute_actual_seq_len)

    # Optimizer: load and save parameter
    from mindspeed.mindspore.core.optimizer.distrib_optimizer import get_parameter_state_dp_zero
    MindSporeAdaptation.register_patch(
        'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.get_parameter_state_dp_zero',
        get_parameter_state_dp_zero)
    from mindspeed.mindspore.core.optimizer.distrib_optimizer import load_parameter_state_from_dp_zero
    MindSporeAdaptation.register_patch(
        'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.load_parameter_state_from_dp_zero',
        load_parameter_state_from_dp_zero)

    # Reuse FP32 param
    if args.reuse_fp32_param:
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from mindspeed.mindspore.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
        target_func = DistributedOptimizer.__init__
        target_func_name = 'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__'
        clear_wrapper_v2(target_func_name, target_func)
        MindSporeAdaptation.register_patch(target_func_name, reuse_fp32_param_distrib_optimizer_init_wrapper)


def _patch_fused_operators(args):
    from mindspeed.mindspore.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
    MindSporeAdaptation.register_patch(
        'mindspeed.ops.npu_rotary_position_embedding.npu_rotary_position_embedding',
        npu_rotary_position_embedding)
    # MoE async comm
    if args.moe_permutation_async_comm:
        if args.moe_token_dispatcher_type == 'alltoall_seq':
            if hasattr(args, 'use_fused_moe_token_permute_and_unpermute') and \
                    args.use_fused_moe_token_permute_and_unpermute and not args.moe_expert_capacity_factor:
                from mindspeed.mindspore.core.fusions.npu_moe_token_unpermute import unpermute
                from mindspeed.mindspore.ops.npu_moe_token_permute import npu_moe_token_permute
                MindSporeAdaptation.register_patch('megatron.core.transformer.moe.moe_utils.permute',
                                                   npu_moe_token_permute)
                MindSporeAdaptation.register_patch('megatron.core.transformer.moe.moe_utils.unpermute', unpermute)

    # CoC (Communication-Computation Overlap)
    if args.use_ascend_coc:
        import torch_npu
        MindSporeAdaptation.register_patch('mindspeed.ops.lcal_functional.CoCOperations.mindspeed_ops', torch_npu)

    # A2AVC
    if args.enable_a2avc:
        from mindspeed.mindspore.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import \
            All2AllSeqTp2epDispatcherImpl
        MindSporeAdaptation.register_patch(
            'mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher.All2AllSeqTp2epDispatcherImpl',
            All2AllSeqTp2epDispatcherImpl)

        from mindspeed.mindspore.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import \
            _PatchedMOEAlltoAllSEQTptoEpTokenDispatcher
        MindSporeAdaptation.register_patch(
            'mindspeed.core.transformer.moe.moe_feature.adaptor.MindSpeedMOEAlltoAllSEQTptoEpTokenDispatcher',
            _PatchedMOEAlltoAllSEQTptoEpTokenDispatcher)

        from mindspeed.mindspore.core.tensor_parallel.mappings import all_to_all_forward_a2avc
        MindSporeAdaptation.register_patch(
            'mindspeed_llm.mindspore.core.tensor_parallel.mappings._AllToAll.forward',
            all_to_all_forward_a2avc)
    else:
        from mindspeed.mindspore.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import preprocess
        MindSporeAdaptation.register_patch(
            'mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher.All2AllSeqTp2epDispatcherImpl.preprocess',
            preprocess)

    # GMM
    from mindspeed.mindspore.ops.gmm import _GMM_patched_load
    MindSporeAdaptation.register_patch('mindspeed.op_builder.gmm_builder.GMMOpBuilder.load', _GMM_patched_load)

    from mindspeed.mindspore.ops.gmm import _GMM_patched_load2
    MindSporeAdaptation.register_patch('mindspeed.op_builder.gmm_builder.GMMV2OpBuilder.load', _GMM_patched_load2)

    from mindspeed.mindspore.ops.npu_ring_attention_update import _ring_atten_patched_load
    MindSporeAdaptation.register_patch(
        "mindspeed.op_builder.npu_ring_attention_update_builder.RingAttentionUpdateOpBuilder.load",
        _ring_atten_patched_load)

    from mindspeed.ops.gmm import _npu_gmm, _npu_gmm_v2
    MindSporeAdaptation.register_patch('torch.ops.mindspeed.npu_gmm', _npu_gmm)
    MindSporeAdaptation.register_patch('torch.ops.mindspeed.npu_gmm_v2', _npu_gmm_v2)

    # ema
    if args.optimizer_selection == 'fused_ema_adamw':
        from mindspeed.mindspore.ops.npu_apply_fused_ema_adamw import _fused_ema_adamw_patched_load
        MindSporeAdaptation.register_patch("mindspeed.op_builder.fused_ema_adamw_builder.FusedEmaAdamWOpBuilder.load", _fused_ema_adamw_patched_load)

    # Gradient accumulation fusion
    if args.gemm_gradient_accumulation_fusion:
        from torch_npu import npu_groupmatmul_add_fp32
        MindSporeAdaptation.register_patch('mindspeed.ops.npu_groupmatmul_add.npu_groupmatmul_add_fp32',
                                           npu_groupmatmul_add_fp32)

    # Matmul add ops
    from mindspeed.mindspore.ops.npu_matmul_add import npu_matmul_add_fp32
    MindSporeAdaptation.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32)

    # Fused AdamW v2
    from torch import npu_apply_fused_adamw_v2
    MindSporeAdaptation.register_patch('mindspeed.ops.npu_apply_fused_adamw_v2.npu_apply_fused_adamw_v2',
                                       npu_apply_fused_adamw_v2)


def pre_validate_args(patch_manager):
    pass


def mindspore_pre_validate_args(args):
    pass


def mindspore_validate_args(args):
    pass


def mindspore_post_validate_args(args):
    pass


def mindspore_pre_register_patches(manager, args):
    pass


def patch_moe_zerc():
    from mindspeed.mindspore.core.transformer.moe.moe_zerc.fwdbwd import \
        transformer_layer_forward_moe_backward_dense_overlaping_zerc, \
        transformer_layer_forward_moe_backward_moe_overlaping_zerc
    MindSporeAdaptation.register_patch(
        'mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_dense_overlaping',
        transformer_layer_forward_moe_backward_dense_overlaping_zerc)
    MindSporeAdaptation.register_patch(
        'mindspeed.core.pipeline_parallel.fb_overlap.overlap_funcs.fwdbwd.transformer_layer_forward_moe_backward_moe_overlaping',
        transformer_layer_forward_moe_backward_moe_overlaping_zerc)
    from mindspeed.mindspore.core.transformer.moe.moe_zerc.token_dispatcher import zerc_alltoall_token_perm1, \
        zerc_alltoall_token_perm2, zerc_alltoall_token_unperm1, zerc_alltoall_token_unperm2
    MindSporeAdaptation.register_patch(
        'mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_perm1',
        zerc_alltoall_token_perm1)
    MindSporeAdaptation.register_patch(
        'mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_perm2',
        zerc_alltoall_token_perm2)
    MindSporeAdaptation.register_patch(
        'mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_unperm1',
        zerc_alltoall_token_unperm1)
    MindSporeAdaptation.register_patch(
        'mindspeed.core.pipeline_parallel.fb_overlap.modules.token_dispatcher.alltoall_token_unperm2',
        zerc_alltoall_token_unperm2)


def patch_swap_optimizer():
    from mindspeed.mindspore.core.optimizer.swap_optimizer.swap_optimizer import swap_adamw_step
    MindSporeAdaptation.register_patch('mindspeed.core.optimizer.adamw.AdamW.step', swap_adamw_step)


def patch_moe_fb_overlap():
    pass


def patch_seq1f1b(args):
    if args.seq1f1b_splits <= 1:
        raise ValueError("when setting enable_seq1f1b, seq1f1b_splits should be greater than 1")
    # attention
    from mindspeed.mindspore.core.pipeline_parallel.seq1f1b.attention import attention_init_wrapper, attention_forward_wrapper, core_attention_forward_wrapper
    MindSporeAdaptation.register_patch('megatron.core.transformer.attention.Attention.__init__', attention_init_wrapper)
    MindSporeAdaptation.register_patch('megatron.core.transformer.attention.Attention.forward', attention_forward_wrapper)
    MindSporeAdaptation.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward', core_attention_forward_wrapper)
    # rotary embedding
    from mindspeed.mindspore.core.pipeline_parallel.seq1f1b.rotary_pos_embedding import rotary_embedding_forward_wrapper
    MindSporeAdaptation.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward', rotary_embedding_forward_wrapper)
    # seq1f1b schedules
    from mindspeed.mindspore.core.pipeline_parallel.seq1f1b.schedules import forward_backward_pipelining_without_interleaving_seq1f1b
    MindSporeAdaptation.register_patch('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving', forward_backward_pipelining_without_interleaving_seq1f1b)
    # recompute
    from megatron.core.tensor_parallel.random import CheckpointFunction
    from mindspeed.mindspore.core.pipeline_parallel.seq1f1b.random import checkpoint_forward_wrapper, checkpoint_backward_wrapper
    target_func_name = 'megatron.core.tensor_parallel.random.CheckpointFunction.forward'
    clear_wrapper_v2(target_func_name, CheckpointFunction.forward)
    MindSporeAdaptation.register_patch(target_func_name, checkpoint_forward_wrapper)
    target_func_name = 'megatron.core.tensor_parallel.random.CheckpointFunction.backward'
    clear_wrapper_v2(target_func_name, CheckpointFunction.backward)
    MindSporeAdaptation.register_patch(target_func_name, checkpoint_backward_wrapper)
    # FA
    from mindspeed_llm.mindspore.core.pipeline_parallel.seq1f1b.custom_dot_product_attention import npu_fusion_attention_wrapper
    MindSporeAdaptation.register_patch('torch_npu.npu_fusion_attention', npu_fusion_attention_wrapper)
    # moe
    from mindspeed_llm.mindspore.core.pipeline_parallel.seq1f1b.router import topk_router_routing_wrapper
    MindSporeAdaptation.register_patch('megatron.core.transformer.moe.router.TopKRouter.routing', topk_router_routing_wrapper)
    # multi-head latent attention
    from mindspeed_llm.mindspore.core.pipeline_parallel.seq1f1b.multi_latent_attention import custom_mla_self_attention_forward
    MindSporeAdaptation.register_patch('mindspeed_llm.tasks.models.transformer.multi_latent_attention.CustomMLASelfAttention.forward', custom_mla_self_attention_forward)
    # seq1f1b sft/pretrain dataloader
    from mindspeed_llm.mindspore.core.pipeline_parallel.seq1f1b.seq1f1b_batch import get_batch_wrapper
    from mindspeed_llm.mindspore.core.pipeline_parallel.seq1f1b.sft_trainer import sft_trainer_loss_func
    MindSporeAdaptation.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_wrapper)
    MindSporeAdaptation.register_patch('mindspeed_llm.tasks.posttrain.sft.sft_trainer.SFTTrainer.get_batch', get_batch_wrapper)
    MindSporeAdaptation.register_patch('mindspeed_llm.tasks.posttrain.sft.sft_trainer.SFTTrainer.loss_func', sft_trainer_loss_func)

    
def _patch_fsdp():
    from mindspeed.core.distributed.custom_fsdp.param_and_grad_buffer import gradient_reduce_preprocessing
    MindSporeAdaptation.register_patch('megatron.core.distributed.custom_fsdp.param_and_grad_buffer.gradient_reduce_preprocessing', gradient_reduce_preprocessing)

    from mindspeed.mindspore.core.distributed.custom_fsdp.param_and_grad_buffer import all_gather_params_wo_coalescing, mark_bucket_ready_wo_coalescing, zero_grad, update_main_grads
    MindSporeAdaptation.register_patch('megatron.core.distributed.custom_fsdp.param_and_grad_buffer.AllGatherPipeline.all_gather_params', all_gather_params_wo_coalescing)
    MindSporeAdaptation.register_patch('megatron.core.distributed.custom_fsdp.param_and_grad_buffer.GradReducePipeline.mark_bucket_ready', mark_bucket_ready_wo_coalescing)
    MindSporeAdaptation.register_patch('megatron.core.distributed.custom_fsdp.param_and_grad_buffer.ParamAndGradBuffer.zero_grad', zero_grad)
    MindSporeAdaptation.register_patch('megatron.core.distributed.custom_fsdp.param_and_grad_buffer.ParamAndGradBuffer.update_main_grads', update_main_grads)

    from mindspeed.mindspore.core.distributed.custom_fsdp.fully_sharded_data_parallel import _register_fsdp_hooks
    MindSporeAdaptation.register_patch('megatron.core.distributed.custom_fsdp.fully_sharded_data_parallel.FullyShardedDataParallel._register_fsdp_hooks', _register_fsdp_hooks)

    from mindspeed_llm.mindspore.core.transformer.moe.router import topk_router_gating_func
    MindSporeAdaptation.register_patch('megatron.core.transformer.moe.router.TopKRouter.gating', topk_router_gating_func)


def mindspore_register_args(group):
    group.add_argument('--enable-a2avc', type=int, choices=[0, 1, 2], default=0,
                       help='0: Disable a2avc,'
                            '1: Enable a2avc & Use mindspore comm_func.py & with verification (run slower than 2),'
                            '2: Enable a2avc & Use msadapter comm_func.py & without verification (run faster than 1)')
    group.add_argument('--enable-share-memory', action='store_true', default=False,
                       help='Enable shared memory for passing actual_seq_len when reset-position-ids is enabled.')
    group.add_argument('--enable-seq1f1b', action='store_true', default=False,
                       help='Enable seq1f1b')
    group.add_argument('--seq1f1b-splits', type=int, default=4,
                       help='num of splits in seq1f1b, if set to 1, then use 1f1b')
    group.add_argument('--seq1f1b-balance-method', type=str,
                       default='average', choices=['average', 'uniform_comp'],
                       help='method to balance sequence and first-then-first-batch')

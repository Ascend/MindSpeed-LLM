# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from mindspeed.patch_utils import MindSpeedPatchesManager
from mindspeed_llm.mindspore.utils_v2 import clear_wrapper_v2


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
    from mindspeed.mindspore.third_party.safetensors.torch import save_file, load_file
    MindSporeAdaptation.register_patch('safetensors.torch.save_file', save_file)
    MindSporeAdaptation.register_patch('safetensors.torch.load_file', load_file)

    MindSporeAdaptation.patches_info["safetensors.torch.save_file"].apply_patch()

    # accelerate
    from mindspeed.mindspore.third_party.accelerate.extract import extract_model_from_parallel
    MindSporeAdaptation.register_patch('accelerate.utils.extract_model_from_parallel', extract_model_from_parallel)

    # transformers
    from mindspeed.mindspore.third_party.transformers.configuration_utils import dict_torch_dtype_to_str
    MindSporeAdaptation.register_patch('transformers.configuration_utils.PretrainedConfig.dict_torch_dtype_to_str',
                                 dict_torch_dtype_to_str)

    from mindspeed.mindspore.third_party.transformers.modeling_utils import load_state_dict, \
        _load_state_dict_into_meta_model, safe_open, get_parameter_dtype
    MindSporeAdaptation.register_patch('transformers.modeling_utils.load_state_dict', load_state_dict)
    MindSporeAdaptation.register_patch('transformers.modeling_utils._load_state_dict_into_meta_model',
                                 _load_state_dict_into_meta_model)
    MindSporeAdaptation.register_patch('transformers.modeling_utils.safe_open', safe_open)
    MindSporeAdaptation.register_patch('transformers.modeling_utils.get_parameter_dtype', get_parameter_dtype)

    from ..core.models.gpt.gpt_model import GPTModel
    MindSporeAdaptation.register_patch('megatron.core.models.gpt.gpt_model.GPTModel', GPTModel)

    if args.moe_permutation_async_comm:
        if args.moe_token_dispatcher_type == 'alltoall_seq':
            if hasattr(args,
                       'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute and not args.moe_expert_capacity_factor:
                from mindspeed.mindspore.core.fusions.npu_moe_token_permute import permute_wrapper
                from mindspeed.mindspore.core.fusions.npu_moe_token_unpermute import unpermute_wrapper
                MindSporeAdaptation.register_patch('megatron.core.transformer.moe.moe_utils.permute', permute_wrapper)
                MindSporeAdaptation.register_patch('megatron.core.transformer.moe.moe_utils.unpermute',
                                                   unpermute_wrapper)

    from mindspeed.mindspore.core.transformer.moe.grouped_gemm_util import Ops
    MindSporeAdaptation.register_patch('megatron.core.transformer.moe.grouped_gemm_util.ops', Ops)

    from mindspeed.mindspore.core.distributed.param_and_grad_buffer import register_grad_ready
    MindSporeAdaptation.register_patch('megatron.core.distributed.param_and_grad_buffer.register_grad_ready',
                                       register_grad_ready)

    from mindspeed.mindspore.core.models.common.embeddings.rotary_pos_embedding import get_rotary_seq_len, \
        local_rotate_half
    MindSporeAdaptation.register_patch(
        'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
        get_rotary_seq_len)
    MindSporeAdaptation.register_patch('megatron.core.models.common.embeddings._rotate_half', local_rotate_half)

    from mindspeed.mindspore.core.optimizer.optimizer import megatron_optimizer_init
    MindSporeAdaptation.register_patch('megatron.core.optimizer.optimizer.MegatronOptimizer.__init__',
                                       megatron_optimizer_init)


    from mindspeed.mindspore.core.tensor_parallel.data import local_build_key_size_numel_dictionaries
    MindSporeAdaptation.register_patch('megatron.core.tensor_parallel.data._build_key_size_numel_dictionaries',
                                       local_build_key_size_numel_dictionaries)  # 1097

    from mindspeed.mindspore.core.tensor_parallel.mappings import all_to_all_forward
    MindSporeAdaptation.register_patch('megatron.core.tensor_parallel.mappings._AllToAll.forward', all_to_all_forward)

    from mindspeed.mindspore.core.tensor_parallel.random import local_set_cuda_rng_state
    MindSporeAdaptation.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state',
                                       local_set_cuda_rng_state)

    from ..mindspore.training.utils import get_batch_on_this_tp_rank
    MindSporeAdaptation.register_patch('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)

    from ..mindspore.core.tensor_parallel.cross_entropy import calculate_predicted_logits, \
        prepare_gradient_calculation_operands
    MindSporeAdaptation.register_patch(
        'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
        calculate_predicted_logits)
    MindSporeAdaptation.register_patch(
        'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.prepare_gradient_calculation_operands',
        prepare_gradient_calculation_operands)

    from mindspeed.mindspore.core.timers import _get_global_min_max_time
    MindSporeAdaptation.register_patch('megatron.core.timers.Timers._get_global_min_max_time', _get_global_min_max_time)

    from ..mindspore.core.optimizer.distrib_optimizer import get_parameter_state_dp_zero
    MindSporeAdaptation.register_patch(
        'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.get_parameter_state_dp_zero',
        get_parameter_state_dp_zero)


    if args.use_ascend_coc:
        from mindspeed.mindspore.ops.lcal_functional import all_gather_matmul, all_gather_matmul_v2, \
            matmul_reduce_scatter, matmul_all_reduce, pure_matmul
        MindSporeAdaptation.register_patch('mindspeed.ops.lcal_functional.CoCOperations.all_gather_matmul',
                                           all_gather_matmul)
        MindSporeAdaptation.register_patch('mindspeed.ops.lcal_functional.CoCOperations.all_gather_matmul_v2',
                                           all_gather_matmul_v2)
        MindSporeAdaptation.register_patch('mindspeed.ops.lcal_functional.CoCOperations.matmul_reduce_scatter',
                                           matmul_reduce_scatter)
        MindSporeAdaptation.register_patch('mindspeed.ops.lcal_functional.CoCOperations.matmul_all_reduce',
                                           matmul_all_reduce)
        MindSporeAdaptation.register_patch('mindspeed.ops.lcal_functional.CoCOperations.pure_matmul', pure_matmul)

    from mindspeed.mindspore.core.transformer.module import fp32_to_float16
    MindSporeAdaptation.register_patch('megatron.core.transformer.module.fp32_to_float16', fp32_to_float16)

    from mindspeed.mindspore.core.transformer.moe.comm_utils import async_all_to_all
    MindSporeAdaptation.register_patch('mindspeed.core.transformer.moe.moe_feature.overlap.comm_utils.async_all_to_all',
                                       async_all_to_all)
    MindSporeAdaptation.register_patch('mindspeed.core.transformer.moe.comm_utils.async_all_to_all',
                                       async_all_to_all)

    if args.moe_fb_overlap:
        patch_moe_fb_overlap()

    if args.swap_optimizer:
        patch_swap_optimizer()

    if args.gemm_gradient_accumulation_fusion:
        from torch_npu import npu_groupmatmul_add_fp32
        MindSporeAdaptation.register_patch('mindspeed.ops.npu_groupmatmul_add.npu_groupmatmul_add_fp32',
                                           npu_groupmatmul_add_fp32)

    from mindspeed.mindspore.ops.npu_matmul_add import npu_matmul_add_fp32
    MindSporeAdaptation.register_patch('fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32', npu_matmul_add_fp32)
    MindSporeAdaptation.register_patch('mindspeed.ops.npu_matmul_add.npu_matmul_add_fp32', npu_matmul_add_fp32)

    if args.reuse_fp32_param:
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
        from mindspeed.mindspore.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper
        target_func = DistributedOptimizer.__init__
        target_func_name = 'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__'
        clear_wrapper_v2(target_func_name, target_func)
        MindSporeAdaptation.register_patch(target_func_name, reuse_fp32_param_distrib_optimizer_init_wrapper)

    from mindspeed_llm.mindspore.core.datasets.blended_megatron_dataset_builder import need_to_build_dataset
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.core.datasets.blended_megatron_dataset_builder.need_to_build_dataset',
        need_to_build_dataset)
    from mindspeed.mindspore.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
    MindSporeAdaptation.register_patch(
        'mindspeed.ops.npu_rotary_position_embedding.npu_rotary_position_embedding',
        npu_rotary_position_embedding)

    from mindspeed_llm.mindspore.tasks.checkpoint.models import register_functions, get_modules_from_pretrained
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.tasks.checkpoint.models.ModelBase._ModelBase__register_functions', register_functions)
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.tasks.checkpoint.models.HuggingfaceModel.get_modules_from_pretrained',
        get_modules_from_pretrained)

    from mindspeed.mindspore.legacy.model.module import fp32_to_float16, float16_to_fp32
    MindSporeAdaptation.register_patch('megatron.legacy.model.module.fp32_to_float16', fp32_to_float16)
    MindSporeAdaptation.register_patch('megatron.legacy.model.module.float16_to_fp32', float16_to_fp32)

    from mindspeed_llm.mindspore.core.datasets.blended_megatron_dataset_builder import need_to_build_dataset
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.core.datasets.blended_megatron_dataset_builder.need_to_build_dataset',
        need_to_build_dataset)

    from mindspeed.mindspore.core.transformer.moe.token_dispatcher import preprocess
    MindSporeAdaptation.register_patch('mindspeed.core.transformer.moe.token_dispatcher.preprocess', preprocess)

    from mindspeed.mindspore.core.pipeline_parallel.schedules import deallocate_output_tensor_
    MindSporeAdaptation.register_patch('megatron.core.pipeline_parallel.schedules.deallocate_output_tensor',
                                       deallocate_output_tensor_)

    from .tasks.common.yarn_rope import yarn_linear_ramp_mask
    MindSporeAdaptation.register_patch(
        'mindspeed_llm.tasks.common.yarn_rope.YarnRotaryPositionEmbedding.yarn_linear_ramp_mask', yarn_linear_ramp_mask)

    if args.optimizer_selection == 'fused_ema_adamw':
        from mindspeed.mindspore.ops.npu_apply_fused_ema_adamw import npu_apply_fused_ema_adamw
        MindSporeAdaptation.register_patch('mindspeed.ops.npu_apply_fused_ema_adamw.npu_apply_fused_ema_adamw', npu_apply_fused_ema_adamw, create_dummy=True, force_patch=True)
    elif args.virtual_optimizer is None:
        from mindspeed.mindspore.core.optimizer.adamw import step_func
        MindSporeAdaptation.register_patch('apex.optimizers.FusedAdam.step', step_func)
        MindSporeAdaptation.register_patch('mindspeed.core.optimizer.adamw.AdamW.step', step_func)

        from mindspeed.mindspore.optimizer.adamw import step_func
        MindSporeAdaptation.register_patch('mindspeed.optimizer.adamw.AdamW.step', step_func)

    from mindspeed.mindspore.core.transformer.moe.moe_layer_overlap_all2all import gmm_op
    MindSporeAdaptation.register_patch('mindspeed.core.transformer.moe.moe_feature.overlap.moe_layer_overlap_all2all.gmm_op', gmm_op)

    from mindspeed.mindspore.ops.gmm import _GMM_patched_load
    MindSporeAdaptation.register_patch('mindspeed.op_builder.gmm_builder.GMMOpBuilder.load', _GMM_patched_load)


    from mindspeed.mindspore.core.pipeline_parallel.schedules import custom_backward
    MindSporeAdaptation.register_patch('megatron.core.pipeline_parallel.schedules.custom_backward', custom_backward)

    from mindspeed.mindspore.core.optimizer.optimizer import scale_loss
    MindSporeAdaptation.register_patch('megatron.core.optimizer.optimizer.MegatronOptimizer.scale_loss', scale_loss)

    from torch import npu_apply_fused_adamw_v2
    MindSporeAdaptation.register_patch('mindspeed.ops.npu_apply_fused_adamw_v2.npu_apply_fused_adamw_v2', npu_apply_fused_adamw_v2)

    if args.virtual_optimizer is None:
        from ..mindspore.core.optimizer.distrib_optimizer import load_parameter_state_from_dp_zero
        MindSporeAdaptation.register_patch(
            'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.load_parameter_state_from_dp_zero',
            load_parameter_state_from_dp_zero)

    if args.enable_a2avc:
        from mindspeed.mindspore.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import All2AllSeqTp2epDispatcherImpl
        MindSporeAdaptation.register_patch(
            'mindspeed.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher.All2AllSeqTp2epDispatcherImpl',
            All2AllSeqTp2epDispatcherImpl)

        from mindspeed.mindspore.core.transformer.moe.moe_feature.tp_extend_ep.token_dispatcher import _PatchedMOEAlltoAllSEQTptoEpTokenDispatcher
        MindSporeAdaptation.register_patch(
            'mindspeed.core.transformer.moe.moe_feature.adaptor.MindSpeedMOEAlltoAllSEQTptoEpTokenDispatcher',
            _PatchedMOEAlltoAllSEQTptoEpTokenDispatcher)

        from mindspeed.mindspore.core.tensor_parallel.mappings import all_to_all_forward_a2avc
        MindSporeAdaptation.register_patch('mindspeed_llm.mindspore.core.tensor_parallel.mappings._AllToAll.forward', all_to_all_forward_a2avc)

    

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
    from mindspeed.mindspore.core.optimizer.swap_optimizer.swap_optimizer import opt_states_initialization, \
        create_tensor_maps, swap_adamw_step
    MindSporeAdaptation.register_patch(
        'mindspeed.core.optimizer.swap_optimizer.swap_optimizer.SwapDistributedOptimizer.opt_states_initialization',
        opt_states_initialization)
    MindSporeAdaptation.register_patch(
        'mindspeed.core.optimizer.swap_optimizer.swap_optimizer.SwapDistributedOptimizer.create_tensor_maps',
        create_tensor_maps)
    MindSporeAdaptation.register_patch('mindspeed.optimizer.adamw.AdamW.step', swap_adamw_step)


def patch_moe_fb_overlap():
    pass


def mindspore_register_args(group):
    group.add_argument('--enable-a2avc', action='store_true', default=False,
                       help='enable a2avc')

    pass

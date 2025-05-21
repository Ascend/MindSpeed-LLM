# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved

from mindspeed_llm.tasks.megatron_adaptor import MegatronAdaptation, MegatronAdaptationABC


class MindSporeAdaptation(MegatronAdaptationABC):
    """
    Adaptations for models in Megatron-LM Core structure.
    """
    @classmethod
    def register(cls, orig_func_name, new_func=None, force_patch=True, create_dummy=False, check_patch=False):
        """
        Register adaptations into collection. Force patch for MindSpore patches.
        """
        if check_patch:
            new_func = MindSporeAdaptation.wrap_print_new_func(new_func)
        MegatronAdaptation.register(orig_func_name, new_func, force_patch, create_dummy)

    @classmethod
    def wrap_print_new_func(cls, new_func):
        from functools import wraps

        # wrap the new func with info print
        def make_patch(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"Stepping into MindSpore patch: {func.__name__}")
                return func(*args, **kwargs)

            return wrapper

        # wrap new_func before handing it off to MegatronAdaptation.register
        new_func_with_print = make_patch(new_func)
        return new_func_with_print

    def execute(self):
        args = MegatronAdaptation.get_args()
        if not hasattr(args, "ai_framework") or args.ai_framework != "mindspore":
            return
        from ..core.models.gpt.gpt_model import GPTModel
        from ..mindspore.core.transformer.moe.moe_layer import moe_layer_init_wrapper, moe_layer_forward
        from mindspeed.mindspore.core.data_parallel.distributed_data_parallel import distributed_data_parallel_init_with_cp
        from mindspeed.mindspore.core.transformer.moe.experts import groupedmlp_init_wrapper, groupedmlp_forward

        MindSporeAdaptation.register('megatron.core.models.gpt.gpt_model.GPTModel', GPTModel)
        MindSporeAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                    distributed_data_parallel_init_with_cp)
        MindSporeAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.__init__',
                                moe_layer_init_wrapper)
        MindSporeAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.__init__',
                                groupedmlp_init_wrapper)
        MindSporeAdaptation.register('megatron.core.transformer.moe.moe_layer.MoELayer.forward', moe_layer_forward)

        if args.moe_permutation_async_comm:
            if args.moe_token_dispatcher_type == 'alltoall':
                if args.moe_alltoall_overlap_comm:
                    from mindspeed.mindspore.core.transformer.moe.legacy_a2a_token_dispatcher import alltoall_token_permutation_new, \
                            alltoall_token_unpermutation_new
                    from mindspeed.mindspore.core.transformer.moe.experts import group_mlp_forward
                    MindSporeAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward', group_mlp_forward)
                    MindSporeAdaptation.register(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_permutation',
                        alltoall_token_permutation_new)
                    MindSporeAdaptation.register(
                        'megatron.core.transformer.moe.token_dispatcher.MoEAlltoAllTokenDispatcher.token_unpermutation',
                        alltoall_token_unpermutation_new)

                if hasattr(args, 'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute and not args.moe_expert_capacity_factor:
                    from mindspeed.mindspore.core.fusions.npu_moe_token_permute import permute_wrapper
                    from mindspeed.mindspore.core.fusions.npu_moe_token_unpermute import unpermute_wrapper
                    MindSporeAdaptation.register('megatron.core.transformer.moe.moe_utils.permute', permute_wrapper)
                    MindSporeAdaptation.register('megatron.core.transformer.moe.moe_utils.unpermute', unpermute_wrapper)

        if not args.moe_alltoall_overlap_comm:
            MindSporeAdaptation.register('megatron.core.transformer.moe.experts.GroupedMLP.forward',
                                    groupedmlp_forward)

        from mindspeed.mindspore.core.distributed.distributed_data_parallel import distributed_data_parallel_init, local_make_param_hook
        MindSporeAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__', distributed_data_parallel_init)
        MindSporeAdaptation.register('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_param_hook', local_make_param_hook)

        from mindspeed.mindspore.core.distributed.param_and_grad_buffer import register_grad_ready
        MindSporeAdaptation.register('megatron.core.distributed.param_and_grad_buffer.register_grad_ready', register_grad_ready)

        from mindspeed.mindspore.core.models.common.embeddings.rotary_pos_embedding import get_rotary_seq_len, local_rotate_half
        MindSporeAdaptation.register('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len', get_rotary_seq_len)
        MindSporeAdaptation.register('megatron.core.models.common.embeddings._rotate_half', local_rotate_half)

        from mindspeed.mindspore.core.optimizer import get_megatron_optimizer
        MindSporeAdaptation.register('megatron.core.optimizer.get_megatron_optimizer', get_megatron_optimizer)
        from mindspeed.mindspore.core.optimizer.optimizer import megatron_optimizer_init
        MindSporeAdaptation.register('megatron.core.optimizer.optimizer.MegatronOptimizer.__init__', megatron_optimizer_init)

        from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_step, backward_step, forward_backward_no_pipelining
        MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_step', forward_step)
        MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.backward_step', backward_step)
        MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining', forward_backward_no_pipelining)
        from mindspeed.mindspore.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving, forward_backward_pipelining_without_interleaving
        MindSporeAdaptation.register('megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving', forward_backward_pipelining_without_interleaving)

        from mindspeed.mindspore.core.tensor_parallel.data import local_build_key_size_numel_dictionaries
        MindSporeAdaptation.register('megatron.core.tensor_parallel.data._build_key_size_numel_dictionaries', local_build_key_size_numel_dictionaries) # 1097

        from mindspeed.mindspore.core.tensor_parallel.mappings import all_to_all_forward
        MindSporeAdaptation.register('megatron.core.tensor_parallel.mappings._AllToAll.forward', all_to_all_forward)

        from mindspeed.mindspore.core.tensor_parallel.random import local_set_cuda_rng_state, checkpoint_function_forward, checkpoint_function_backward
        MindSporeAdaptation.register('megatron.core.tensor_parallel.random._set_cuda_rng_state', local_set_cuda_rng_state)
        MindSporeAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.forward', checkpoint_function_forward)
        MindSporeAdaptation.register('megatron.core.tensor_parallel.random.CheckpointFunction.backward', checkpoint_function_backward)

        from ..mindspore.training.utils import get_batch_on_this_tp_rank
        MindSporeAdaptation.register('megatron.training.utils.get_batch_on_this_tp_rank', get_batch_on_this_tp_rank)

        from ..mindspore.core.tensor_parallel.cross_entropy import calculate_predicted_logits, prepare_gradient_calculation_operands
        MindSporeAdaptation.register(
            'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
            calculate_predicted_logits)
        MindSporeAdaptation.register(
            'megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.prepare_gradient_calculation_operands',
            prepare_gradient_calculation_operands)

        from mindspeed.mindspore.core.timers import _get_global_min_max_time
        MindSporeAdaptation.register('megatron.core.timers.Timers._get_global_min_max_time', _get_global_min_max_time)


        from ..mindspore.core.optimizer.distrib_optimizer import get_parameter_state_dp_zero
        MindSporeAdaptation.register('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.get_parameter_state_dp_zero',
                            get_parameter_state_dp_zero)

        if args.async_log_allreduce:
            from mindspeed.mindspore.core.data_parallel.async_log_allreduce import get_async_reduced_loss_value
            MindSporeAdaptation.register('mindspeed.core.data_parallel.async_log_allreduce.get_async_reduced_loss_value',
                                get_async_reduced_loss_value)

        from mindspeed.mindspore.core.tensor_parallel.random import CheckpointWithoutOutput, \
            CheckpointFunctionWithoutOutput
        MindSporeAdaptation.register('mindspeed.core.tensor_parallel.random.CheckpointWithoutOutput',
                                     CheckpointWithoutOutput)
        MindSporeAdaptation.register('mindspeed.core.tensor_parallel.random.CheckpointFunctionWithoutOutput',
                                     CheckpointFunctionWithoutOutput)
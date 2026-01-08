from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


_QUANT_STATE_CHOICES = ('fp8', 'hif8', 'mxfp8')


class LowPrecisionOptimizerFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('low-precision-optimizer', optimization_level=0)

    @staticmethod
    def _normalize_flags(args):
        quant_states = getattr(args, 'quant_states', None)
        if isinstance(quant_states, str):
            quant_states = quant_states.lower()
        if quant_states is not None and quant_states not in _QUANT_STATE_CHOICES:
            raise AssertionError(
                f"Low precision optimizer only supports quant_states {_QUANT_STATE_CHOICES}, got '{quant_states}'."
            )
        quant_states_enabled = bool(quant_states)
        quant_grads_requested = getattr(args, 'quant_grads', False)
        use_quant = quant_states_enabled or quant_grads_requested


        if use_quant:
            args.use_quant_optimizer = True
            args.quant_states = quant_states
            args.quant_states_enabled = quant_states_enabled
            if quant_grads_requested:
                args.quant_grads = True
                quant_grad_dtype = getattr(args, 'quant_grads_dtype', None)
                if quant_grad_dtype in {'fp16'}:
                    args.quant_grads_dtype = quant_grad_dtype
                else:
                    args.quant_grads_dtype = 'fp16'
            else:
                args.quant_grads = False
                args.quant_grads_dtype = None

            return True

        args.quant_states = quant_states
        args.quant_states_enabled = False
        if getattr(args, 'quant_grads', False):
            args.quant_grads = False
        args.quant_grads_dtype = None
        args.use_quant_optimizer = False
        return False

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        option_strings = {
            opt
            for action in parser._actions
            for opt in action.option_strings
        }
        if '--quant-states' not in option_strings:
            group.add_argument('--quant-states', choices=_QUANT_STATE_CHOICES, default=None,
                               help='Select quantization format for optimizer states (default: disabled).')
        if '--quant-grads' not in option_strings:
            group.add_argument('--quant-grads', action='store_true',
                               help='Enable gradient quantization; dtype inferred from main_grads or defaults to fp16.')

    def validate_args(self, args):
        self._normalize_flags(args)
        if (
            getattr(args, "quant_grads", False)
            and getattr(args, "moe_fb_overlap", False)
            and getattr(args, "gradient_accumulation_fusion", False)
        ):
            raise AssertionError(
                "quant_grads is incompatible with MoE fb-overlap + "
                "gradient_accumulation_fusion. Please disable "
                "--no-gradient-accumulation-fusion or turn off --quant-grads."
            )

    def register_patches(self, patch_manager, args):
        quant_enabled = self._normalize_flags(args)

        if not quant_enabled:
            return

        quant_grads_enabled = bool(getattr(args, 'quant_grads', False))
        patch_specs = []
        
        import mindspeed.core.optimizer.low_precision.quant_optimizer_hooks as optimizer_hooks
        import mindspeed.core.optimizer.low_precision.quant_distributed_hooks as distributed_hooks
        from mindspeed.core.optimizer.low_precision import quant_grad_clip as grad_clip
        from mindspeed.core.optimizer.low_precision import param_and_grad_buffer
        from mindspeed.core.optimizer.low_precision import finalize_model_grads as quant_finalize
        from mindspeed.core.models.gpt.gpt_model import gptmodel_init_wrapper
        from mindspeed.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes_wrapper
        from mindspeed.core.optimizer.low_precision.language_model import transformer_language_model_init_wrapper

        patch_specs.extend(
            [
                (
                    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.prepare_grads',
                    optimizer_hooks.prepare_grads_impl,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step_with_ready_grads',
                    optimizer_hooks.step_with_ready_grads_impl,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
                    optimizer_hooks.mixed_precision_optimizer_step_impl,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                    optimizer_hooks.reuse_fp32_param_init_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                    optimizer_hooks.optimizer_config_init_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer.optimizer_config.OptimizerConfig.__post_init__',
                    optimizer_hooks.optimizer_config_post_init_wrapper,
                    True,
                ),
                (
                    'megatron.core.optimizer._get_megatron_optimizer_based_on_param_groups',
                    optimizer_hooks.get_optimizer_builder_wrapper,
                    True,
                ),

                (
                    'megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync',
                    param_and_grad_buffer.quant_grad_start_grad_sync_wrapper,
                    False,
                ),
                (
                    'megatron.core.distributed.param_and_grad_buffer._ParamAndGradBuffer.__init__',
                    param_and_grad_buffer.quant_grad_param_and_grad_buffer_init_wrapper,
                    False,
                ),
                (
                    'megatron.core.distributed.finalize_model_grads._allreduce_position_embedding_grads',
                    quant_finalize._allreduce_position_embedding_grads,
                    False,
                ),
                (
                    'megatron.core.distributed.finalize_model_grads._allreduce_layernorm_grads',
                    quant_finalize._allreduce_layernorm_grads,
                    False,
                ),
                (
                    'megatron.core.distributed.finalize_model_grads._allreduce_conditional_embedding_grads',
                    quant_finalize._allreduce_conditional_embedding_grads,
                    False,
                ),
                (
                    'megatron.core.distributed.finalize_model_grads._update_router_expert_bias',
                    quant_finalize._update_router_expert_bias,
                    False,
                ),
                (
                    'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel._make_backward_post_hook',
                    distributed_hooks.ddp_make_backward_post_hook_wrapper,
                    False,
                ),
                (
                    'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                    optimizer_hooks.distributed_optimizer_init_wrapper,
                    True,
                ),
                (
                    'megatron.core.models.gpt.gpt_model.GPTModel.__init__',
                    gptmodel_init_wrapper,
                    True,
                ),
                (
                    'megatron.core.tensor_parallel.layers.copy_tensor_model_parallel_attributes',
                    copy_tensor_model_parallel_attributes_wrapper,
                    True,
                ),
                (
                    'megatron.legacy.model.language_model.TransformerLanguageModel.__init__',
                    transformer_language_model_init_wrapper,
                    True,
                ),
            ]
        )
        if quant_grads_enabled:
            patch_specs.extend(
                [
                    (
                        'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._collect_main_grad_data_for_unscaling',
                        distributed_hooks.collect_main_grad_data_for_unscaling_quant,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_model_grads_to_main_grads',
                        distributed_hooks.copy_model_grads_to_main_grads_quant,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params._collect_main_grad_data_for_unscaling',
                        optimizer_hooks.collect_main_grad_data_for_unscaling_wrapper,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params._copy_model_grads_to_main_grads',
                        optimizer_hooks.copy_model_grads_to_main_grads,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan',
                        optimizer_hooks.unscale_main_grads_and_check_for_nan,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.optimizer.MegatronOptimizer.get_main_grads_for_grad_norm',
                        optimizer_hooks.get_main_grads_for_grad_norm,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.optimizer._zero_grad_group_helper',
                        optimizer_hooks.zero_grad_group_helper_wrapper,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.clip_grads.get_grad_norm_fp32',
                        grad_clip.get_grad_norm_fp32,
                        True,
                    ),
                    (
                        'megatron.core.optimizer.clip_grads.clip_grad_by_total_norm_fp32',
                        grad_clip.clip_grad_by_total_norm_fp32_wrapper,
                        True,
                    ),
                ]
            )
        for target, func, force in patch_specs:
            patch_manager.register_patch(target, func, force_patch=force)

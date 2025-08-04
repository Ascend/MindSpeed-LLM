from argparse import ArgumentParser

from mindspeed.features_manager.megatron_basic.megatron_basic import MegatronBasicFeature as MindSpeedMegatronBasicFeature


class MegatronBasicFeature(MindSpeedMegatronBasicFeature):

    def register_patches(self, patch_manager, args):
        try:
            import megatron.training
            only_mcore = False
        except ModuleNotFoundError:
            only_mcore = True

        self.register_mcore_basic_patches(patch_manager, args)
        if not only_mcore:
            self.register_non_mcore_basic_patches(patch_manager, args)

    def register_args(self, parser: ArgumentParser):
        super().register_args(parser)
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--stage",
                            default=None,
                            choices=["sft", "dpo", "orm", "prm", "simpo", "ray_ppo", "ray_online_dpo", "ray_grpo", "trl_ppo"],
                            help='Determine training mode')
        group.add_argument('--scale-depth', type=float, default=None,
                            help='scale-depth')
        group.add_argument('--no-shared-storage', action='store_true',
                            default=False, help='if no shared storage, set it')
        group.add_argument('--num-layer-list',
                            type=str, help='a list of number of layers, '
                                        'seperated by comma; e.g., 4,4,4,4')

    def register_mcore_basic_patches(self, pm, args):
        # norm patches
        from mindspeed.core.megatron_basic.megatron_basic import PTNorm
        pm.register_patch('megatron.core.models.gpt.gpt_layer_specs.LNImpl',
                           PTNorm)
        pm.register_patch('megatron.core.transformer.torch_norm.WrappedTorchNorm',
                           PTNorm)
        pm.register_patch('megatron.core.transformer.transformer_block.LayerNormImpl',
                           PTNorm)
        pm.register_patch('megatron.core.extensions.transformer_engine.TENorm',
                           PTNorm)

        # coalescing_manager patches
        from mindspeed.core.distributed.param_and_grad_buffer import start_param_sync, finish_param_sync, start_grad_sync, finish_grad_sync
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_param_sync',
                           start_param_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_param_sync',
                           finish_param_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync',
                           start_grad_sync)
        pm.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_grad_sync',
                           finish_grad_sync)

        # Currently, it is not supported to Cast shard fp32 main params to fp8 model params
        from mindspeed.core.fp8_utils import quantize_param_shard
        pm.register_patch('megatron.core.fp8_utils.quantize_param_shard',
                           quantize_param_shard)

        # fix get_megatron_optimizer for core_r0.12.0
        from mindspeed.core.megatron_basic.get_megatron_optimizer import get_megatron_optimizer
        pm.register_patch('megatron.core.optimizer.get_megatron_optimizer',
                           get_megatron_optimizer)

        # fix count_zeros in ChainedOptimizer for core_r0.12.1.
        from mindspeed.core.megatron_basic.count_zero_fix import step
        pm.register_patch('megatron.core.optimizer.optimizer.ChainedOptimizer.step',
                           step)

        from mindspeed_llm.core.transformer.transformer_block import get_layer_offset_wrapper
        from mindspeed_llm.core import TransformerLayer
        pm.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset',
                            get_layer_offset_wrapper)
        pm.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer', TransformerLayer)

    def register_non_mcore_basic_patches(self, pm, args):
        # args parser patch
        from mindspeed_llm.training.utils import print_args_wrapper
        from mindspeed_llm.training.arguments import validate_args_v2_decorator, parse_args_decorator
        from mindspeed_llm.core.transformer.transformer_config import transformer_config_post_init_wrapper
        pm.register_patch('megatron.training.arguments.parse_args',
                          parse_args_decorator)
        pm.register_patch('megatron.training.arguments.validate_args',
                          validate_args_v2_decorator)
        pm.register_patch('megatron.training.arguments._print_args',
                          print_args_wrapper)
        pm.register_patch('megatron.training.yaml_arguments.validate_yaml',
                          validate_args_v2_decorator)
        pm.register_patch('megatron.training.yaml_arguments._print_args',
                          print_args_wrapper)
        pm.register_patch("megatron.core.transformer.transformer_config.TransformerConfig.__post_init__",
                          transformer_config_post_init_wrapper)

        # initialization patches
        from mindspeed.core.megatron_basic.megatron_basic import _set_cuda_rng_state, _compile_dependencies, \
            get_device_wrapper
        pm.register_patch('megatron.core.tensor_parallel.random._set_cuda_rng_state',
                          _set_cuda_rng_state)
        pm.register_patch('megatron.training.initialize._compile_dependencies',
                          _compile_dependencies)
        pm.register_patch('megatron.training.dist_signal_handler.get_device',
                          get_device_wrapper)

        from mindspeed.core.megatron_basic.megatron_basic import get_device_arch_version
        pm.register_patch('megatron.training.utils.get_device_arch_version',
                          get_device_arch_version)
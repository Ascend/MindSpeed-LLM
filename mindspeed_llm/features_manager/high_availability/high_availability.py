from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class HighAvailabilityFeature(MindSpeedFeature):

    def __init__(self):
        super(HighAvailabilityFeature, self).__init__('high-availability')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--enable-high-availability', action='store_true',
                           help='switch of the high availability feature')
        group.add_argument('--enable-optimizer-state-local-copy', action='store_true',
                           help='high availability feature, enable parameter state local copy of distributed optimizer')
        group.add_argument('--enable-hbmfault-repair', action='store_true',
                           help='high availability feature, enable hbmfault repair')


    def validate_args(self, args):
        if args.enable_high_availability:
            try:
                import mindio_ttp
            except ModuleNotFoundError as e:
                raise AssertionError(f"High availability feature requires the mindio_ttp package but is not installed.") from e
        if args.enable_optimizer_state_local_copy and not args.enable_high_availability:
            raise AssertionError('switch of the local copy is unsupported, please enable high availability feature first.')
        if args.enable_hbmfault_repair and not args.enable_high_availability:
            raise AssertionError(
                'switch of the enable hbmfault repair is unsupported, please enable high availability feature first.')
        if args.enable_high_availability and args.use_dist_ckpt:
            raise AssertionError('switch of the high availability feature is unsupported')


    def register_patches(self, patch_manager, args):
        from training import setup_model_and_optimizer_wrapper
        from core import (get_megatron_optimizer_wrapper, clip_grad_norm_fp32_wrapper,
                            distributed_optimizer_init_wrapper,
                            start_grad_sync_wrapper, distributed_data_parallel_init_wrapper,
                            distributed_optimizer_init_for_reuse_fp32_wrapper,
                            get_parameter_state_dp_zero_with_high_availability_wrapper)

        if args.enable_high_availability:
            patch_manager.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                                        distributed_data_parallel_init_wrapper)
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer.Bucket.start_grad_sync',
                                        start_grad_sync_wrapper)
            patch_manager.register_patch('megatron.training.training.get_megatron_optimizer',
                                        get_megatron_optimizer_wrapper)
            patch_manager.register_patch('megatron.core.optimizer.optimizer.clip_grad_norm_fp32',
                                        clip_grad_norm_fp32_wrapper)
            patch_manager.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                        distributed_optimizer_init_wrapper)
            patch_manager.register_patch('megatron.training.training.setup_model_and_optimizer',
                                        setup_model_and_optimizer_wrapper)
            if args.reuse_fp32_param:
                from mindspeed.optimizer.optimizer import mixed_precision_optimizer_step, reuse_fp32_param_init_wrapper, \
                    optimizer_config_init_wrapper
                patch_manager.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
                                            mixed_precision_optimizer_step)
                patch_manager.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                                            reuse_fp32_param_init_wrapper)
                patch_manager.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                                            optimizer_config_init_wrapper)
                patch_manager.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                            distributed_optimizer_init_for_reuse_fp32_wrapper)
                patch_manager.register_patch('mindio_ttp.adaptor.TTPReplicaOptimizer.get_parameter_state_dp_zero_for_ttp',
                                            get_parameter_state_dp_zero_with_high_availability_wrapper)

from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class HighAvailabilityFeature(MindSpeedFeature):

    def __init__(self):
        super(HighAvailabilityFeature, self).__init__(feature_name='high-availability', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--enable-high-availability', action='store_true',
                            help='switch of the high availability feature')
        group.add_argument('--enable-hbmfault-repair', action='store_true',
                            help='high availability feature, enable hbmfault repair')
        group.add_argument('--enable-worker-reboot', action='store_true',
                            help='high availability feature, enable worker reboot')
        group.add_argument('--distributed-optimizer-no-replica', action='store_true',
                            help='high availability feature, repair from ckpt and disable replica optimizer')

    def validate_args(self, args):
        if args.enable_high_availability:
            try:
                import mindio_ttp
            except ModuleNotFoundError as e:
                raise AssertionError(
                    f"High availability feature requires the mindio_ttp package but is not installed.") from e
        if args.enable_hbmfault_repair and not args.enable_high_availability:
            raise AssertionError(
                'switch of the enable hbmfault repair is unsupported, please enable high availability feature first.')
        if args.enable_high_availability and args.use_dist_ckpt:
            raise AssertionError('switch of the high availability feature is unsupported')
        if args.swap_optimizer and args.enable_high_availability:
            raise AssertionError('switch of the high availability feature is unsupported')

    def register_patches(self, patch_manager, args):
        from .initialize_patch import setup_model_and_optimizer_wrapper, initialize_distributed_wrapper
        from mindspeed_llm.core import (start_grad_sync_wrapper, distributed_data_parallel_init_wrapper,
                                        start_param_sync_wrapper, param_and_grad_bucket_group_init_wrapper,
                                        get_megatron_optimizer_wrapper, get_grad_norm_fp32_wrapper,
                                        distributed_optimizer_init_wrapper,
                                        distributed_optimizer_init_for_reuse_fp32_wrapper,
                                        get_parameter_state_dp_zero_with_high_availability_wrapper)
        from mindspeed_llm.core.pipeline_parallel.schedules import high_availability_get_forward_backward_func_wrapper

        if args.enable_high_availability:
            patch_manager.register_patch('megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.__init__',
                                          distributed_data_parallel_init_wrapper)
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync',
                                          start_grad_sync_wrapper)
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.__init__',
                                          param_and_grad_bucket_group_init_wrapper)
            patch_manager.register_patch('megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_param_sync',
                                          start_param_sync_wrapper)
            patch_manager.register_patch('megatron.training.training.get_megatron_optimizer',
                                          get_megatron_optimizer_wrapper)
            patch_manager.register_patch('megatron.training.initialize._initialize_distributed',
                                          initialize_distributed_wrapper)
            patch_manager.register_patch('megatron.core.optimizer.clip_grads.get_grad_norm_fp32',
                                          get_grad_norm_fp32_wrapper)
            patch_manager.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                          distributed_optimizer_init_wrapper)
            patch_manager.register_patch('megatron.training.training.setup_model_and_optimizer',
                                          setup_model_and_optimizer_wrapper)
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                          high_availability_get_forward_backward_func_wrapper)
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
            if args.enable_worker_reboot:
                from .initialize_patch import build_train_valid_test_data_iterators_wrapper
                from mindspeed_llm.features_manager.high_availability.communication_patch import new_group_wrapper
                patch_manager.register_patch('megatron.training.training.build_train_valid_test_data_iterators',
                                              build_train_valid_test_data_iterators_wrapper)
                patch_manager.register_patch('torch.distributed.distributed_c10d.new_group',
                                              new_group_wrapper)


class HighAvailabilityCommFeature(MindSpeedFeature):
    def __init__(self):
        super(HighAvailabilityCommFeature, self).__init__(feature_name='high-availability-comm', optimization_level=0)

    def pre_patch(self, patch_manager, args):
        from mindspeed_llm.features_manager.high_availability.communication_patch import communication_wrapper
        for communication in ['barrier', 'all_reduce', '_all_gather_base', 'broadcast', 'all_gather_into_tensor']:
            patch_manager.register_patch('torch.distributed.distributed_c10d.' + communication,
                                          communication_wrapper)

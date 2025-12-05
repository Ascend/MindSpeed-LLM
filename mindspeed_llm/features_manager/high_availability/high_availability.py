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
        group.add_argument('--enable-elastic-training', action='store_true',
                           help='high availability feature, enable elastic training')

    def pre_validate_args(self, args):
        from mindspeed_llm.tasks.high_availability.high_availability_helper import get_env_args
        get_env_args(args)

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
        if args.enable_high_availability and args.swap_attention:
            raise AssertionError(
                'switch of the high availability feature is unsupported, please disable swap attention first.')
        if args.swap_optimizer and args.enable_high_availability:
            raise AssertionError('switch of the high availability feature is unsupported')
        if args.enable_elastic_training and not args.enable_high_availability:
            raise AssertionError(
                'switch of the enable elastic training is unsupported, please enable high availability feature first.')
        if args.enable_elastic_training and not args.use_distributed_optimizer:
            raise AssertionError(
                'switch of the enable elastic training is unsupported, please enable use-distributed-optimizer first.')
        if args.enable_elastic_training and args.use_custom_fsdp:
            raise AssertionError(
                'switch of the enable elastic training is unsupported when reuse-fp32-param is enabled.')
        if args.enable_elastic_training and args.reuse_fp32_param:
            raise AssertionError(
                'switch of the enable elastic training is unsupported when reuse-fp32-param is enabled.')
        if args.enable_elastic_training and (args.expert_model_parallel_size > 1 or args.context_parallel_size > 1):
            raise AssertionError(
                'switch of the enable elastic training is unsupported when expert-model-parallel-size, context '
                'parallel size is set.')

    def pre_register_patches(self, patch_manager, args):
        from mindspeed_llm.tasks.high_availability.communication_patch import communication_wrapper, barrier_wrapper
        from mindspeed_llm.tasks.high_availability.high_availability_helper import skip_reuse_register_patches
        patch_manager.register_patch('torch.distributed.barrier',
                                     barrier_wrapper)
        for communication in ['all_reduce', '_all_gather_base', 'broadcast', 'all_gather_into_tensor']:
            patch_manager.register_patch('torch.distributed.distributed_c10d.' + communication,
                                         communication_wrapper)
        from mindspeed_llm.tasks.high_availability.communication_patch import (group_index_two_torch_wrapper,
                                                                               group_index_three_torch_wrapper, all_to_all_single_wrapper)
        patch_manager.register_patch('torch.distributed.all_to_all_single',
                                     all_to_all_single_wrapper)
        for communication in ['all_gather', 'all_to_all', 'all_reduce_coalesced', 'all_gather_object',
                              'broadcast_object_list', 'all_gather_coalesced', 'irecv', 'isend']:
            patch_manager.register_patch('torch.distributed.' + communication,
                                         group_index_two_torch_wrapper)
        for communication in ['gather', 'scatter', 'reduce', 'reduce_scatter', 'gather_object',
                              'scatter_object_list', 'reduce_scatter_tensor', '_reduce_scatter_base']:
            patch_manager.register_patch('torch.distributed.' + communication,
                                         group_index_three_torch_wrapper)
        from mindspeed.features_manager import ReuseFP32Param
        ReuseFP32Param.register_patches = skip_reuse_register_patches(ReuseFP32Param.register_patches, args)

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.tasks.high_availability.initialize_patch import initialize_distributed_wrapper
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
            patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                          high_availability_get_forward_backward_func_wrapper)
            if args.reuse_fp32_param:
                from mindspeed.core.memory.reuse_param.adaptor import reuse_fp32_param_init_wrapper, optimizer_config_init_wrapper
                patch_manager.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
                                              reuse_fp32_param_init_wrapper)
                patch_manager.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
                                              optimizer_config_init_wrapper)
                patch_manager.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
                                              distributed_optimizer_init_for_reuse_fp32_wrapper)
                patch_manager.register_patch('mindspeed_llm.core.high_availability.TTPReplicaOptimizer.get_parameter_state_dp_zero_for_ttp',
                                              get_parameter_state_dp_zero_with_high_availability_wrapper)
            if args.enable_worker_reboot or args.enable_elastic_training:
                from mindspeed_llm.tasks.high_availability.initialize_patch import build_train_valid_test_data_iterators_wrapper
                from mindspeed_llm.tasks.high_availability.communication_patch import new_group_wrapper
                patch_manager.register_patch('megatron.training.training.build_train_valid_test_data_iterators',
                                              build_train_valid_test_data_iterators_wrapper)
                patch_manager.register_patch('torch.distributed.distributed_c10d.new_group',
                                              new_group_wrapper)
            if args.enable_elastic_training:
                from mindspeed_llm.core.pipeline_parallel.schedules import forward_step_wrapper
                from mindspeed_llm.core.optimizer.distrib_optimizer import get_parameter_state_dp_zero_wrapper
                from mindspeed_llm.core.timers import patch_world_size_func_wrapper, log_wrapper
                from mindspeed_llm.training.utils import is_last_rank_wrapper, print_rank_last_wrapper
                from mindspeed_llm.core.optimizer_param_scheduler import optimizer_param_scheduler_step_wrapper
                from mindspeed_llm.core.pipeline_parallel.schedules import (
                    elastic_training_get_forward_backward_func_wrapper)
                from mindspeed_llm.training.training import num_floating_point_operations_wrapper
                from mindspeed_llm.training.one_logger_utils import track_app_tag_wrapper
                patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.forward_step',
                                             forward_step_wrapper)
                patch_manager.register_patch(
                    'megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.get_parameter_state_dp_zero',
                    get_parameter_state_dp_zero_wrapper)
                patch_manager.register_patch('megatron.core.timers.Timers._get_elapsed_time_all_ranks',
                                             patch_world_size_func_wrapper)
                patch_manager.register_patch('megatron.core.timers.Timers._get_all_ranks_time_string',
                                             patch_world_size_func_wrapper)
                patch_manager.register_patch('megatron.core.timers.Timers.log',
                                             log_wrapper)
                patch_manager.register_patch('megatron.training.utils.is_last_rank',
                                             is_last_rank_wrapper)
                patch_manager.register_patch('megatron.core.optimizer_param_scheduler.OptimizerParamScheduler.step',
                                             optimizer_param_scheduler_step_wrapper)
                patch_manager.register_patch('megatron.core.pipeline_parallel.schedules.get_forward_backward_func',
                                             elastic_training_get_forward_backward_func_wrapper)
                patch_manager.register_patch('megatron.training.one_logger_utils.track_app_tag',
                                             track_app_tag_wrapper)
                patch_manager.register_patch('megatron.training.training.num_floating_point_operations',
                                             num_floating_point_operations_wrapper)
                patch_manager.register_patch('megatron.training.utils.print_rank_last',
                                             print_rank_last_wrapper)






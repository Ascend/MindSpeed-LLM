import unittest
from unittest.mock import patch, MagicMock

from megatron.core.enums import ModelType

from mindspeed_llm.core.layerwise_disaggregated_training.training import (
    get_model,
    train_step
)


class TestTraining(unittest.TestCase):
    def setUp(self):
        # 重置全局变量状态
        pass
    
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.dataclasses.fields')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.mpu')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_model_config')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.Float16Module')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.correct_amax_history_if_needed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.DDP')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.DistributedDataParallelConfig')
    def test_get_model(self, mock_ddp_config, mock_ddp, mock_correct_amax, mock_float16, mock_get_model_config, mock_mpu, mock_get_args, mock_fields):
        """测试get_model函数"""
        # 模拟args
        class MockArgs:
            virtual_pipeline_model_parallel_size = None
            init_model_with_meta_device = False
            fp16 = False
            bf16 = False
            use_torch_fsdp2 = False
            use_custom_fsdp = False
            data_parallel_random_init = False
            layerwise_disaggregated_training = False
            accumulate_allreduce_grads_in_fp32 = False
            check_for_nan_in_loss_and_grad = False
            check_for_large_grads = False
            ddp_num_buckets = None
            ddp_bucket_size = None
            ddp_pad_buckets_for_high_nccl_busbw = False
            ddp_average_in_collective = False
            use_precision_aware_optimizer = False
            overlap_grad_reduce = True
            overlap_param_gather_with_optimizer_step = False
        
        mock_args = MockArgs()
        mock_get_args.return_value = mock_args
        
        # 模拟dataclasses.fields的返回值
        class MockField:
            def __init__(self, name):
                self.name = name
        
        mock_fields.return_value = [MockField('overlap_grad_reduce'), MockField('bucket_size'), MockField('ddp_pad_buckets_for_high_nccl_busbw'), MockField('ddp_average_in_collective')]
        
        # 模拟mpu
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
        mock_mpu.is_pipeline_first_stage.return_value = True
        mock_mpu.is_pipeline_last_stage.return_value = True
        mock_mpu.get_data_parallel_rank.return_value = 0
        mock_mpu.get_tensor_model_parallel_rank.return_value = 0
        mock_mpu.get_pipeline_model_parallel_rank.return_value = 0
        
        # 模拟模型提供者函数
        def mock_model_provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
            model = MagicMock()
            model.parameters.return_value = []
            return model
        
        # 测试基本情况
        model = get_model(mock_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)
        self.assertIsInstance(model, list)
        self.assertEqual(len(model), 1)
    
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_timers')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_forward_backward_func')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_num_microbatches')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.logical_and_across_model_parallel_group')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.reduce_max_stat_across_model_parallel_group')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.mpu')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.unwrap_model')
    def test_train_step(self, mock_unwrap_model, mock_mpu, mock_reduce_max, mock_logical_and, mock_get_num_microbatches, mock_get_forward_backward, mock_get_rerun_state, mock_get_timers, mock_get_args):
        """测试train_step函数"""
        # 模拟args
        class MockArgs:
            curr_iteration = 0
            iteration = 0
            external_cuda_graph = False
            layerwise_disaggregated_training = False
            seq_length = 1024
            micro_batch_size = 4
            decoder_seq_length = 1024
            empty_unused_memory_level = 0
            vision_pretraining = False
            log_num_zeros_in_grad = False
            data_parallel_size = 1
            barrier_with_L1_time = False
            use_distributed_optimizer = False
            overlap_param_gather = False
        
        mock_args = MockArgs()
        mock_get_args.return_value = mock_args
        
        # 模拟timers
        mock_timer = MagicMock()
        mock_timer.start = MagicMock()
        mock_timer.stop = MagicMock()
        mock_timers = MagicMock()
        mock_timers.__call__ = MagicMock(return_value=mock_timer)
        mock_get_timers.return_value = mock_timers
        
        # 模拟rerun_state_machine
        mock_rerun_state = MagicMock()
        mock_rerun_state.should_run_forward_backward.side_effect = [True, False]  # 第一次返回True，第二次返回False
        mock_rerun_state.should_checkpoint_and_exit.return_value = (False, False, 0)
        mock_get_rerun_state.return_value = mock_rerun_state
        
        # 模拟forward_backward_func
        mock_forward_backward = MagicMock()
        # 返回一个包含至少一个字典的列表，这样losses_reduced[0]就不会引发索引错误
        mock_forward_backward.return_value = [{'loss': 1.0}]
        mock_get_forward_backward.return_value = mock_forward_backward
        
        # 模拟get_num_microbatches
        mock_get_num_microbatches.return_value = 1
        
        # 模拟logical_and_across_model_parallel_group
        mock_logical_and.return_value = True
        
        # 模拟reduce_max_stat_across_model_parallel_group
        mock_reduce_max.return_value = 0.0
        
        # 模拟mpu
        mock_mpu.is_pipeline_last_stage.return_value = True
        mock_mpu.is_pipeline_first_stage.return_value = False
        
        # 模拟optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.zero_grad = MagicMock()
        mock_optimizer.step.return_value = (True, 0.0, 0)
        
        # 模拟opt_param_scheduler
        mock_opt_param_scheduler = MagicMock()
        mock_opt_param_scheduler.step = MagicMock()
        
        # 模拟config
        class MockConfig:
            layerwise_disaggregated_training = False
        
        mock_config = MockConfig()
        
        # 模拟model
        mock_model = [MagicMock()]
        mock_model[0].zero_grad_buffer = MagicMock()
        
        # 测试基本情况
        result = train_step(
            forward_step_func=MagicMock(),
            data_iterator=MagicMock(),
            model=mock_model,
            optimizer=mock_optimizer,
            opt_param_scheduler=mock_opt_param_scheduler,
            config=mock_config
        )
        
        # 验证返回值结构
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)
    
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_timers')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_forward_backward_func')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.get_num_microbatches')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.logical_and_across_model_parallel_group')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.reduce_max_stat_across_model_parallel_group')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.training.mpu')
    def test_train_step_with_layerwise_disaggregated_training(self, mock_mpu, mock_reduce_max, mock_logical_and, mock_get_num_microbatches, mock_get_forward_backward, mock_get_rerun_state, mock_get_timers, mock_get_args):
        """测试启用layerwise_disaggregated_training的情况"""
        # 模拟args
        class MockArgs:
            curr_iteration = 0
            iteration = 0
            external_cuda_graph = False
            layerwise_disaggregated_training = True
            seq_length = 1024
            micro_batch_size = 4
            decoder_seq_length = 1024
            empty_unused_memory_level = 0
            vision_pretraining = False
            log_num_zeros_in_grad = False
            data_parallel_size = 1
            barrier_with_L1_time = False
            use_distributed_optimizer = False
            overlap_param_gather = False
        
        mock_args = MockArgs()
        mock_get_args.return_value = mock_args
        
        # 模拟timers
        mock_timer = MagicMock()
        mock_timer.start = MagicMock()
        mock_timer.stop = MagicMock()
        mock_timers = MagicMock()
        mock_timers.__call__ = MagicMock(return_value=mock_timer)
        mock_get_timers.return_value = mock_timers
        
        # 模拟rerun_state_machine
        mock_rerun_state = MagicMock()
        mock_rerun_state.should_run_forward_backward.side_effect = [True, False]  # 第一次返回True，第二次返回False
        mock_rerun_state.should_checkpoint_and_exit.return_value = (False, False, 0)
        mock_get_rerun_state.return_value = mock_rerun_state
        
        # 模拟forward_backward_func
        mock_forward_backward = MagicMock()
        # 模拟losses_reduced返回值
        mock_losses = [{'loss': 1.0}]
        mock_forward_backward.return_value = mock_losses
        mock_get_forward_backward.return_value = mock_forward_backward
        
        # 模拟get_num_microbatches
        mock_get_num_microbatches.return_value = 1
        
        # 模拟logical_and_across_model_parallel_group
        mock_logical_and.return_value = True
        
        # 模拟reduce_max_stat_across_model_parallel_group
        mock_reduce_max.return_value = 0.0
        
        # 模拟mpu
        mock_mpu.is_pipeline_first_stage.return_value = True
        
        # 模拟optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.zero_grad = MagicMock()
        mock_optimizer.step.return_value = (True, 0.0, 0)
        
        # 模拟opt_param_scheduler
        mock_opt_param_scheduler = MagicMock()
        mock_opt_param_scheduler.step = MagicMock()
        
        # 模拟config
        class MockConfig:
            layerwise_disaggregated_training = True
        
        mock_config = MockConfig()
        
        # 模拟model
        mock_model = [MagicMock()]
        mock_model[0].zero_grad_buffer = MagicMock()
        
        # 测试启用layerwise_disaggregated_training的情况
        result = train_step(
            forward_step_func=MagicMock(),
            data_iterator=MagicMock(),
            model=mock_model,
            optimizer=mock_optimizer,
            opt_param_scheduler=mock_opt_param_scheduler,
            config=mock_config
        )
        
        # 验证返回值结构
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 7)
        # 验证损失值被正确计算
        self.assertIn('loss', result[0])

if __name__ == '__main__':
    unittest.main()

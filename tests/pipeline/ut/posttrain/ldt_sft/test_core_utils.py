import unittest
from unittest import mock
import torch

# Now import the functions to test
from mindspeed_llm.core.layerwise_disaggregated_training.utils import (
    _allreduce_model_parallel,
    vtp_reduce_max_stat_across_model_parallel_group,
    vtp_logical_and_across_model_parallel_group,
    vtp_get_grad_norm_fp32,
    vtp_timer_barrier_wrapper,
    vtp_all_gather_into_tensor_wrapper
)


def identity(x):
    return x


class TestCoreUtils(unittest.TestCase):
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.torch.distributed.all_reduce')
    def test_allreduce_model_parallel_non_vtp(self, mock_all_reduce):
        # Setup mock
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        op = torch.distributed.ReduceOp.SUM
        group = mock.MagicMock()
        
        # Call the function
        _allreduce_model_parallel(tensor, op, group)
        
        # Verify all_reduce was called
        mock_all_reduce.assert_called_once_with(tensor, op=op, group=group)
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.parallel_state.is_vtp_enabled')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.parallel_state.vtp_allreduce')
    def test_allreduce_model_parallel_vtp(self, mock_vtp_allreduce, mock_is_vtp):
        # Setup mock
        tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        op = torch.distributed.ReduceOp.SUM
        group = mock.MagicMock()
        
        # Mock VTP enabled
        mock_is_vtp.return_value = True
        
        # Call the function
        _allreduce_model_parallel(tensor, op, group)
        
        # Verify vtp_allreduce was called
        mock_vtp_allreduce.assert_called_once_with(tensor, op=op)
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils._allreduce_model_parallel')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.mpu.get_model_parallel_group')
    def test_vtp_reduce_max_stat_across_model_parallel_group_none(self, mock_get_group, mock_allreduce):
        # Setup mock
        mock_get_group.return_value = mock.MagicMock()
        
        # Mock tensor item return value
        mock_tensor = mock.MagicMock()
        mock_tensor.item.return_value = -1.0
        with mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.torch.tensor', return_value=mock_tensor):
            # Call the function with None
            result = vtp_reduce_max_stat_across_model_parallel_group(None)
            
            # Verify the result
            self.assertIsNone(result)
            mock_allreduce.assert_called_once()
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils._allreduce_model_parallel')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.mpu.get_model_parallel_group')
    def test_vtp_reduce_max_stat_across_model_parallel_group_value(self, mock_get_group, mock_allreduce):
        # Setup mock
        mock_get_group.return_value = mock.MagicMock()
        
        # Mock tensor item return value
        mock_tensor = mock.MagicMock()
        mock_tensor.item.return_value = 5.0
        with mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.torch.tensor', return_value=mock_tensor):
            # Call the function with a value
            result = vtp_reduce_max_stat_across_model_parallel_group(5.0)
            
            # Verify the result
            self.assertEqual(result, 5.0)
            mock_allreduce.assert_called_once()
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils._allreduce_model_parallel')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.mpu.get_model_parallel_group')
    def test_vtp_logical_and_across_model_parallel_group_true(self, mock_get_group, mock_allreduce):
        # Setup mock
        mock_get_group.return_value = mock.MagicMock()
        
        # Mock tensor item return value
        mock_tensor = mock.MagicMock()
        mock_tensor.item.return_value = 1
        with mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.torch.tensor', return_value=mock_tensor):
            # Call the function with True
            result = vtp_logical_and_across_model_parallel_group(True)
            
            # Verify the result
            self.assertTrue(result)
            mock_allreduce.assert_called_once()
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils._allreduce_model_parallel')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.mpu.get_model_parallel_group')
    def test_vtp_logical_and_across_model_parallel_group_false(self, mock_get_group, mock_allreduce):
        # Setup mock
        mock_get_group.return_value = mock.MagicMock()
        
        # Mock tensor item return value
        mock_tensor = mock.MagicMock()
        mock_tensor.item.return_value = 0
        with mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.torch.tensor', return_value=mock_tensor):
            # Call the function with False
            result = vtp_logical_and_across_model_parallel_group(False)
            
            # Verify the result
            self.assertFalse(result)
            mock_allreduce.assert_called_once()
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils._allreduce_model_parallel')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.get_data_parallel_group_if_dtensor')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.to_local_if_dtensor')
    def test_vtp_get_grad_norm_fp32_inf(self, mock_to_local, mock_get_data_group, mock_allreduce):
        # Setup mock
        mock_to_local.side_effect = identity
        mock_get_data_group.return_value = None
        
        # Create test data
        grads_for_norm = [torch.tensor([1.0, 2.0, 3.0], device='cuda')]
        
        # Call the function with norm_type=inf
        result = vtp_get_grad_norm_fp32(grads_for_norm, norm_type=float('inf'))
        
        # Verify the result
        self.assertIsInstance(result, float)
        mock_allreduce.assert_called_once()
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils._allreduce_model_parallel')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.get_data_parallel_group_if_dtensor')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.utils.to_local_if_dtensor')
    def test_vtp_get_grad_norm_fp32_other(self, mock_to_local, mock_get_data_group, mock_allreduce):
        # Setup mock
        mock_to_local.side_effect = identity
        mock_get_data_group.return_value = None
        
        # Create test data
        grads_for_norm = [torch.tensor([1.0, 2.0, 3.0], device='cuda')]
        
        # Call the function with norm_type=1
        result = vtp_get_grad_norm_fp32(grads_for_norm, norm_type=1.0)
        
        # Verify the result
        self.assertIsInstance(result, float)
        mock_allreduce.assert_called_once()
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.parallel_state.is_vtp_enabled')
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.parallel_state.vtp_hierarchical_barrier')
    def test_vtp_timer_barrier_wrapper_vtp(self, mock_hierarchical_barrier, mock_is_vtp):
        # Setup mock
        original_barrier = mock.MagicMock()
        mock_is_vtp.return_value = True
        
        # Create wrapper
        wrapper = vtp_timer_barrier_wrapper(original_barrier)
        
        # Call wrapper with group=None
        result = wrapper(group=None)
        
        # Verify vtp_hierarchical_barrier was called
        mock_hierarchical_barrier.assert_called_once()
        self.assertIsNone(result)
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.parallel_state.is_vtp_enabled')
    def test_vtp_timer_barrier_wrapper_non_vtp(self, mock_is_vtp):
        # Setup mock
        original_barrier = mock.MagicMock()
        original_barrier.return_value = "barrier_result"
        mock_is_vtp.return_value = False
        
        # Create wrapper
        wrapper = vtp_timer_barrier_wrapper(original_barrier)
        
        # Call wrapper with group=None
        result = wrapper(group=None)
        
        # Verify original_barrier was called
        original_barrier.assert_called_once_with(group=None)
        self.assertEqual(result, "barrier_result")
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.parallel_state.is_vtp_enabled')
    def test_vtp_all_gather_into_tensor_wrapper_vtp(self, mock_is_vtp):
        # Setup mock
        original_all_gather = mock.MagicMock()
        original_all_gather.return_value = "gather_result"
        mock_is_vtp.return_value = True
        
        # Create wrapper
        wrapper = vtp_all_gather_into_tensor_wrapper(original_all_gather)
        
        # Call wrapper with group=None
        output_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        input_tensor = torch.tensor([1.0], device='cuda')
        result = wrapper(output_tensor, input_tensor, group=None, async_op=False)
        
        # Verify original_all_gather was called
        original_all_gather.assert_called_once_with(output_tensor, input_tensor, group=None, async_op=False)
        self.assertEqual(result, "gather_result")
    
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.parallel_state.is_vtp_enabled')
    def test_vtp_all_gather_into_tensor_wrapper_non_vtp(self, mock_is_vtp):
        # Setup mock
        original_all_gather = mock.MagicMock()
        original_all_gather.return_value = "gather_result"
        mock_is_vtp.return_value = False
        
        # Create wrapper
        wrapper = vtp_all_gather_into_tensor_wrapper(original_all_gather)
        
        # Call wrapper with group=None
        output_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        input_tensor = torch.tensor([1.0], device='cuda')
        result = wrapper(output_tensor, input_tensor, group=None, async_op=False)
        
        # Verify original_all_gather was called
        original_all_gather.assert_called_once_with(output_tensor, input_tensor, group=None, async_op=False)
        self.assertEqual(result, "gather_result")

if __name__ == '__main__':
    unittest.main()

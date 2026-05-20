import unittest
from unittest import mock

from mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel import (
    finish_grad_sync,
    finish_grad_sync_ldt,
    register_grad_ready,
)


class TestDistributedDataParallel(unittest.TestCase):
    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.finish_grad_sync_ldt')
    def test_finish_grad_sync_empty_buckets(self, mock_finish_grad_sync_ldt):
        mock_self = mock.MagicMock()
        mock_self.bucket_groups = []
        mock_self.expert_parallel_bucket_groups = []

        finish_grad_sync(mock_self)

        mock_finish_grad_sync_ldt.assert_not_called()

    @mock.patch('mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.finish_grad_sync_ldt')
    def test_finish_grad_sync_with_buckets(self, mock_finish_grad_sync_ldt):
        mock_self = mock.MagicMock()
        bucket_group1 = mock.MagicMock()
        bucket_group2 = mock.MagicMock()
        expert_bucket_group = mock.MagicMock()
        mock_self.bucket_groups = [bucket_group1, bucket_group2]
        mock_self.expert_parallel_bucket_groups = [expert_bucket_group]

        finish_grad_sync(mock_self)

        self.assertEqual(mock_finish_grad_sync_ldt.call_count, 3)
        mock_finish_grad_sync_ldt.assert_any_call(bucket_group1)
        mock_finish_grad_sync_ldt.assert_any_call(bucket_group2)
        mock_finish_grad_sync_ldt.assert_any_call(expert_bucket_group)

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_finish_grad_sync_ldt_pipeline_first_stage(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = True

        bucket_group = mock.MagicMock()

        finish_grad_sync_ldt(bucket_group)

        bucket_group.start_grad_sync.assert_not_called()

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_finish_grad_sync_ldt_non_overlap_grad_reduce(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        bucket_group = mock.MagicMock()
        bucket_group.ddp_config.overlap_grad_reduce = False

        finish_grad_sync_ldt(bucket_group)

        self.assertFalse(bucket_group.param_gather_dispatched)
        bucket_group.start_grad_sync.assert_called_once()

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.torch.cuda.default_stream'
    )
    def test_finish_grad_sync_ldt_multi_dist_optimizer(self, mock_default_stream, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        bucket_group = mock.MagicMock()
        bucket_group.ddp_config.overlap_grad_reduce = True
        bucket_group.ddp_config.num_distributed_optimizer_instances = 2

        finish_grad_sync_ldt(bucket_group)

        mock_default_stream.return_value.wait_stream.assert_called_once_with(bucket_group.communication_stream)
        bucket_group.start_grad_sync.assert_not_called()

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_finish_grad_sync_ldt_handle_wait(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        bucket_group = mock.MagicMock()
        bucket_group.ddp_config.overlap_grad_reduce = True
        bucket_group.ddp_config.num_distributed_optimizer_instances = 1
        mock_handle = mock.MagicMock()
        bucket_group.grad_reduce_handle = [mock_handle]

        finish_grad_sync_ldt(bucket_group)

        mock_handle.wait.assert_called_once()
        self.assertIsNone(bucket_group.grad_reduce_handle)

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_finish_grad_sync_ldt_no_handle(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        bucket_group = mock.MagicMock()
        bucket_group.ddp_config.overlap_grad_reduce = True
        bucket_group.ddp_config.num_distributed_optimizer_instances = 1
        bucket_group.grad_reduce_handle = None
        bucket_group.params_with_grad = []
        bucket_group.params = []

        with self.assertRaises(AssertionError):
            finish_grad_sync_ldt(bucket_group)

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_register_grad_ready_pipeline_first_stage(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = True

        mock_self = mock.MagicMock()
        param = mock.MagicMock()

        register_grad_ready(mock_self, param)

        mock_self.params_with_grad.add.assert_not_called()

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_register_grad_ready_not_overlap_grad_reduce(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        mock_self = mock.MagicMock()
        mock_self.ddp_config.overlap_grad_reduce = False
        param = mock.MagicMock()

        with self.assertRaises(AssertionError):
            register_grad_ready(mock_self, param)

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_register_grad_ready_not_last_microbatch(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        mock_self = mock.MagicMock()
        mock_self.ddp_config.overlap_grad_reduce = True
        mock_self.is_last_microbatch = False
        param = mock.MagicMock()

        register_grad_ready(mock_self, param)

        mock_self.params_with_grad.add.assert_not_called()

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_register_grad_ready_param_not_in_bucket(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        mock_self = mock.MagicMock()
        mock_self.ddp_config.overlap_grad_reduce = True
        mock_self.is_last_microbatch = True
        mock_self.param_to_bucket = {}
        param = mock.MagicMock()

        with self.assertRaises(AssertionError):
            register_grad_ready(mock_self, param)

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_register_grad_ready_grad_twice(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        mock_self = mock.MagicMock()
        mock_self.ddp_config.overlap_grad_reduce = True
        mock_self.is_last_microbatch = True
        param = mock.MagicMock()
        mock_self.param_to_bucket = {param: mock.MagicMock()}
        mock_self.params_with_grad = {param}

        with self.assertRaises(AssertionError):
            register_grad_ready(mock_self, param)

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_register_grad_ready_not_all_params_ready(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        mock_self = mock.MagicMock()
        mock_self.ddp_config.overlap_grad_reduce = True
        mock_self.is_last_microbatch = True
        param = mock.MagicMock()
        mock_self.param_to_bucket = {param: mock.MagicMock()}
        mock_self.params_with_grad = mock.MagicMock()
        mock_self.params_with_grad.__len__.return_value = 1
        mock_self.params = [param, mock.MagicMock()]

        register_grad_ready(mock_self, param)

        mock_self.params_with_grad.add.assert_called_once_with(param)
        mock_self.start_grad_sync.assert_not_called()

    @mock.patch(
        'mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel.parallel_state.is_pipeline_first_stage'
    )
    def test_register_grad_ready_all_params_ready(self, mock_is_pipeline_first_stage):
        mock_is_pipeline_first_stage.return_value = False

        mock_self = mock.MagicMock()
        mock_self.ddp_config.overlap_grad_reduce = True
        mock_self.is_last_microbatch = True
        param = mock.MagicMock()
        mock_self.param_to_bucket = {param: mock.MagicMock()}
        mock_self.params_with_grad = mock.MagicMock()
        mock_self.params_with_grad.__len__.return_value = 1
        mock_self.params = [param]

        register_grad_ready(mock_self, param)

        mock_self.params_with_grad.add.assert_called_once_with(param)
        mock_self.start_grad_sync.assert_called_once()


if __name__ == '__main__':
    unittest.main()

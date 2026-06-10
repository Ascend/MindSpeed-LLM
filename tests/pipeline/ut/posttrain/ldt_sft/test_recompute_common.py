import unittest
from unittest.mock import patch, MagicMock

from mindspeed_llm.core.layerwise_disaggregated_training.recompute_common import should_recompute


class TestShouldRecompute(unittest.TestCase):
    """Unit tests for should_recompute in recompute_common.py."""

    def setUp(self):
        self.config = MagicMock()
        self.config.virtual_pipeline_model_parallel_size = 2
        self.config.enable_recompute_layers_per_pp_rank = True
        self.config.num_layer_list = [(3, 0), (3, 0)]
        self.config.recompute_num_layers = None

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_vpp_rank_none_defaults_to_zero(self, mock_mpu):
        """vpp_rank is None and enable_recompute flag False → defaults to 0."""
        self.config.enable_recompute_layers_per_pp_rank = False
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = None
        mock_mpu.is_pipeline_first_stage.return_value = True

        result = should_recompute(self.config, layer_number=1, num_recompute=None)
        # layer_number=1, chunk_prefix=6, first_stage: recompute_priority=0, full_recompute_layers=None
        # num_recompute=None → True
        self.assertTrue(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_vpp_size_none_defaults_to_one(self, mock_mpu):
        """vpp_size is None → defaults to 1."""
        self.config.virtual_pipeline_model_parallel_size = None
        self.config.enable_recompute_layers_per_pp_rank = False
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = True

        result = should_recompute(self.config, layer_number=1, num_recompute=2)
        # recompute_priority=0 < num_recompute=2 → True
        self.assertTrue(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_first_stage_layer_exceeds_chunk_prefix(self, mock_mpu):
        """First stage: layer_number > chunk_prefix → return True (checkpoint)."""
        self.config.num_layer_list = [(2, 0), (2, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = True

        # chunk_prefix = 2+2 = 4, layer_number=5 > 4
        result = should_recompute(self.config, layer_number=5, num_recompute=None)
        self.assertTrue(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_first_stage_within_chunk_prefix(self, mock_mpu):
        """First stage: layer within chunk_prefix, check recompute_priority."""
        self.config.num_layer_list = [(3, 0), (3, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = True

        # chunk_prefix=6, layer_number=3, recompute_priority=2
        result = should_recompute(self.config, layer_number=3, num_recompute=2)
        # recompute_priority=2, num_recompute=2 → priority < num_recompute is False
        self.assertFalse(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_non_first_stage_computes_layer_offset(self, mock_mpu):
        """Non-first stage adjusts layer_number by chunk_start."""
        self.config.num_layer_list = [(2, 0), (3, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = False

        # chunk_start: layer 1-2 → start=0, layer 3-5 → start=2
        # layer_number=4 → second chunk, layer_number adjusted to 4-2=2, recompute_priority=1
        result = should_recompute(self.config, layer_number=4, num_recompute=1)
        # recompute_priority=1 < num_recompute=1 → False
        self.assertFalse(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_non_first_stage_first_chunk(self, mock_mpu):
        """Non-first stage: layer in the first chunk."""
        self.config.num_layer_list = [(4, 0), (4, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = False

        # layer_number=2, first chunk (start=0), recompute_priority=1
        result = should_recompute(self.config, layer_number=2, num_recompute=2)
        # recompute_priority=1 < num_recompute=2 → True
        self.assertTrue(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_full_recompute_layers_does_full_recompute(self, mock_mpu):
        """full_recompute_layers: priority below threshold → do full recompute (False)."""
        self.config.recompute_num_layers = 3
        self.config.num_layer_list = [(3, 0), (3, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = True

        # layer_number=2, recompute_priority=1 < full_recompute_layers=3 → False
        result = should_recompute(self.config, layer_number=2, num_recompute=None)
        self.assertFalse(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_full_recompute_layers_does_selective_recompute(self, mock_mpu):
        """full_recompute_layers: priority above full but below full+num → selective (True)."""
        self.config.recompute_num_layers = 2
        self.config.num_layer_list = [(3, 0), (3, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = True

        # layer_number=3, recompute_priority=2 ≥ full_recompute_layers=2
        # num_recompute=1, recompute_priority=2 < full_recompute_layers+num_recompute=3 → True
        result = should_recompute(self.config, layer_number=3, num_recompute=1)
        self.assertTrue(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_full_recompute_layers_no_recompute_needed(self, mock_mpu):
        """full_recompute_layers: priority above full+num → no recompute (False)."""
        self.config.recompute_num_layers = 2
        self.config.num_layer_list = [(4, 0), (4, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = True

        # layer_number=5, recompute_priority=4 ≥ 2, but 4 ≥ 2+2=4 → False
        result = should_recompute(self.config, layer_number=5, num_recompute=2)
        self.assertFalse(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_num_recompute_none_returns_true(self, mock_mpu):
        """num_recompute=None and full_recompute_layers=None → always True."""
        self.config.recompute_num_layers = None
        self.config.num_layer_list = [(3, 0), (3, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = True

        result = should_recompute(self.config, layer_number=1, num_recompute=None)
        self.assertTrue(result)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_common.mpu')
    def test_priority_exceeds_num_recompute(self, mock_mpu):
        """recompute_priority >= num_recompute → no recompute (False)."""
        self.config.recompute_num_layers = None
        self.config.num_layer_list = [(3, 0), (3, 0)]
        mock_mpu.get_virtual_pipeline_model_parallel_rank.return_value = 0
        mock_mpu.is_pipeline_first_stage.return_value = True

        # layer_number=3, recompute_priority=2, num_recompute=1 → False
        result = should_recompute(self.config, layer_number=3, num_recompute=1)
        self.assertFalse(result)

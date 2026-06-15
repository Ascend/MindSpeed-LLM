import unittest
from unittest.mock import patch, MagicMock

from mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor import granular_module_allocation


class TestGranularModuleAllocation(unittest.TestCase):
    """Unit tests for granular_module_allocation in recompute_adaptor.py."""

    def setUp(self):
        self.mock_self = MagicMock()
        self.mock_self.num_prefetch = 4

    def _make_args(self, pp_size=2, vpp_layers=2, reduce_recompute=False):
        """Helper to create a mock args namespace."""
        args = MagicMock()
        args.pipeline_model_parallel_size = pp_size
        args.num_layers_per_virtual_pipeline_stage = vpp_layers
        args.reduce_recompute_for_last_chunk = reduce_recompute
        return args

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_simple_prefetch_le_vpp(self, mock_ps, mock_get_args):
        """num_prefetch <= vpp_size: simple allocation, one swap layer per chunk."""
        self.mock_self.num_prefetch = 1
        mock_get_args.return_value = self._make_args()
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = False

        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=1, cur_pp_noop_layers=[])

        group, interval, num_prefetch, noop = result
        self.assertEqual(interval, 0)
        self.assertEqual(num_prefetch, 1)
        self.assertEqual(group[0], [['0'], ['']])  # swap_list
        self.assertEqual(group[2], [['0'], ['']])  # recompute_list

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_round_robin_prefetch_gt_vpp(self, mock_ps, mock_get_args):
        """num_prefetch > vpp_size: round-robin allocation of layers."""
        self.mock_self.num_prefetch = 4
        mock_get_args.return_value = self._make_args()
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = False

        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=1, cur_pp_noop_layers=[])

        group = result[0]
        # chunk 1 cleared by swap_list[1] = ['']
        self.assertEqual(group[0], [['0', '1'], ['']])  # swap_list
        self.assertEqual(group[2], [['0'], ['']])  # recompute simple

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_round_robin_recompute_gt_vpp(self, mock_ps, mock_get_args):
        """recompute_num_layers > vpp_size: round-robin recompute allocation."""
        mock_get_args.return_value = self._make_args()
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = False

        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=4, cur_pp_noop_layers=[])

        group = result[0]
        # recompute_list: both chunks get '0' and '1', then chunk 1 cleared
        self.assertEqual(group[2], [['0', '1'], ['']])

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_reduce_recompute_for_last_stage(self, mock_ps, mock_get_args):
        """Last stage with reduce_recompute_for_last_chunk: removes last recompute layer."""
        mock_get_args.return_value = self._make_args(reduce_recompute=True)
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = True

        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=2, cur_pp_noop_layers=[])

        group = result[0]
        # recompute_list: simple alloc ['0'] for chunk 0, then last chunk reduced → ['']
        # The result after reduce should have recompute_list[-1] = ['']
        self.assertEqual(group[2][-1], [''])

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_reduce_recompute_not_last_stage_no_effect(self, mock_ps, mock_get_args):
        """reduce_recompute has no effect when not on last stage."""
        mock_get_args.return_value = self._make_args(reduce_recompute=True)
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = False

        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=1, cur_pp_noop_layers=[])

        group = result[0]
        # recompute simple: chunk 0 gets '0', chunk 1 gets ''
        self.assertEqual(group[2][0], ['0'])

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_noop_layers_mask_entries(self, mock_ps, mock_get_args):
        """cur_pp_noop_layers masks corresponding swap/recompute entries to ''."""
        args = self._make_args(pp_size=2, vpp_layers=3)
        mock_get_args.return_value = args
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = False

        # vpp=0, layer_id = vpp*vpp_layer*pp_size + i + vpp_layer*cur_pp_rank
        # = 0*3*2 + i + 3*0 = i, so layer_id 0, 1, 2 for i=0,1,2
        # We mark layer_id=0 as noop
        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=1, cur_pp_noop_layers=[0])

        group = result[0]
        # swap_list chunk 0 entry 0 should be '' (masked by noop)
        self.assertEqual(group[0][0][0], '')
        # recompute_list chunk 0 entry 0 should also be ''
        if len(group[2][0]) > 0:
            self.assertEqual(group[2][0][0], '')

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_noop_empty_list_no_effect(self, mock_ps, mock_get_args):
        """Empty cur_pp_noop_layers: no masking."""
        mock_get_args.return_value = self._make_args()
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = False

        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=1, cur_pp_noop_layers=[])

        group = result[0]
        # swap chunk 0 should have '0' (not masked)
        self.assertIn('0', group[0][0])

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_result_structure(self, mock_ps, mock_get_args):
        """Verify the full return tuple structure."""
        mock_get_args.return_value = self._make_args()
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = False

        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=1, cur_pp_noop_layers=[])

        # result = [prefetch_recompute_group, interval, num_prefetch, cur_pp_noop_layers]
        self.assertEqual(len(result), 4)
        prefetch_recompute_group = result[0]
        # prefetch_recompute_group = [swap_list, prefetch_list, recompute_list]
        self.assertEqual(len(prefetch_recompute_group), 3)
        self.assertEqual(result[1], 0)  # interval
        self.assertEqual(result[2], 4)  # self.num_prefetch
        self.assertEqual(result[3], [])  # cur_pp_noop_layers

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor.parallel_state')
    def test_vpp1_round_robin(self, mock_ps, mock_get_args):
        """vpp_size=2 with num_prefetch=3 and recompute=3: round-robin, chunk 1 cleared."""
        self.mock_self.num_prefetch = 3
        mock_get_args.return_value = self._make_args(pp_size=1)
        mock_ps.get_pipeline_model_parallel_rank.return_value = 0
        mock_ps.is_pipeline_last_stage.return_value = False

        result = granular_module_allocation(self.mock_self, vpp_size=2, recompute_num_layers=3, cur_pp_noop_layers=[])

        group = result[0]
        # round-robin: chunk 0 gets ['0','1'], chunk 1 gets ['0']; then chunk 1 cleared
        self.assertEqual(group[0], [['0', '1'], ['']])
        self.assertEqual(group[2], [['0', '1'], ['']])

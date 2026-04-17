import unittest
import os
import tempfile
from unittest import mock
import argparse

# Now import the functions to test
from mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp import (
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    read_iteration,
    find_tp_ranks_for_stage,
    save_checkpoint,
    save_tracker,
    copy_metadata,
    load_ckpt,
    prepare_iter_dir,
    _merge_stages_to_vpp,
    merge_checkpoints,
    split_checkpoints,
    main
)


class TestConvertCkptPpVpp(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self._tmp_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self._tmp_obj.name

        self.addCleanup(self._tmp_obj.cleanup)
    
    def test_get_checkpoint_name(self):
        checkpoints_path = os.path.join(self.temp_dir, "checkpoints")
        iteration = 12345
        tensor_rank = 0
        pipeline_rank = 1
        
        expected_path = os.path.join(
            checkpoints_path,
            "iter_0012345",
            "mp_rank_00_001",
            "model_optim_rng.pt"
        )
        
        result = get_checkpoint_name(checkpoints_path, iteration, tensor_rank, pipeline_rank)
        self.assertEqual(result, expected_path)
    
    def test_get_checkpoint_tracker_filename(self):
        checkpoints_path = os.path.join(self.temp_dir, "checkpoints")
        expected_path = os.path.join(checkpoints_path, "latest_checkpointed_iteration.txt")
        
        result = get_checkpoint_tracker_filename(checkpoints_path)
        self.assertEqual(result, expected_path)
    
    def test_read_iteration(self):
        # Create tracker file
        checkpoints_path = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)
        tracker_path = get_checkpoint_tracker_filename(checkpoints_path)
        
        expected_iteration = 54321
        with open(tracker_path, "w") as f:
            f.write(str(expected_iteration))
        
        result = read_iteration(checkpoints_path)
        self.assertEqual(result, expected_iteration)
    
    def test_read_iteration_file_not_found(self):
        checkpoints_path = os.path.join(self.temp_dir, "non_existent")
        with self.assertRaises(FileNotFoundError):
            read_iteration(checkpoints_path)
    
    def test_find_tp_ranks_for_stage(self):
        # Create test directory structure
        iter_dir = os.path.join(self.temp_dir, "iter_0012345")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Create some mp_rank directories
        os.makedirs(os.path.join(iter_dir, "mp_rank_00_000"), exist_ok=True)
        os.makedirs(os.path.join(iter_dir, "mp_rank_01_000"), exist_ok=True)
        os.makedirs(os.path.join(iter_dir, "mp_rank_00_001"), exist_ok=True)
        
        # Test for PP rank 0
        result = find_tp_ranks_for_stage(iter_dir, 0)
        self.assertEqual(result, [0, 1])
        
        # Test for PP rank 1
        result = find_tp_ranks_for_stage(iter_dir, 1)
        self.assertEqual(result, [0])
    
    def test_save_checkpoint(self):
        save_iter_dir = os.path.join(self.temp_dir, "iter_0012345")
        tp_rank = 0
        pp_rank = 1
        state_dict = {"model": {"key": "value"}}
        
        # Mock torch.save
        with mock.patch('torch.save') as mock_save:
            save_checkpoint(save_iter_dir, tp_rank, pp_rank, state_dict)
            
            # Verify the directory was created
            expected_dir = os.path.join(save_iter_dir, "mp_rank_00_001")
            self.assertTrue(os.path.exists(expected_dir))
            
            # Verify torch.save was called
            mock_save.assert_called_once()
    
    def test_save_tracker(self):
        save_dir = os.path.join(self.temp_dir, "checkpoints")
        iteration = 12345
        
        save_tracker(save_dir, iteration)
        
        # Verify the tracker file was created
        tracker_path = get_checkpoint_tracker_filename(save_dir)
        self.assertTrue(os.path.exists(tracker_path))
        
        # Verify the content
        with open(tracker_path, "r") as f:
            content = f.read().strip()
        self.assertEqual(content, str(iteration))
    
    def test_copy_metadata(self):
        state_dict = {
            "optimizer": "opt_state",
            "opt_param_scheduler": "scheduler_state",
            "rng_state": "rng_state",
            "other_key": "other_value"
        }
        
        expected_meta = {
            "optimizer": "opt_state",
            "opt_param_scheduler": "scheduler_state",
            "rng_state": "rng_state"
        }
        
        result = copy_metadata(state_dict)
        self.assertEqual(result, expected_meta)
    
    def test_load_ckpt(self):
        # Create a dummy checkpoint file
        checkpoint_path = os.path.join(self.temp_dir, "model_optim_rng.pt")
        with open(checkpoint_path, "w") as f:
            f.write("dummy content")
        
        # Mock torch.load
        with mock.patch('torch.load') as mock_load:
            mock_load.return_value = {}
            result = load_ckpt(checkpoint_path)
            # Should return an empty dict due to our mock
            self.assertEqual(result, {})
            # Verify torch.load was called
            mock_load.assert_called_once()
    
    def test_load_ckpt_file_not_found(self):
        checkpoint_path = os.path.join(self.temp_dir, "non_existent.pt")
        with self.assertRaises(FileNotFoundError):
            load_ckpt(checkpoint_path)
    
    def test_prepare_iter_dir(self):
        load_dir = self.temp_dir
        iteration = 12345
        iter_dir = os.path.join(load_dir, "iter_0012345")
        os.makedirs(iter_dir, exist_ok=True)
        
        result = prepare_iter_dir(load_dir, iteration)
        self.assertEqual(result, iter_dir)
    
    def test_prepare_iter_dir_not_found(self):
        load_dir = self.temp_dir
        iteration = 12345
        with self.assertRaises(FileNotFoundError):
            prepare_iter_dir(load_dir, iteration)
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.load_ckpt')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.save_checkpoint')
    def test_merge_stages_to_vpp(self, mock_save_checkpoint, mock_load_ckpt):
        # Setup mock return values
        mock_load_ckpt.return_value = {
            "args": mock.MagicMock(),
            "checkpoint_version": 3.0,
            "iteration": 12345,
            "optimizer": "opt_state",
            "model": {"key": "value"}
        }
        
        # Setup test parameters
        iter_dir = self.temp_dir
        load_dir = self.temp_dir
        iteration = 12345
        stages = [0, 1]
        tp_ranks = [0]
        save_iter_dir = os.path.join(self.temp_dir, "save_iter")
        save_pp_rank = 0
        label = "Test"
        
        # Create save directory
        os.makedirs(save_iter_dir, exist_ok=True)
        
        # Call the function
        _merge_stages_to_vpp(iter_dir, load_dir, iteration, stages, tp_ranks, save_iter_dir, save_pp_rank, label)
        
        # Verify load_ckpt was called twice (once per stage)
        self.assertEqual(mock_load_ckpt.call_count, 2)
        
        # Verify save_checkpoint was called
        mock_save_checkpoint.assert_called_once()
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.read_iteration')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.prepare_iter_dir')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.find_tp_ranks_for_stage')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp._merge_stages_to_vpp')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.save_tracker')
    def test_merge_checkpoints_single_source(self, mock_save_tracker, mock_merge_stages, mock_find_tp, mock_prepare_iter, mock_read_iter):
        # Setup mock return values
        mock_read_iter.return_value = 12345
        mock_prepare_iter.return_value = self.temp_dir
        mock_find_tp.return_value = [0]
        
        # Create args
        args = argparse.Namespace(
            load_dir=self.temp_dir,
            load_dir_edge=None,
            load_dir_cloud=None,
            save_dir_edge=os.path.join(self.temp_dir, "edge"),
            save_dir_cloud=os.path.join(self.temp_dir, "cloud"),
            merge_stages="0,3",
            merge_cloud_stages="1,2",
            middle_stages=None,
            iteration=None
        )
        
        # Create output directories
        os.makedirs(args.save_dir_edge, exist_ok=True)
        os.makedirs(args.save_dir_cloud, exist_ok=True)
        
        # Call the function
        merge_checkpoints(args)
        
        # Verify functions were called
        mock_read_iter.assert_called_once()
        mock_prepare_iter.assert_called_once()
        mock_find_tp.assert_called()
        mock_merge_stages.assert_called()
        mock_save_tracker.assert_called()
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.read_iteration')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.prepare_iter_dir')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.find_tp_ranks_for_stage')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.load_ckpt')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.save_checkpoint')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.save_tracker')
    def test_merge_checkpoints_middle_stages(self, mock_save_tracker, mock_save_checkpoint, mock_load_ckpt, mock_find_tp, mock_prepare_iter, mock_read_iter):
        # Setup mock return values
        mock_read_iter.return_value = 12345
        mock_prepare_iter.return_value = self.temp_dir
        mock_find_tp.return_value = [0]
        mock_load_ckpt.return_value = {
            "args": mock.MagicMock(),
            "checkpoint_version": 3.0,
            "iteration": 12345,
            "optimizer": "opt_state",
            "model": {"key": "value"}
        }
        
        # Create args
        args = argparse.Namespace(
            load_dir=self.temp_dir,
            load_dir_edge=None,
            load_dir_cloud=None,
            save_dir_edge=os.path.join(self.temp_dir, "edge"),
            save_dir_cloud=os.path.join(self.temp_dir, "cloud"),
            merge_stages="0,3",
            merge_cloud_stages=None,
            middle_stages="1,2",
            iteration=None
        )
        
        # Create output directories
        os.makedirs(args.save_dir_edge, exist_ok=True)
        os.makedirs(args.save_dir_cloud, exist_ok=True)
        
        # Call the function
        merge_checkpoints(args)
        
        # Verify functions were called
        mock_read_iter.assert_called_once()
        mock_prepare_iter.assert_called_once()
        mock_find_tp.assert_called()
        mock_load_ckpt.assert_called()
        mock_save_checkpoint.assert_called()
        mock_save_tracker.assert_called()
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.read_iteration')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.find_tp_ranks_for_stage')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.load_ckpt')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.save_checkpoint')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.save_tracker')
    def test_split_checkpoints(self, mock_save_tracker, mock_save_checkpoint, mock_load_ckpt, mock_find_tp, mock_read_iter):
        # Setup mock return values
        mock_read_iter.return_value = 12345
        mock_find_tp.return_value = [0]
        mock_load_ckpt.return_value = {
            "args": mock.MagicMock(),
            "checkpoint_version": 3.0,
            "iteration": 12345,
            "optimizer": "opt_state",
            "model0": {"key1": "value1"},
            "model1": {"key2": "value2"}
        }
        
        # Create args
        args = argparse.Namespace(
            load_dir_edge=self.temp_dir,
            load_dir_cloud=self.temp_dir,
            save_dir=os.path.join(self.temp_dir, "save"),
            split_rank=0,
            split_cloud_rank=0,
            num_cloud_vpp_chunks=2,
            middle_ranks=None,
            iteration=None
        )
        
        # Create edge iteration directory
        edge_iter_dir = os.path.join(args.load_dir_edge, "iter_0012345")
        os.makedirs(edge_iter_dir, exist_ok=True)
        # Create a dummy mp_rank directory
        os.makedirs(os.path.join(edge_iter_dir, "mp_rank_00_000"), exist_ok=True)
        
        # Create cloud iteration directory
        cloud_iter_dir = os.path.join(args.load_dir_cloud, "iter_0012345")
        os.makedirs(cloud_iter_dir, exist_ok=True)
        # Create a dummy mp_rank directory
        os.makedirs(os.path.join(cloud_iter_dir, "mp_rank_00_000"), exist_ok=True)
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Call the function
        split_checkpoints(args)
        
        # Verify functions were called
        mock_read_iter.assert_called_once()
        mock_find_tp.assert_called()
        mock_load_ckpt.assert_called()
        mock_save_checkpoint.assert_called()
        mock_save_tracker.assert_called_once()
    
    @mock.patch('argparse.ArgumentParser.parse_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.merge_checkpoints')
    def test_main_merge(self, mock_merge, mock_parse_args):
        # Setup mock
        mock_parse_args.return_value = argparse.Namespace(
            command='merge',
            load_dir=self.temp_dir,
            save_dir_edge=os.path.join(self.temp_dir, "edge"),
            save_dir_cloud=os.path.join(self.temp_dir, "cloud"),
            merge_stages="0,3"
        )
        
        # Call main
        main()
        
        # Verify merge_checkpoints was called
        mock_merge.assert_called_once()
    
    @mock.patch('argparse.ArgumentParser.parse_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.convert_ckpt_pp_vpp.split_checkpoints')
    def test_main_split(self, mock_split, mock_parse_args):
        # Setup mock
        mock_parse_args.return_value = argparse.Namespace(
            command='split',
            load_dir_edge=self.temp_dir,
            load_dir_cloud=self.temp_dir,
            save_dir=os.path.join(self.temp_dir, "save"),
            split_rank=0
        )
        
        # Call main
        main()
        
        # Verify split_checkpoints was called
        mock_split.assert_called_once()
    
    @mock.patch('argparse.ArgumentParser.parse_args')
    def test_main_invalid_command(self, mock_parse_args):
        # Setup mock with invalid command
        mock_parse_args.return_value = argparse.Namespace(
            command='invalid'
        )
        
        # Call main and expect ValueError
        with self.assertRaises(ValueError):
            main()

if __name__ == '__main__':
    unittest.main()

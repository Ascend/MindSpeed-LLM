import unittest
from unittest import mock
import numpy as np

# Now import the functions and classes to test
from mindspeed_llm.tasks.posttrain.ldt_sft.utils import (
    train_valid_test_datasets_provider_ldt,
    build_train_valid_test_datasets,
    _build_train_valid_test_datasets,
    LDTDecoderPackedMTFDataset,
    _build_index_mappings
)


class TestUtils(unittest.TestCase):
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.get_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.core_gpt_dataset_config_from_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.MockGPTDataset')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.GPTDataset')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.print_rank_0')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.build_train_valid_test_datasets')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.BlendedMegatronDatasetBuilder')
    def test_train_valid_test_datasets_provider_ldt_instruction_dataset(self, mock_builder, mock_build_datasets, mock_print, mock_gpt_dataset, mock_mock_dataset, mock_config_from_args, mock_get_args):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_args.is_instruction_dataset = True
        mock_args.is_pairwise_dataset = False
        mock_args.data_path = ["test_data"]
        mock_args.split = "90,5,5"
        mock_args.seq_length = 1024
        mock_args.seed = 42
        mock_get_args.return_value = mock_args
        
        mock_config = mock.MagicMock()
        mock_config.mock = False
        mock_config_from_args.return_value = mock_config
        
        mock_train_ds = mock.MagicMock()
        mock_valid_ds = mock.MagicMock()
        mock_test_ds = mock.MagicMock()
        mock_build_datasets.return_value = (mock_train_ds, mock_valid_ds, mock_test_ds)
        
        # Call the function
        train_val_test_num_samples = [1000, 100, 100]
        result = train_valid_test_datasets_provider_ldt(train_val_test_num_samples)
        
        # Verify the result
        self.assertEqual(result, (mock_train_ds, mock_valid_ds, mock_test_ds))
        mock_build_datasets.assert_called_once()
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.get_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.core_gpt_dataset_config_from_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.MockGPTDataset')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.GPTDataset')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.print_rank_0')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.BlendedMegatronDatasetBuilder')
    def test_train_valid_test_datasets_provider_ldt_normal_dataset(self, mock_builder, mock_print, mock_gpt_dataset, mock_mock_dataset, mock_config_from_args, mock_get_args):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_args.is_instruction_dataset = False
        mock_args.is_pairwise_dataset = False
        mock_get_args.return_value = mock_args
        
        mock_config = mock.MagicMock()
        mock_config.mock = False
        mock_config_from_args.return_value = mock_config
        
        mock_train_ds = mock.MagicMock()
        mock_valid_ds = mock.MagicMock()
        mock_test_ds = mock.MagicMock()
        mock_builder_instance = mock.MagicMock()
        mock_builder_instance.build.return_value = (mock_train_ds, mock_valid_ds, mock_test_ds)
        mock_builder.return_value = mock_builder_instance
        
        # Call the function
        train_val_test_num_samples = [1000, 100, 100]
        result = train_valid_test_datasets_provider_ldt(train_val_test_num_samples)
        
        # Verify the result
        self.assertEqual(result, (mock_train_ds, mock_valid_ds, mock_test_ds))
        mock_builder.assert_called_once()
        mock_builder_instance.build.assert_called_once()
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.get_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.build_tokenizer')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils._build_train_valid_test_datasets')
    def test_build_train_valid_test_datasets(self, mock_build_datasets, mock_build_tokenizer, mock_get_args):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_get_args.return_value = mock_args
        
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.pad = 0
        mock_tokenizer.eos = 1
        mock_build_tokenizer.return_value = mock_tokenizer
        
        mock_train_ds = mock.MagicMock()
        mock_valid_ds = mock.MagicMock()
        mock_test_ds = mock.MagicMock()
        mock_build_datasets.return_value = (mock_train_ds, mock_valid_ds, mock_test_ds)
        
        # Call the function
        data_prefix = ["test_data"]
        splits_string = "90,5,5"
        seq_length = 1024
        train_valid_test_num_samples = [1000, 100, 100]
        seed = 42
        
        result = build_train_valid_test_datasets(
            data_prefix=data_prefix,
            splits_string=splits_string,
            seq_length=seq_length,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seed=seed
        )
        
        # Verify the result
        self.assertEqual(result, (mock_train_ds, mock_valid_ds, mock_test_ds))
        mock_build_datasets.assert_called_once()
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.get_packed_indexed_dataset')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.get_train_valid_test_split_')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.print_rank_0')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.LDTDecoderPackedMTFDataset')
    def test_build_train_valid_test_datasets_internal(self, mock_dataset, mock_print, mock_split, mock_get_indexed_dataset):
        # Setup mock return values
        mock_indexed_dataset = {"test": [1, 2, 3, 4, 5]}
        mock_get_indexed_dataset.return_value = mock_indexed_dataset
        
        mock_split.return_value = [0, 3, 4, 5]  # train: 0-3, valid: 3-4, test: 4-5
        
        mock_train_ds = mock.MagicMock()
        mock_valid_ds = mock.MagicMock()
        mock_test_ds = mock.MagicMock()
        mock_dataset.side_effect = [mock_train_ds, mock_valid_ds, mock_test_ds]
        
        # Call the function
        data_prefix = "test_data"
        splits_string = "60,20,20"
        seq_length = 1024
        pad_token = 0
        eos_token = 1
        train_valid_test_num_samples = [1000, 100, 100]
        seed = 42
        
        result = _build_train_valid_test_datasets(
            data_prefix=data_prefix,
            splits_string=splits_string,
            seq_length=seq_length,
            pad_token=pad_token,
            eos_token=eos_token,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seed=seed
        )
        
        # Verify the result
        self.assertEqual(result, (mock_train_ds, mock_valid_ds, mock_test_ds))
        self.assertEqual(mock_dataset.call_count, 3)
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.get_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.MTFDataset')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils._build_index_mappings')
    def test_ldt_decoder_packed_mtf_dataset_init(self, mock_build_index, mock_mtf_dataset, mock_get_args):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_args.no_shuffle = False
        mock_get_args.return_value = mock_args
        
        mock_mtf_instance = mock.MagicMock()
        mock_mtf_dataset.return_value = mock_mtf_instance
        
        mock_shuffle_index = [1, 2, 3, 4, 5]
        mock_build_index.return_value = mock_shuffle_index
        
        # Call the constructor
        name = "train"
        data_prefix = "test_data"
        documents = [0, 1, 2, 3, 4]
        seq_length = 1024
        pad_token = 0
        eos_token = 1
        num_samples = 1000
        seed = 42
        
        dataset = LDTDecoderPackedMTFDataset(
            name=name,
            data_prefix=data_prefix,
            documents=documents,
            seq_length=seq_length,
            pad_token=pad_token,
            eos_token=eos_token,
            num_samples=num_samples,
            seed=seed
        )
        
        # Verify the attributes
        self.assertEqual(dataset.args, mock_args)
        self.assertEqual(dataset.mtf_dataset, mock_mtf_instance)
        self.assertEqual(dataset.pad_token, pad_token)
        self.assertEqual(dataset.seq_length, seq_length)
        self.assertEqual(dataset.eos_token, eos_token)
        self.assertEqual(dataset.shuffle_index, mock_shuffle_index)
        self.assertEqual(dataset.cur_batch_index, [])
        self.assertEqual(dataset.iteration, 1)
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.get_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.torch.distributed.get_rank')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.torch.cuda.device_count')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.os.path.isfile')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.print_rank_0')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils._build_shuffle_idx')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils._build_sequential_idx')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.random.shuffle')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.np.save')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.torch.distributed.barrier')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.torch.cuda.LongTensor')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.is_vtp_enabled')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.parallel_state.get_data_parallel_group')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.parallel_state.get_context_parallel_group')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.parallel_state.get_tensor_model_parallel_group')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.parallel_state.get_pipeline_model_parallel_group')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.torch.distributed.all_reduce')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.torch.distributed.get_world_size')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.check_equal')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.utils.np.load')
    def test_build_index_mappings(self, mock_np_load, mock_check_equal, mock_get_world_size, mock_all_reduce, 
                                mock_get_pipeline_group, mock_get_tensor_group, mock_get_context_group, 
                                mock_get_data_group, mock_is_vtp, mock_long_tensor, mock_barrier, 
                                mock_np_save, mock_random_shuffle, mock_build_sequential, mock_build_shuffle, 
                                mock_print, mock_isfile, mock_device_count, mock_get_rank, mock_get_args):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_args.padded_samples = False
        mock_args.stage = "test"
        mock_args.full_shuffle_instruction_dataset = False
        mock_get_args.return_value = mock_args
        
        mock_get_rank.return_value = 0
        mock_device_count.return_value = 1
        mock_isfile.return_value = False
        
        mock_shuffle_idx = np.array([1, 2, 3, 4, 5])
        mock_build_shuffle.return_value = mock_shuffle_idx
        mock_np_load.return_value = mock_shuffle_idx
        
        mock_long_tensor_instance = mock.MagicMock()
        mock_long_tensor_instance.__getitem__.return_value.item.return_value = 1
        mock_long_tensor.return_value = mock_long_tensor_instance
        
        mock_is_vtp.return_value = False
        mock_get_world_size.side_effect = [8, 2]  # total world size, tensor parallel world size
        
        # Call the function
        name = "train"
        data_prefix = "test_data"
        start_index = 0
        nb_documents = 10
        mtf_dataset = mock.MagicMock()
        num_samples = 100
        seq_length = 1024
        seed = 42
        shuffle = True
        
        result = _build_index_mappings(
            name=name,
            data_prefix=data_prefix,
            start_index=start_index,
            nb_documents=nb_documents,
            mtf_dataset=mtf_dataset,
            num_samples=num_samples,
            seq_length=seq_length,
            seed=seed,
            shuffle=shuffle
        )
        
        # Verify the result
        self.assertListEqual(result.tolist(), mock_shuffle_idx.tolist())
        mock_np_save.assert_called_once()
        mock_np_load.assert_called_once()

if __name__ == '__main__':
    unittest.main()

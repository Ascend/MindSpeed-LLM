import unittest
import tempfile
from unittest import mock
from functools import partial

import torch

# Now import the functions and classes to test
from mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer import (
    core_transformer_config_from_args,
    ldt_core_transformer_config_from_args,
    build_train_args,
    LDTSFTTrainer
)


class TestLDTSFTTrainer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self._tmp_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self._tmp_obj.name

        self.addCleanup(self._tmp_obj.cleanup)
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.dataclasses.fields')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.TransformerConfig')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.MLATransformerConfig')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.HeterogeneousTransformerConfig')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.F.silu')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.squared_relu')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.torch.nn.init.xavier_uniform_')
    def test_core_transformer_config_from_args(self, mock_xavier, mock_squared_relu, mock_silu, mock_hetero_config, mock_mla_config, mock_config, mock_dataclass_fields):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_args.multi_latent_attention = False
        mock_args.heterogeneous_layers_config_path = None
        mock_args.no_persist_layer_norm = False
        mock_args.apply_layernorm_1p = False
        mock_args.norm_epsilon = 1e-5
        mock_args.params_dtype = torch.float32
        mock_args.overlap_p2p_comm = True
        mock_args.num_experts = 0
        mock_args.rotary_interleaved = False
        mock_args.decoder_first_pipeline_num_layers = 0
        mock_args.decoder_last_pipeline_num_layers = 0
        mock_args.fp8_param_gather = False
        mock_args.swiglu = False
        mock_args.bias_gelu_fusion = False
        mock_args.squared_relu = False
        mock_args.init_method_xavier_uniform = False
        mock_args.group_query_attention = False
        mock_args.config_logger_dir = self.temp_dir
        mock_args.cp_comm_type = []
        mock_args.is_hybrid_model = False
        
        # Mock dataclasses.fields to return empty list
        mock_dataclass_fields.return_value = []
        
        # Call the function
        result = core_transformer_config_from_args(mock_args)
        
        # Verify the result
        mock_config.assert_called_once()
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.core_transformer_config_from_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_layer_offset')
    def test_ldt_core_transformer_config_from_args(self, mock_get_layer_offset, mock_core_config):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_args.pipeline_model_parallel_size = 1
        mock_args.num_layers_per_virtual_pipeline_stage = None
        mock_args.moe_expert_capacity_factor = None
        mock_args.num_layer_list = None
        mock_args.num_layers = 10
        
        mock_config = mock.MagicMock()
        mock_core_config.return_value = mock_config
        
        # Call the function
        result = ldt_core_transformer_config_from_args(mock_args)
        
        # Verify the result
        mock_core_config.assert_called_once_with(mock_args)
        self.assertEqual(result, mock_config)
    
    @mock.patch('megatron.training.training.setup_model_and_optimizer')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_model_config')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.build_train_valid_test_data_iterators')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.one_logger_utils')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.print_datetime')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.print_rank_0')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.mpu.is_pipeline_first_stage')
    def test_build_train_args_first_stage(self, mock_is_first_stage, mock_print_rank, mock_print_datetime, mock_one_logger, mock_build_iterators, mock_get_model_config, mock_setup_model):
        # Setup mock return values
        mock_is_first_stage.return_value = True
        
        mock_model = [mock.MagicMock()]
        mock_optimizer = mock.MagicMock()
        mock_scheduler = mock.MagicMock()
        mock_setup_model.return_value = (mock_model, mock_optimizer, mock_scheduler)
        
        mock_config = mock.MagicMock()
        mock_get_model_config.return_value = mock_config
        
        mock_train_iter = mock.MagicMock()
        mock_valid_iter = mock.MagicMock()
        mock_test_iter = mock.MagicMock()
        mock_build_iterators.return_value = (mock_train_iter, mock_valid_iter, mock_test_iter)
        
        # Call the function
        args = mock.MagicMock()
        args.lu_lora_final_layer_index = None
        args.mtp_num_layers = None
        args.schedules_method = "test"
        args.virtual_pipeline_model_parallel_size = None
        
        timers = mock.MagicMock()
        train_valid_test_dataset_provider = mock.MagicMock()
        model_provider = mock.MagicMock()
        model_type = "gpt"
        forward_step_func = mock.MagicMock()
        process_non_loss_data_func = mock.MagicMock()
        app_metrics = {}
        
        input_args = (
            args, timers, train_valid_test_dataset_provider, 
            model_provider, model_type, forward_step_func, 
            process_non_loss_data_func, app_metrics
        )
        
        train_args, test_data_iterator_list = build_train_args(*input_args)
        
        # Verify the result
        self.assertEqual(len(train_args), 8)
        self.assertEqual(len(test_data_iterator_list), 1)
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.SFTTrainer.__init__')
    def test_ldt_sft_trainer_init(self, mock_sft_init):
        # Call the constructor
        trainer = LDTSFTTrainer()
        
        # Verify the parent class was initialized
        mock_sft_init.assert_called_once()
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.SFTTrainer.__init__')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.train_valid_test_datasets_provider_ldt')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.set_jit_fusion_options')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.print_rank_0')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.build_train_args')
    def test_ldt_sft_trainer_initialize(self, mock_build_train, mock_print_rank, mock_set_jit, mock_dataset_provider, mock_sft_init):
        # Setup mock return values
        mock_sft_init.return_value = None
        
        mock_train_args = mock.MagicMock()
        mock_test_data = mock.MagicMock()
        mock_build_train.return_value = (mock_train_args, [mock_test_data])
        
        # Create trainer instance
        trainer = LDTSFTTrainer()
        trainer.args = mock.MagicMock()
        trainer.timers = mock.MagicMock()
        trainer.model_provider = mock.MagicMock()
        trainer.model_type = "gpt"
        trainer.forward_step = mock.MagicMock()
        trainer.process_non_loss_data_func = mock.MagicMock()
        trainer.log_initialization = mock.MagicMock()
        trainer.synchronize_start_time = mock.MagicMock()
        
        # Call initialize
        trainer.initialize()
        
        # Verify the dataset provider was set
        self.assertEqual(trainer.train_valid_test_datasets_provider, mock_dataset_provider)
        self.assertTrue(trainer.train_valid_test_datasets_provider.is_distributed)
        mock_build_train.assert_called_once()
        self.assertEqual(trainer.train_args, mock_train_args)
        self.assertEqual(trainer.test_data_iterator_list, [mock_test_data])
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.SFTTrainer.__init__')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.print_rank_0')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.core_transformer_config_from_yaml')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.ldt_core_transformer_config_from_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.import_module')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_gpt_layer_with_transformer_engine_spec')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_gpt_layer_local_spec')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_gpt_mtp_block_spec')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.GPTModel')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.megatron')
    def test_ldt_sft_trainer_model_provider(self, mock_megatron, mock_gpt_model, mock_get_mtp_spec, mock_get_local_spec, mock_get_te_spec, mock_import, mock_ldt_config, mock_yaml_config, mock_print_rank, mock_get_args, mock_sft_init):
        # Setup mock return values
        mock_sft_init.return_value = None
        
        mock_args = mock.MagicMock()
        mock_args.transformer_impl = "local"
        mock_args.yaml_cfg = None
        mock_args.use_mcore_models = True
        mock_args.spec = None
        mock_args.num_experts = 0
        mock_args.moe_grouped_gemm = False
        mock_args.mtp_num_layers = None
        mock_args.padded_vocab_size = 50257
        mock_args.max_position_embeddings = 2048
        mock_args.fp16_lm_cross_entropy = False
        mock_args.untie_embeddings_and_output_weights = False
        mock_args.position_embedding_type = "rotary"
        mock_args.rotary_percent = 1.0
        mock_args.rotary_seq_len_interpolation_factor = 1.0
        mock_get_args.return_value = mock_args
        
        mock_config = mock.MagicMock()
        mock_ldt_config.return_value = mock_config
        
        mock_transformer_spec = mock.MagicMock()
        mock_get_local_spec.return_value = mock_transformer_spec
        
        mock_gpt_instance = mock.MagicMock()
        mock_gpt_model.return_value = mock_gpt_instance
        
        # Create trainer instance
        trainer = LDTSFTTrainer()
        
        # Call model_provider
        result = trainer.model_provider(pre_process=True, post_process=True)
        
        # Verify the result
        mock_gpt_model.assert_called_once()
        self.assertEqual(result, mock_gpt_instance)
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.SFTTrainer.__init__')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_timers')
    def test_ldt_sft_trainer_forward_step(self, mock_get_timers, mock_get_args, mock_sft_init):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_args.use_legacy_models = False
        mock_get_args.return_value = mock_args
        
        mock_timers = mock.MagicMock()
        mock_get_timers.return_value = mock_timers
        
        # Create trainer instance
        trainer = LDTSFTTrainer()
        trainer.get_batch = mock.MagicMock()
        trainer.get_batch.return_value = (torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]]), torch.tensor([[1, 1, 1]]), torch.tensor([[0, 1, 2]]))
        trainer.loss_func = mock.MagicMock()
        
        # Call forward_step with batch=None
        data_iterator = mock.MagicMock()
        model = mock.MagicMock()
        model.return_value = torch.tensor([[0.1, 0.2, 0.3]])
        
        output_tensor, loss_func = trainer.forward_step(data_iterator, model)
        
        # Verify the result
        self.assertIsInstance(output_tensor, torch.Tensor)
        self.assertIsInstance(loss_func, partial)
        
        # Call forward_step with batch provided
        batch = {
            "tokens": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, 3]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "position_ids": torch.tensor([[0, 1, 2]])
        }
        
        output_tensor, loss_func = trainer.forward_step(data_iterator, model, batch=batch)
        
        # Verify the result
        self.assertIsInstance(output_tensor, torch.Tensor)
        self.assertIsInstance(loss_func, partial)
    
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.SFTTrainer.__init__')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.get_args')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.print_rank_0')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.print_datetime')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.save_checkpoint')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.evaluate_and_print_results')
    @mock.patch('mindspeed_llm.tasks.posttrain.ldt_sft.ldt_sft_trainer.train')
    def test_ldt_sft_trainer_train(self, mock_train, mock_evaluate, mock_save_checkpoint, mock_print_datetime, mock_print_rank, mock_get_args, mock_sft_init):
        # Setup mock return values
        mock_args = mock.MagicMock()
        mock_args.skip_train = False
        mock_args.dataloader_type = "test"
        mock_args.retro_project_dir = None
        mock_args.do_train = True
        mock_args.train_iters = 100
        mock_args.save = True
        mock_args.save_interval = 10
        mock_args.do_valid = True
        mock_args.do_test = True
        mock_get_args.return_value = mock_args
        
        # Create trainer instance
        trainer = LDTSFTTrainer()
        trainer.test_data_iterator_list = [mock.MagicMock()]
        trainer.train_args = [
            mock.MagicMock(),  # forward_step_func
            mock.MagicMock(),  # model
            mock.MagicMock(),  # optimizer
            mock.MagicMock(),  # opt_param_scheduler
            mock.MagicMock(),  # train_data_iterator
            mock.MagicMock(),  # valid_data_iterator
            mock.MagicMock(),  # process_non_loss_data_func
            mock.MagicMock()   # config
        ]
        
        # Call train
        mock_train.return_value = (50, 1000000)
        trainer.train()
        
        # Verify the function was called
        mock_train.assert_called_once()
        mock_evaluate.assert_called()

if __name__ == '__main__':
    unittest.main()

from unittest.mock import patch, MagicMock

# pylint: disable=redefined-outer-name, too-many-lines
import pytest
import torch
from megatron.core.transformer.enums import AttnBackend

# Test targets
from mindspeed_llm.core.layerwise_disaggregated_training.initialize import (
    initialize_megatron,
    _initialize_distributed,
    validate_args_ldt,
    _validate_args_ldt,
)


class MockArgs:
    """Mock args object for testing"""

    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.seed = 42
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.virtual_pipeline_model_parallel_size = None
        self.pipeline_model_parallel_split_rank = None
        self.pipeline_model_parallel_comm_backend = None
        self.context_parallel_size = 1
        self.hierarchical_context_parallel_sizes = None
        self.expert_model_parallel_size = 1
        self.num_distributed_optimizer_instances = 1
        self.expert_tensor_parallel_size = None
        self.distributed_timeout_minutes = 60
        self.nccl_communicator_config_path = None
        self.use_tp_pp_dp_mapping = False
        self.encoder_tensor_model_parallel_size = 0
        self.encoder_pipeline_model_parallel_size = 0
        self.enable_gloo_process_groups = False
        self.layerwise_disaggregated_training = False
        self.data_parallel_size = 1
        self.lazy_mpu_init = False
        self.use_cpu_initialization = False
        self.data_parallel_random_init = False
        self.te_rng_tracker = False
        self.inference_rng_tracker = False
        self.enable_cuda_graph = False
        self.num_experts = None
        self.tp_comm_overlap = False
        self.ckpt_convert_format = None
        self.ckpt_convert_save = None
        self.load = None
        self.use_checkpoint_args = False
        self.non_persistent_ckpt_type = None
        self.async_save = False
        self.use_persistent_ckpt_worker = False
        self.num_layer_list = None
        self.yaml_cfg = None
        self.rerun_mode = 'disabled'
        self.error_injection_rate = 0.0
        self.error_injection_type = 'correct_result'
        self.result_rejected_tracker_filename = None
        self.external_cuda_graph = False
        self.distributed_backend = 'nccl'
        self.local_rank = 0
        self.attention_backend = 'flash'
        self.spec = None
        self.use_legacy_models = False
        self.ckpt_format = 'torch'
        self.micro_batch_size = 1
        self.global_batch_size = None
        self.num_layers_per_virtual_pipeline_stage = None
        self.num_virtual_stages_per_pipeline_rank = None
        self.overlap_p2p_comm = False
        self.decoder_first_pipeline_num_layers = None
        self.decoder_last_pipeline_num_layers = None
        self.num_layers = 12
        self.decoder_num_layers = None
        self.account_for_embedding_in_pipeline_split = False
        self.account_for_loss_in_pipeline_split = False
        self.data_parallel_sharding_strategy = None
        self.overlap_param_gather = False
        self.overlap_grad_reduce = False
        self.use_distributed_optimizer = False
        self.use_torch_fsdp2 = False
        self.gradient_accumulation_fusion = False
        self.untie_embeddings_and_output_weights = True
        self.fp16 = False
        self.bf16 = True
        self.overlap_param_gather_with_optimizer_step = False
        self.main_grads_dtype = 'fp32'
        self.main_params_dtype = 'bf16'
        self.exp_avg_dtype = 'fp32'
        self.exp_avg_sq_dtype = 'fp32'
        self.fp8_param_gather = False
        self.use_custom_fsdp = False
        self.loss_scale = None
        self.check_for_nan_in_loss_and_grad = True
        self.accumulate_allreduce_grads_in_fp32 = False
        self.grad_reduce_in_bf16 = False
        self.dataloader_type = None
        self.num_dataset_builder_threads = 1
        self.train_iters = None
        self.train_samples = None
        self.lr_decay_iters = None
        self.lr_decay_samples = None
        self.lr_warmup_samples = 0
        self.rampup_batch_size = None
        self.lr_warmup_fraction = None
        self.lr_warmup_iters = 0
        self.encoder_num_layers = None
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.max_position_embeddings = 1024
        self.ffn_hidden_size = None
        self.swiglu = False
        self.kv_channels = None
        self.seq_length = 1024
        self.encoder_seq_length = None
        self.decoder_seq_length = None
        self.lr = None
        self.min_lr = 0.0
        self.save = None
        self.save_interval = None
        self.fp16_lm_cross_entropy = False
        self.fp32_residual_connection = False
        self.moe_grouped_gemm = False
        self.weight_decay_incr_style = 'constant'
        self.start_weight_decay = None
        self.end_weight_decay = None
        self.weight_decay = 0.01
        self.distribute_saved_activations = False
        self.recompute_granularity = None
        self.recompute_method = None
        self.sequence_parallel = False
        self.add_bias_linear = True
        self.add_qkv_bias = True
        self.retro_add_retriever = False
        self.decoupled_lr = None
        self.decoupled_min_lr = None
        self.use_rotary_position_embeddings = False
        self.rotary_interleaved = False
        self.apply_rope_fusion = False
        self.position_embedding_type = 'rope'
        self.add_position_embedding = True
        self.transformer_impl = 'transformer_engine'
        self.mrope_section = None
        self.moe_ffn_hidden_size = None
        self.use_dist_ckpt = False
        self.deterministic_mode = False
        self.use_flash_attn = False
        self.cross_entropy_loss_fusion = False
        self.apply_query_key_layer_scaling = False
        self.attention_softmax_in_fp32 = False
        self.iterations_to_skip = []
        self.ckpt_fully_parallel_save_deprecated = False
        self.use_dist_ckpt_deprecated = False
        self.dist_ckpt_format_deprecated = False
        self.inference_batch_times_seqlen_threshold = -1
        self.inference_dynamic_batching = False
        self.inference_dynamic_batching_buffer_size_gb = None
        self.inference_dynamic_batching_buffer_guaranteed_fraction = None
        self.moe_use_upcycling = False
        self.no_load_optim = False
        self.no_load_rng = False
        self.optimizer_cpu_offload = False
        self.use_precision_aware_optimizer = False
        self.non_persistent_local_ckpt_dir = None
        self.replication = False
        self.replication_jump = None
        self.mtp_num_layers = None
        self.bias_gelu_fusion = True
        self.heterogeneous_layers_config_path = None
        self.heterogeneous_layers_config_encoded_json = None
        self.recompute_activations = None
        self.cp_comm_type = ''
        self.data_path = None
        self.data_args_path = None
        self.split = None
        self.train_data_path = None
        self.valid_data_path = None
        self.test_data_path = None
        self.per_split_data_args_path = None
        self.mock_data = 0
        self.retro_project_dir = None
        self.check_weight_hash_across_dp_replicas_interval = None
        self.ckpt_fully_parallel_save = True
        self.batch_size = None
        self.warmup = None
        self.model_parallel_size = None
        self.checkpoint_activations = None


@pytest.fixture
def mock_args():
    """Fixture for mock args"""
    return MockArgs()


class TestInitializeMegatron:
    """Tests for initialize_megatron function"""

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.set_global_variables')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.setup_logging')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.initialize_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._initialize_distributed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._set_random_seed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._init_autoresume')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._compile_dependencies')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    def test_initialize_megatron_basic(
        self,
        mock_features_manager,
        mock_vdp,
        mock_get_args,
        mock_compile,
        mock_autoresume,
        mock_set_seed,
        mock_init_dist,
        mock_rerun,
        mock_logging,
        mock_global_vars,
        mock_parse_args,
        mock_args,
    ):
        """Test basic initialize_megatron call"""
        mock_args.layerwise_disaggregated_training = True
        mock_parse_args.return_value = mock_args
        mock_get_args.return_value = mock_args

        result = initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=False)

        assert result is None
        mock_parse_args.assert_called_once()
        mock_global_vars.assert_called_once()
        mock_logging.assert_called_once()
        mock_rerun.assert_called_once()
        mock_init_dist.assert_called_once()
        mock_set_seed.assert_called_once()
        mock_autoresume.assert_called_once()
        mock_compile.assert_called_once()

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    def test_initialize_megatron_no_cuda_error(self, mock_parse_args):
        """Test initialize_megatron raises error when CUDA not available and not allowed"""
        mock_parse_args.return_value = MockArgs()

        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(ValueError, match="Megatron requires CUDA"):
                initialize_megatron(allow_no_cuda=False)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.set_global_variables')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.setup_logging')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.initialize_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    def test_initialize_megatron_skip_mpu(
        self,
        mock_features_manager,
        mock_vdp,
        mock_get_args,
        mock_rerun,
        mock_logging,
        mock_global_vars,
        mock_parse_args,
    ):
        """Test initialize_megatron with skip_mpu_initialization=True"""
        args = MockArgs()
        args.layerwise_disaggregated_training = True
        mock_parse_args.return_value = args
        mock_get_args.return_value = args

        result = initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

        assert result is None

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.set_global_variables')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.setup_logging')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.initialize_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.mpu')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    def test_initialize_megatron_lazy_mpu(
        self,
        mock_features_manager,
        mock_vdp,
        mock_get_args,
        mock_mpu,
        mock_rerun,
        mock_logging,
        mock_global_vars,
        mock_parse_args,
    ):
        """Test initialize_megatron with lazy_mpu_init=True"""
        args = MockArgs()
        args.lazy_mpu_init = True
        args.layerwise_disaggregated_training = True
        mock_parse_args.return_value = args
        mock_get_args.return_value = args

        result = initialize_megatron(allow_no_cuda=True)

        assert callable(result)
        mock_mpu.set_tensor_model_parallel_world_size.assert_called_once()
        mock_mpu.set_tensor_model_parallel_rank.assert_called_once()


class TestInitializeDistributed:
    """Tests for _initialize_distributed function"""

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.mpu')
    def test_initialize_distributed_already_initialized(self, mock_mpu, mock_get_args, mock_args):
        """Test _initialize_distributed when torch.distributed is already initialized"""
        mock_get_args.return_value = mock_args

        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_rank', return_value=0):
                with patch('torch.distributed.get_world_size', return_value=1):
                    _initialize_distributed(None, None)

                    # Should not call init_process_group
                    assert not mock_mpu.initialize_model_parallel.called

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.mpu')
    def test_initialize_distributed_new(self, mock_mpu, mock_get_args, mock_args):
        """Test _initialize_distributed with new initialization"""
        mock_get_args.return_value = mock_args
        mock_mpu.model_parallel_is_initialized.return_value = False

        with patch('torch.distributed.is_initialized', return_value=False):
            with patch('torch.distributed.init_process_group'):
                with patch('torch.cuda.device_count', return_value=1):
                    with patch('torch.cuda.set_device'):
                        _initialize_distributed(None, None)

                        mock_mpu.initialize_model_parallel.assert_called_once()


class TestValidateArgsLdt:
    """Tests for validate_args_ldt function"""

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._validate_args_ldt')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._add_dummy_args_v2')
    def test_validate_args_ldt_basic(self, mock_add_dummy, mock_validate, mock_features_manager, mock_args):
        """Test validate_args_ldt function"""
        mock_validate.return_value = mock_args

        result = validate_args_ldt(mock_args)

        assert result == mock_args
        mock_features_manager.pre_validate_features_args.assert_called_once_with(mock_args)
        mock_validate.assert_called_once_with(mock_args, {})
        mock_features_manager.post_validate_features_args.assert_called_once()
        mock_add_dummy.assert_called_once_with(mock_args)
        mock_features_manager.validate_features_args.assert_called_once()


class TestValidateArgsLdtInternal:
    """Tests for _validate_args_ldt function"""

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_invalid_ckpt_type(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with invalid checkpoint type"""
        mock_args.non_persistent_ckpt_type = 'invalid'

        with pytest.raises(AssertionError, match="Currently only global and local checkpoints are supported"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_invalid_legacy_model_format(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with invalid legacy model checkpoint format"""
        mock_args.use_legacy_models = True
        mock_args.ckpt_format = 'torch_dist'

        with pytest.raises(AssertionError, match="legacy model format only supports"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_invalid_batch_size(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with invalid batch_size argument"""
        mock_args.batch_size = 10

        with pytest.raises(AssertionError, match="--batch-size argument is no longer valid"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_invalid_warmup(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with invalid warmup argument"""
        mock_args.warmup = 0.1

        with pytest.raises(AssertionError, match="--warmup argument is no longer valid"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_invalid_model_parallel_size(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with invalid model_parallel_size argument"""
        mock_args.model_parallel_size = 2

        with pytest.raises(AssertionError, match="--model-parallel-size is no longer valid"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_invalid_checkpoint_activations(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with invalid checkpoint_activations argument"""
        mock_args.checkpoint_activations = True

        with pytest.raises(AssertionError, match="--checkpoint-activations is no longer valid"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_micro_batch_size_zero(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with zero micro_batch_size"""
        mock_args.micro_batch_size = 0

        with pytest.raises(AssertionError, match="micro_batch_size must be greater than 0"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_missing_required_args(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with missing required argument"""
        mock_args.num_layers = None

        with pytest.raises(AssertionError):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_fp16_and_bf16_both_enabled(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with both fp16 and bf16 enabled"""
        mock_args.fp16 = True
        mock_args.bf16 = True

        with pytest.raises(AssertionError, match="fp16 and bf16 cannot both be enabled"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_validate_args_ldt_valid_args(self, mock_vdp, mock_args):
        """Test _validate_args_ldt with valid arguments"""
        result = _validate_args_ldt(mock_args)

        assert result is not None
        # Check some defaults were set
        assert mock_args.params_dtype == torch.bfloat16
        assert mock_args.dataloader_type == 'single'
        assert mock_args.consumed_train_samples == 0


class TestInitializeMegatronExtended:
    """Extended tests for initialize_megatron - checkpoint conversion, extra paths"""

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    def test_ckpt_convert_no_save(self, mock_parse_args):
        """ckpt_convert_format set but ckpt_convert_save None -> ValueError"""
        args = MockArgs()
        args.ckpt_convert_format = 'torch'
        args.ckpt_convert_save = None
        mock_parse_args.return_value = args
        with pytest.raises(ValueError, match="--ckpt-convert-save is required"):
            initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    def test_ckpt_convert_no_load(self, mock_parse_args):
        """ckpt_convert_format set but load None -> ValueError"""
        args = MockArgs()
        args.ckpt_convert_format = 'torch'
        args.ckpt_convert_save = '/save'
        args.load = None
        mock_parse_args.return_value = args
        with pytest.raises(ValueError, match="--load is required"):
            initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    def test_use_checkpoint_args_no_load(self, mock_parse_args):
        """use_checkpoint_args=True but load=None -> ValueError"""
        args = MockArgs()
        args.use_checkpoint_args = True
        args.load = None
        mock_parse_args.return_value = args
        with pytest.raises(ValueError, match="--use-checkpoint-args requires --load argument"):
            initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch.dict(
        'sys.modules',
        {
            'nvidia_resiliency_ext': MagicMock(),
            'nvidia_resiliency_ext.checkpointing': MagicMock(),
            'nvidia_resiliency_ext.checkpointing.local': MagicMock(),
            'nvidia_resiliency_ext.checkpointing.local.ckpt_managers': MagicMock(),
            'nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager': MagicMock(),
        },
    )
    def test_use_checkpoint_args_local_ckpt(self, mock_parse_args):
        """use_checkpoint_args=True with non_persistent_ckpt_type=local -> ValueError"""
        args = MockArgs()
        args.use_checkpoint_args = True
        args.load = '/load'
        args.non_persistent_ckpt_type = 'local'
        mock_parse_args.return_value = args
        with pytest.raises(
            ValueError, match="--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local"
        ):
            initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.set_global_variables')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.setup_logging')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.initialize_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._initialize_distributed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._set_random_seed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._init_autoresume')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._compile_dependencies')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.init_persistent_async_worker')
    def test_async_save_path(
        self,
        mock_async_worker,
        mock_features_manager,
        mock_vdp,
        mock_get_args,
        mock_compile,
        mock_autoresume,
        mock_set_seed,
        mock_init_dist,
        mock_rerun,
        mock_logging,
        mock_global_vars,
        mock_parse_args,
        mock_args,
    ):
        """async_save=True, use_persistent_ckpt_worker=True -> init_persistent_async_worker called"""
        mock_args.async_save = True
        mock_args.use_persistent_ckpt_worker = True
        mock_args.layerwise_disaggregated_training = True
        mock_parse_args.return_value = mock_args
        mock_get_args.return_value = mock_args
        result = initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=False)
        assert result is None
        mock_async_worker.assert_called_once()

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    def test_num_layer_list_wrong_length(self, mock_parse_args):
        """num_layer_list length != pp_size + 1 -> ValueError"""
        args = MockArgs()
        args.num_layer_list = '1,2,3'
        args.pipeline_model_parallel_size = 1
        args.layerwise_disaggregated_training = True
        mock_parse_args.return_value = args
        with pytest.raises(ValueError, match=r"len\(args.num_layer_list\) != args.pipeline_model_parallel_size \+ 1"):
            initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.set_global_variables')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.setup_logging')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.initialize_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._initialize_distributed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._set_random_seed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._init_autoresume')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._compile_dependencies')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.init_persistent_async_worker')
    def test_num_layer_list_valid(
        self,
        mock_async_worker,
        mock_features_manager,
        mock_vdp,
        mock_get_args,
        mock_compile,
        mock_autoresume,
        mock_set_seed,
        mock_init_dist,
        mock_rerun,
        mock_logging,
        mock_global_vars,
        mock_parse_args,
        mock_args,
    ):
        """num_layer_list with valid length passes and gets transformed"""
        mock_args.num_layer_list = '1,2,3'
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        mock_args.layerwise_disaggregated_training = True
        mock_parse_args.return_value = mock_args
        mock_get_args.return_value = mock_args
        # Num layers used in validation: 12, pp_size=2, 12%2=0, so pass
        # total_model_size = 0 + 2 = 2, world_size=2, 2%2=0, pass
        result = initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=False)
        assert result is None

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.set_global_variables')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.setup_logging')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.initialize_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._initialize_distributed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._set_random_seed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._init_autoresume')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._compile_dependencies')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.validate_yaml')
    def test_non_ldt_yaml_path(
        self,
        mock_validate_yaml,
        mock_features_manager,
        mock_vdp,
        mock_get_args,
        mock_compile,
        mock_autoresume,
        mock_set_seed,
        mock_init_dist,
        mock_rerun,
        mock_logging,
        mock_global_vars,
        mock_parse_args,
        mock_args,
    ):
        """layerwise_disaggregated_training=False with yaml_cfg -> validate_yaml path"""
        mock_args.layerwise_disaggregated_training = False
        mock_args.yaml_cfg = 'config.yaml'
        mock_parse_args.return_value = mock_args
        mock_get_args.return_value = mock_args
        result = initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
        assert result is None
        mock_validate_yaml.assert_called_once()

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.set_global_variables')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.setup_logging')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.initialize_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._initialize_distributed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._set_random_seed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._init_autoresume')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._compile_dependencies')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.validate_args')
    def test_non_ldt_validate_path(
        self,
        mock_validate,
        mock_features_manager,
        mock_vdp,
        mock_get_args,
        mock_compile,
        mock_autoresume,
        mock_set_seed,
        mock_init_dist,
        mock_rerun,
        mock_logging,
        mock_global_vars,
        mock_parse_args,
        mock_args,
    ):
        """layerwise_disaggregated_training=False without yaml_cfg -> validate_args path"""
        mock_args.layerwise_disaggregated_training = False
        mock_args.yaml_cfg = None
        mock_parse_args.return_value = mock_args
        mock_get_args.return_value = mock_args
        result = initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
        assert result is None
        mock_validate.assert_called_once()

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.set_global_variables')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.setup_logging')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.initialize_rerun_state_machine')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._initialize_distributed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._set_random_seed')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._init_autoresume')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._compile_dependencies')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._initialize_tp_communicators')
    def test_tp_comm_overlap_path(
        self,
        mock_tp_comm,
        mock_features_manager,
        mock_vdp,
        mock_get_args,
        mock_compile,
        mock_autoresume,
        mock_set_seed,
        mock_init_dist,
        mock_rerun,
        mock_logging,
        mock_global_vars,
        mock_parse_args,
        mock_args,
    ):
        """tp_comm_overlap=True -> _initialize_tp_communicators called"""
        mock_args.tp_comm_overlap = True
        mock_args.tensor_model_parallel_size = 2
        mock_args.world_size = 2
        mock_args.sequence_parallel = True
        mock_args.layerwise_disaggregated_training = True
        mock_parse_args.return_value = mock_args
        mock_get_args.return_value = mock_args
        # tp=2, so sequence_parallel won't be disabled
        # CUDA_DEVICE_MAX_CONNECTIONS check at line 846: tp=2, get_device_arch_version() < 10
        # We need to either mock get_device_arch_version >= 10 or set CUDA_DEVICE_MAX_CONNECTIONS=1
        with patch.dict('os.environ', {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}, clear=True):
            result = initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=False)
        assert result is None
        mock_tp_comm.assert_called_once()

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.parse_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.MindSpeedFeaturesManager')
    def test_use_checkpoint_args_local_ckpt_in_validate(
        self, mock_features_manager, mock_vdp, mock_get_args, mock_parse_args
    ):
        """Test use_checkpoint_args with local ckpt via parsed_args to avoid parse_args call"""
        # Manually provide parsed_args to skip the parse_args call
        args = MockArgs()
        args.use_checkpoint_args = True
        args.load = '/load'
        args.non_persistent_ckpt_type = 'local'
        mock_get_args.return_value = args
        # skip_mpu True avoids need for further setup
        # But this will hit the validate_args_ldt code path where use_checkpoint_args is not checked
        # Actually the use_checkpoint_args check is in initialize_megatron before validate_args_ldt
        # But if we use parsed_args, we need to test that path.
        # Let's test via initialize_megatron with parsed_args
        with pytest.raises(
            ValueError, match="--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local"
        ):
            initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True, parsed_args=args)


class TestValidateArgsLdtExtended:
    """Extended tests for _validate_args_ldt to cover all branches"""

    # ============== 2.1 Early validation checks ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('importlib.util.find_spec', return_value=MagicMock())
    def test_local_ckpt_type_passes(self, mock_find_spec, mock_vdp, mock_args):
        """non_persistent_ckpt_type='local' with mocked nvidia_resiliency_ext -> passes"""
        mock_args.non_persistent_ckpt_type = 'local'
        mock_args.non_persistent_local_ckpt_dir = '/ckpt'
        result = _validate_args_ldt(mock_args)
        assert result is not None

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_local_ckpt_type_import_error(self, mock_vdp, mock_args):
        """non_persistent_ckpt_type='local' without nvidia_resiliency_ext -> RuntimeError"""
        mock_args.non_persistent_ckpt_type = 'local'
        with pytest.raises(RuntimeError, match="nvidia_resiliency_ext is required"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_encoder_tp_mismatch(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """encoder_tp != tp when encoder_pp=0 and num_experts=0 -> AssertionError"""
        mock_args.encoder_pipeline_model_parallel_size = 0
        mock_args.encoder_tensor_model_parallel_size = 2
        mock_args.tensor_model_parallel_size = 1
        mock_args.num_experts = 0  # Must be 0, not None, to trigger the check at line 322
        mock_args.world_size = 1
        with pytest.raises(AssertionError, match="If non-MOE encoder shares first decoder pipeline rank"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_encoder_attn_heads_not_divisible(self, mock_vdp, mock_args):
        """num_attention_heads not divisible by encoder_tp -> AssertionError"""
        mock_args.encoder_tensor_model_parallel_size = 5
        mock_args.world_size = 1
        with pytest.raises(
            AssertionError, match="num_attention_heads must be divisible by encoder_tensor_model_parallel_size"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_world_size_not_divisible(self, mock_vdp, mock_args):
        """world_size not divisible by total_model_size -> AssertionError"""
        mock_args.world_size = 3
        mock_args.tensor_model_parallel_size = 2
        with pytest.raises(AssertionError, match="world size"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_attention_backend_local_no_spec(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """attention_backend=local but spec not matching -> AssertionError"""
        mock_args.attention_backend = AttnBackend.local
        mock_args.spec = ['transformer']
        with pytest.raises(AssertionError, match="--attention-backend local is only supported with --spec local"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_device_arch_version', return_value=10)
    def test_mixed_encoder_pp_gt_zero(self, mock_dev_arch, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """encoder_pp>0 and encoder_tp=0 -> encoder_tp set to tp"""
        mock_args.encoder_pipeline_model_parallel_size = 1
        mock_args.encoder_tensor_model_parallel_size = 0
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 2
        # total = 2*1*1 + 2*2*1 = 6
        mock_args.world_size = 6
        with patch.dict('os.environ', {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}, clear=True):
            _validate_args_ldt(mock_args)
        assert mock_args.encoder_tensor_model_parallel_size == 2

    # ============== 2.2 Pipeline model parallel split ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_pipeline_split_rank_pp_zero(self, mock_vdp, mock_args):
        """pipeline_model_parallel_split_rank >= pp -> AssertionError"""
        mock_args.pipeline_model_parallel_split_rank = 2
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        with pytest.raises(AssertionError, match="pipeline_model_parallel_size must be greater than 0"):
            _validate_args_ldt(mock_args)

    # ============== 2.3 Hierarchical context parallel ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_device_arch_version', return_value=10)
    def test_hierarchical_cp_product_mismatch(self, mock_dev_arch, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """hierarchical_cp_sizes product != context_parallel_size -> AssertionError"""
        mock_args.hierarchical_context_parallel_sizes = [2, 2]
        mock_args.context_parallel_size = 2
        mock_args.world_size = 2
        # total = 0 + 2 = 2, 2%2=0, passes world_size check
        with pytest.raises(
            AssertionError, match="context_parallel_size must equal product of hierarchical_context_parallel_sizes"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_cp_comm_type_a2a_p2p_no_hierarchical(self, mock_vdp, mock_args):
        """a2a+p2p in cp_comm_type but hierarchical_context_parallel_sizes is None -> AssertionError"""
        mock_args.cp_comm_type = 'a2a+p2p'
        mock_args.hierarchical_context_parallel_sizes = None
        with pytest.raises(AssertionError, match="--hierarchical-context-parallel-sizes must be set"):
            _validate_args_ldt(mock_args)

    # ============== 2.4 Data path validation ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_data_path_no_split(self, mock_vdp, mock_args):
        """data_path set without split -> split gets default value"""
        mock_args.data_path = '/data'
        mock_args.split = None
        result = _validate_args_ldt(mock_args)
        assert result is not None
        assert mock_args.split == '969, 30, 1'

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_data_path_and_data_args_path(self, mock_vdp, mock_args):
        """Both data_path and data_args_path set -> AssertionError"""
        mock_args.data_path = '/data'
        mock_args.data_args_path = '/data_args'
        with pytest.raises(AssertionError, match="Only one of data_path or data_args_path should be specified"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_per_split_data_path_conflict(self, mock_vdp, mock_args):
        """Both train_data_path and per_split_data_args_path set -> AssertionError"""
        mock_args.train_data_path = '/train'
        mock_args.per_split_data_args_path = '/args'
        with pytest.raises(AssertionError, match="Only one of per_split_data_path"):
            _validate_args_ldt(mock_args)

    # ============== 2.5 Global batch size ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_micro_batch_size_none(self, mock_vdp, mock_args):
        """micro_batch_size is None -> AssertionError"""
        mock_args.micro_batch_size = None
        with pytest.raises(AssertionError, match="micro_batch_size must not be None"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    def test_global_batch_size_zero_or_negative(self, mock_vdp, mock_args):
        """global_batch_size = 0 -> AssertionError"""
        mock_args.global_batch_size = 0
        with pytest.raises(AssertionError, match="global_batch_size must be greater than 0"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_global_batch_size_auto_calc(self, mock_print, mock_vdp, mock_args):
        """global_batch_size is None -> auto calculated from micro_batch_size * dp_size"""
        mock_args.global_batch_size = None
        mock_args.micro_batch_size = 4
        # dp_size = world_size // total_model_size = 1 // 1 = 1
        result = _validate_args_ldt(mock_args)
        assert result.global_batch_size == 4 * 1

    # ============== 2.6 Virtual pipeline parallelism ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_both_vp_args_set(self, mock_print, mock_vdp, mock_args):
        """Both virtual pipeline args set -> AssertionError"""
        mock_args.num_layers_per_virtual_pipeline_stage = 2
        mock_args.num_virtual_stages_per_pipeline_rank = 2
        with pytest.raises(AssertionError, match="cannot be set at the same time"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_vp_overlap_p2p_pp_1(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """vp with overlap_p2p_comm but pp=1 -> AssertionError (unless pp > 1)"""
        mock_args.num_layers_per_virtual_pipeline_stage = 2
        mock_args.overlap_p2p_comm = True
        # pp=1, overlap_p2p_comm=True, 1 <= 1 -> AssertionError
        # world_size=1, total=1, 1%1=0, passes
        with pytest.raises(AssertionError, match="pipeline-model-parallel size should be greater than 1"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_vp_no_overlap_p2p_pp_2(self, mock_print, mock_vdp, mock_args):
        """vp without overlap_p2p_comm but pp<=2 -> AssertionError"""
        mock_args.num_layers_per_virtual_pipeline_stage = 2
        mock_args.overlap_p2p_comm = False
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        with pytest.raises(AssertionError, match="p2p communication overlap is disabled"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_vp_uneven_decoder_pipeline(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """vp with decoder_first/last_pipeline_num_layers but no num_virtual_stages -> AssertionError"""
        mock_args.num_layers_per_virtual_pipeline_stage = 2
        mock_args.decoder_first_pipeline_num_layers = 2
        # pp=1 by default, overlap_p2p_comm=False by default, so with vp args, check becomes:
        # line 476: else: if pp <= 2 -> 1 <= 2 -> AssertionError: pp > 2 required
        # Need pp > 2. Set pp=3, world_size=3
        # Actually the test needs to reach line 485-488, so need to bypass pp checks
        # Set overlap_p2p_comm=True so check at line 471 is: pp <= 1, pp=3, 3>1, passes
        mock_args.overlap_p2p_comm = True
        mock_args.pipeline_model_parallel_size = 3
        mock_args.world_size = 3
        # Now reaches line 484: num_virtual_stages_per_pipeline_rank is None -> True
        # Line 485: decoder_first_pipeline_num_layers=2 not None -> AssertionError
        with pytest.raises(AssertionError, match="please use --num-virtual-stages-per-pipeline-rank"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_vp_num_layers_not_divisible(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """vp: num_layers not divisible by pp_size -> AssertionError"""
        mock_args.num_layers_per_virtual_pipeline_stage = 2
        mock_args.num_layers = 11
        # pp=1, overlap_p2p_comm=False by default, line 476: pp <= 2 -> 1 <= 2, would fail early
        # Need to reach line 500, so bypass the pp check: set overlap_p2p_comm=True, pp=3, world_size=3
        mock_args.overlap_p2p_comm = True
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        # overlap_p2p_comm=True, pp=2, 2<=1? No, 2>1, passes line 471
        # Reaches line 484: num_virtual_stages_per_pipeline_rank is None -> True
        # decoder_first/last are None, so line 485 passes
        # num_layers=11, 11%2 != 0 -> AssertionError at line 500-501
        with pytest.raises(
            AssertionError, match="number of layers of the model must be divisible pipeline model parallel size"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_vp_layers_per_stage_not_divisible(self, mock_print, mock_vdp, mock_args):
        """vp: layers per stage not divisible by vp_stage_size -> AssertionError"""
        mock_args.num_layers_per_virtual_pipeline_stage = 3
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        # Skip overlap_p2p_comm check: pp=2, 2<=2, fails. So set pp=3, world_size=3
        mock_args.pipeline_model_parallel_size = 3
        mock_args.world_size = 3
        mock_args.num_layers = 10
        # total_model_size = 0 + 3 = 3, 3%3=0, passes
        # num_layers=10, 10%3=1 -> AssertionError at 500-501
        # Actually the check is num_layers % transformer_pp (pp=3), 10%3=1, fails first
        with pytest.raises(AssertionError):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_non_interleaved_num_layers_not_divisible(self, mock_print, mock_vdp, mock_args):
        """non-interleaved: num_layers not divisible by pp_size -> AssertionError"""
        mock_args.num_layers = 11
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        # total = 0 + 2 = 2, 2%2 = 0, dp = 1, passes
        # No vp args, so num_layers(11) % 2 != 0
        with pytest.raises(AssertionError, match="Number of layers should be divisible"):
            _validate_args_ldt(mock_args)

    # ============== 2.7 Data parallel sharding ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_dp_sharding_optim_grads_params(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """data_parallel_sharding_strategy='optim_grads_params' -> overlap flags set"""
        mock_args.data_parallel_sharding_strategy = 'optim_grads_params'
        mock_args.use_distributed_optimizer = True
        # Must set use_dist_ckpt to pass the gloo check at line 984
        mock_args.use_dist_ckpt = True
        mock_args.ckpt_format = 'torch_dist'
        result = _validate_args_ldt(mock_args)
        assert result.overlap_param_gather is True
        assert result.overlap_grad_reduce is True

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_dp_sharding_optim_grads(self, mock_print, mock_vdp, mock_args):
        """data_parallel_sharding_strategy='optim_grads' -> overlap_grad_reduce set"""
        mock_args.data_parallel_sharding_strategy = 'optim_grads'
        result = _validate_args_ldt(mock_args)
        assert result.overlap_grad_reduce is True

    # ============== 2.8 overlap_param_gather ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_overlap_param_gather_no_dist_opt(self, mock_print, mock_vdp, mock_args):
        """overlap_param_gather without dist_opt -> AssertionError"""
        mock_args.overlap_param_gather = True
        mock_args.use_distributed_optimizer = False
        with pytest.raises(AssertionError, match="--overlap-param-gather only supported with distributed optimizer"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_overlap_param_gather_no_overlap_grad_reduce(self, mock_print, mock_vdp, mock_args):
        """overlap_param_gather without overlap_grad_reduce -> AssertionError"""
        mock_args.overlap_param_gather = True
        mock_args.use_distributed_optimizer = True
        mock_args.overlap_grad_reduce = False
        with pytest.raises(AssertionError, match="Must use --overlap-param-gather with --overlap-grad-reduce"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_overlap_param_gather_legacy_models(self, mock_print, mock_vdp, mock_args):
        """overlap_param_gather with legacy_models -> AssertionError"""
        mock_args.overlap_param_gather = True
        mock_args.use_distributed_optimizer = True
        mock_args.overlap_grad_reduce = True
        mock_args.use_legacy_models = True
        with pytest.raises(AssertionError, match="--overlap-param-gather only supported with MCore models"):
            _validate_args_ldt(mock_args)

    # ============== 2.9 use_torch_fsdp2 ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fsdp2_pp_not_1(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fsdp2 with pp!=1 -> AssertionError"""
        mock_args.use_torch_fsdp2 = True
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        with pytest.raises(AssertionError, match="--use-torch-fsdp2 is not supported with pipeline parallelism"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fsdp2_ep_not_1(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fsdp2 with ep!=1 -> AssertionError"""
        mock_args.use_torch_fsdp2 = True
        mock_args.expert_model_parallel_size = 2
        mock_args.num_experts = 2
        with pytest.raises(AssertionError, match="--use-torch-fsdp2 is not supported with expert parallelism"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fsdp2_use_dist_opt(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fsdp2 with use_distributed_optimizer -> AssertionError"""
        mock_args.use_torch_fsdp2 = True
        mock_args.use_distributed_optimizer = True
        with pytest.raises(AssertionError, match="--use-torch-fsdp2 is not supported with MCore"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fsdp2_grad_accum_fusion(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fsdp2 with gradient_accumulation_fusion -> AssertionError"""
        mock_args.use_torch_fsdp2 = True
        mock_args.gradient_accumulation_fusion = True
        with pytest.raises(
            AssertionError, match="--use-torch-fsdp2 is not supported with gradient accumulation fusion"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fsdp2_ckpt_format(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fsdp2 with ckpt_format not in allowed list -> AssertionError"""
        mock_args.use_torch_fsdp2 = True
        mock_args.ckpt_format = 'torch'
        with pytest.raises(AssertionError, match="--use-torch-fsdp2 requires --ckpt-format"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fsdp2_untie_embeddings(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fsdp2 with untie_embeddings_and_output_weights=False -> AssertionError"""
        mock_args.use_torch_fsdp2 = True
        mock_args.ckpt_format = 'torch_dcp'  # Must be in allowed list to pass ckpt_format check
        mock_args.untie_embeddings_and_output_weights = False
        with pytest.raises(AssertionError, match="--use-torch-fsdp2 requires --untie-embeddings-and-output-weights"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fsdp2_fp16(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fsdp2 with fp16 -> AssertionError"""
        mock_args.use_torch_fsdp2 = True
        mock_args.ckpt_format = 'torch_dcp'  # Must be in allowed list
        mock_args.fp16 = True
        mock_args.bf16 = False
        with pytest.raises(AssertionError, match="--use-torch-fsdp2 not supported with fp16 yet"):
            _validate_args_ldt(mock_args)

    # ============== 2.10 overlap_param_gather_with_optimizer_step ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_overlap_param_gather_step_no_dist_opt(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """overlap_param_gather_with_optimizer_step without dist_opt -> AssertionError"""
        mock_args.overlap_param_gather_with_optimizer_step = True
        mock_args.use_distributed_optimizer = False
        with pytest.raises(
            AssertionError, match="--overlap-param-gather-with-optimizer-step only supported with distributed optimizer"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_overlap_param_gather_step_no_overlap_param(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """overlap_param_gather_with_optimizer_step without overlap_param_gather -> AssertionError"""
        mock_args.overlap_param_gather_with_optimizer_step = True
        mock_args.use_distributed_optimizer = True
        mock_args.overlap_param_gather = False
        with pytest.raises(
            AssertionError, match="Must use --overlap-param-gather-with-optimizer-step with --overlap-param-gather"
        ):
            _validate_args_ldt(mock_args)

    # ============== 2.11 fp8_param_gather ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fp8_param_gather_no_dist_opt_no_fsdp2(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fp8_param_gather without dist_opt and without fsdp2 -> AssertionError"""
        mock_args.fp8_param_gather = True
        mock_args.use_distributed_optimizer = False
        mock_args.use_torch_fsdp2 = False
        with pytest.raises(AssertionError, match="--fp8-param-gather only supported with distributed optimizer"):
            _validate_args_ldt(mock_args)

    # ============== 2.12 use_custom_fsdp ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_custom_fsdp_no_dist_opt(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """use_custom_fsdp without dist_opt -> AssertionError"""
        mock_args.use_custom_fsdp = True
        mock_args.use_distributed_optimizer = False
        with pytest.raises(AssertionError, match="--use-custom-fsdp only supported with distributed optimizer"):
            _validate_args_ldt(mock_args)

    # ============== 2.13 bf16 specific ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_bf16_accumulate_allreduce_wrong_dtype(self, mock_print, mock_vdp, mock_args):
        """bf16 with accumulate_allreduce_grads_in_fp32 but wrong main_grads_dtype -> AssertionError"""
        mock_args.bf16 = True
        mock_args.accumulate_allreduce_grads_in_fp32 = True
        mock_args.main_grads_dtype = 'bf16'
        with pytest.raises(AssertionError, match="--main-grads-dtype can only be fp32"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_bf16_grad_reduce_in_bf16(self, mock_print, mock_vdp, mock_args):
        """bf16 with grad_reduce_in_bf16 -> accumulate_allreduce_grads_in_fp32 set to False"""
        mock_args.bf16 = True
        mock_args.grad_reduce_in_bf16 = True
        result = _validate_args_ldt(mock_args)
        assert result.accumulate_allreduce_grads_in_fp32 is False

    # ============== 2.14 num_dataset_builder_threads ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_num_dataset_builder_threads_zero(self, mock_print, mock_vdp, mock_args):
        """num_dataset_builder_threads = 0 -> AssertionError"""
        mock_args.num_dataset_builder_threads = 0
        with pytest.raises(AssertionError, match="num_dataset_builder_threads must be greater than 0"):
            _validate_args_ldt(mock_args)

    # ============== 2.15 train_iters + train_samples checks ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_train_iters_with_train_samples(self, mock_print, mock_vdp, mock_args):
        """train_iters and train_samples both set -> AssertionError"""
        mock_args.train_iters = 100
        mock_args.train_samples = 100
        with pytest.raises(AssertionError, match="expected iteration-based training"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_train_iters_with_lr_decay_samples(self, mock_print, mock_vdp, mock_args):
        """train_iters with lr_decay_samples -> AssertionError"""
        mock_args.train_iters = 100
        mock_args.lr_decay_samples = 10
        with pytest.raises(AssertionError, match="expected iteration-based learning rate decay"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_train_iters_with_lr_warmup_samples(self, mock_print, mock_vdp, mock_args):
        """train_iters with lr_warmup_samples -> AssertionError"""
        mock_args.train_iters = 100
        mock_args.lr_warmup_samples = 10
        with pytest.raises(AssertionError, match="expected iteration-based learning rate warmup"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_train_iters_with_rampup_batch_size(self, mock_print, mock_vdp, mock_args):
        """train_iters with rampup_batch_size -> AssertionError"""
        mock_args.train_iters = 100
        mock_args.rampup_batch_size = 10
        with pytest.raises(AssertionError, match="expected no batch-size rampup"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_train_iters_with_lr_warmup_fraction_and_iters(self, mock_print, mock_vdp, mock_args):
        """train_iters with both lr_warmup_fraction and lr_warmup_iters -> AssertionError"""
        mock_args.train_iters = 100
        mock_args.lr_warmup_fraction = 0.1
        mock_args.lr_warmup_iters = 10
        with pytest.raises(AssertionError, match="can only specify one of"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_train_samples_with_lr_decay_iters(self, mock_print, mock_vdp, mock_args):
        """train_samples with lr_decay_iters -> AssertionError"""
        mock_args.train_samples = 100
        mock_args.lr_decay_iters = 10
        with pytest.raises(AssertionError, match="expected sample-based learning rate decay"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_train_samples_with_lr_warmup_iters(self, mock_print, mock_vdp, mock_args):
        """train_samples with lr_warmup_iters -> AssertionError"""
        mock_args.train_samples = 100
        mock_args.lr_warmup_iters = 10
        with pytest.raises(AssertionError, match="expected sample-based learnig rate warmup"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_train_samples_with_lr_warmup_fraction_and_samples(self, mock_print, mock_vdp, mock_args):
        """train_samples with both lr_warmup_fraction and lr_warmup_samples -> AssertionError"""
        mock_args.train_samples = 100
        mock_args.lr_warmup_fraction = 0.1
        mock_args.lr_warmup_samples = 10
        with pytest.raises(AssertionError, match="can only specify one of"):
            _validate_args_ldt(mock_args)

    # ============== 2.16 num_layers and encoder_num_layers ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_num_layers_and_encoder_num_layers(self, mock_print, mock_vdp, mock_args):
        """both num_layers and encoder_num_layers specified -> AssertionError"""
        mock_args.num_layers = 12
        mock_args.encoder_num_layers = 12
        with pytest.raises(AssertionError, match="cannot have both num-layers and encoder-num-layers"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_no_num_layers_no_encoder(self, mock_print, mock_vdp, mock_args):
        """neither num_layers nor encoder_num_layers specified -> AssertionError"""
        mock_args.num_layers = None
        mock_args.encoder_num_layers = None
        with pytest.raises(AssertionError, match="either num-layers or encoder-num-layers should be specified"):
            _validate_args_ldt(mock_args)

    # ============== 2.17 KV channels, seq_length, etc ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_hidden_size_not_divisible(self, mock_print, mock_vdp, mock_args):
        """hidden_size not divisible by num_attention_heads -> AssertionError"""
        mock_args.hidden_size = 769
        with pytest.raises(AssertionError, match="hidden_size must be divisible by num_attention_heads"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_seq_length_not_multiple_of_2cp(self, mock_print, mock_vdp, mock_args):
        """seq_length not divisible by 2*context_parallel_size -> AssertionError"""
        mock_args.context_parallel_size = 3
        mock_args.world_size = 3
        mock_args.tensor_model_parallel_size = 1
        mock_args.pipeline_model_parallel_size = 1
        # total = 3, 3%3 = 0, passes
        # seq_length=1024, 1024 % (3*2) = 1024%6 != 0
        with pytest.raises(AssertionError, match="seq-length should be a multiple of"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_seq_length_and_encoder_seq_length(self, mock_print, mock_vdp, mock_args):
        """both seq_length and encoder_seq_length specified -> AssertionError"""
        mock_args.seq_length = 1024
        mock_args.encoder_seq_length = 1024
        with pytest.raises(AssertionError, match="Cannot specify both seq_length and encoder_seq_length"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_max_position_embeddings_lt_seq_length(self, mock_print, mock_vdp, mock_args):
        """max_position_embeddings < seq_length -> AssertionError"""
        mock_args.max_position_embeddings = 512
        mock_args.seq_length = 1024
        # seq_length is set, encoder_seq_length is None, so encoder_seq_length = 1024
        with pytest.raises(AssertionError, match="max_position_embeddings"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_decoder_seq_length_gt_max_pos(self, mock_print, mock_vdp, mock_args):
        """decoder_seq_length > max_position_embeddings -> AssertionError"""
        mock_args.decoder_seq_length = 2048
        with pytest.raises(AssertionError, match="max_position_embeddings must be >= decoder_seq_length"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_min_lr_gt_lr(self, mock_print, mock_vdp, mock_args):
        """min_lr > lr -> AssertionError"""
        mock_args.min_lr = 0.1
        mock_args.lr = 0.01
        with pytest.raises(AssertionError, match="min_lr must be <= lr"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_save_no_save_interval(self, mock_print, mock_vdp, mock_args):
        """save set without save_interval -> AssertionError"""
        mock_args.save = '/save'
        mock_args.save_interval = None
        with pytest.raises(AssertionError, match="save_interval must be specified"):
            _validate_args_ldt(mock_args)

    # ============== 2.18 fp16, fp32 residual, moe, weight_decay ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_fp16_lm_cross_entropy_no_fp16(self, mock_print, mock_vdp, mock_args):
        """fp16_lm_cross_entropy without fp16 mode -> AssertionError"""
        mock_args.fp16_lm_cross_entropy = True
        mock_args.fp16 = False
        mock_args.bf16 = True
        with pytest.raises(AssertionError, match="lm cross entropy in fp16 only support in fp16 mode"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_fp32_residual_no_fp16_bf16(self, mock_print, mock_vdp, mock_args):
        """fp32_residual_connection without fp16/bf16 -> AssertionError"""
        mock_args.fp32_residual_connection = True
        mock_args.fp16 = False
        mock_args.bf16 = False
        with pytest.raises(AssertionError, match="residual connection in fp32 only supported when using fp16 or bf16"):
            _validate_args_ldt(mock_args)

    # ============== 2.19 distribute_saved, retro, ep, misc ==============

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_distribute_saved_tp1(self, mock_print, mock_vdp, mock_args):
        """distribute_saved_activations with tp=1 -> AssertionError"""
        mock_args.distribute_saved_activations = True
        mock_args.tensor_model_parallel_size = 1
        with pytest.raises(
            AssertionError, match="can distribute recomputed activations only across tensor model parallel groups"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_distribute_saved_no_full_recompute(self, mock_torch, mock_print, mock_vdp, mock_args):
        """distribute_saved_activations with non-full recompute -> AssertionError"""
        mock_args.distribute_saved_activations = True
        mock_args.tensor_model_parallel_size = 2
        mock_args.world_size = 2
        mock_args.recompute_granularity = 'selective'
        with pytest.raises(
            AssertionError, match="distributed recompute activations is only application to full recompute granularity"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_retro_add_retriever_no_train_samples(self, mock_print, mock_vdp, mock_args):
        """retro_add_retriever without train_samples -> AssertionError"""
        mock_args.retro_add_retriever = True
        mock_args.train_samples = None
        with pytest.raises(AssertionError, match="args.train_samples should be auto-loaded"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_device_arch_version', return_value=10)
    def test_retro_with_sequence_parallel(self, mock_dev_arch, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """retro_add_retriever with sequence_parallel -> AssertionError"""
        mock_args.retro_add_retriever = True
        mock_args.train_samples = 100
        mock_args.sequence_parallel = True
        # Need tp=2 so sp isn't auto-disabled at line 835-838
        mock_args.tensor_model_parallel_size = 2
        mock_args.world_size = 2
        # Need CUDA_DEVICE_MAX_CONNECTIONS=1 for the tp>1 check at line 846
        with patch.dict('os.environ', {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}, clear=True):
            with pytest.raises(AssertionError, match="retro currently does not support sequence parallelism"):
                _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_retro_with_pipeline_parallelism(self, mock_print, mock_vdp, mock_args):
        """retro_add_retriever with pp!=1 -> AssertionError"""
        mock_args.retro_add_retriever = True
        mock_args.train_samples = 100
        mock_args.sequence_parallel = False
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        with pytest.raises(AssertionError, match="retro currently does not support pipeline parallelism"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_expert_parallel_no_num_experts(self, mock_print, mock_vdp, mock_args):
        """expert_model_parallel_size > 1 without num_experts -> AssertionError"""
        mock_args.expert_model_parallel_size = 2
        mock_args.num_experts = None
        mock_args.world_size = 2
        with pytest.raises(AssertionError, match="num_experts must be non None to use expert model parallelism"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    def test_expert_parallel_num_experts_not_divisible(self, mock_print, mock_vdp, mock_args):
        """expert_model_parallel_size > 1 and num_experts not divisible -> AssertionError"""
        mock_args.expert_model_parallel_size = 2
        mock_args.num_experts = 3
        mock_args.world_size = 2
        with pytest.raises(AssertionError, match="Number of experts should be a multiple of"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_device_arch_version', return_value=10)
    def test_tp_gt_1_or_cp_gt_1_no_cuda_max_connections(
        self, mock_dev_arch, mock_torch_ver, mock_print, mock_vdp, mock_args
    ):
        """tp>1 or cp>1 without CUDA_DEVICE_MAX_CONNECTIONS=1 -> AssertionError"""
        mock_args.tensor_model_parallel_size = 2
        mock_args.world_size = 2
        # get_device_arch_version < 10 when fsdp is not used, and CUDA_DEVICE_MAX_CONNECTIONS not set to 1
        mock_dev_arch.return_value = 9
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AssertionError, match="CUDA_DEVICE_MAX_CONNECTIONS"):
                _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_device_arch_version', return_value=10)
    def test_tp_gt_1_blackwell_no_error(self, mock_dev_arch, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """tp>1 on blackwell arch (>=10) with no CUDA_DEVICE_MAX_CONNECTIONS -> no error"""
        mock_args.tensor_model_parallel_size = 2
        mock_args.world_size = 2
        mock_dev_arch.return_value = 10
        # tp>1 enters the CUDA check block. The first check is whether use_torch_fsdp2 or use_custom_fsdp.
        # Since neither is set, the else branch checks CUDA_DEVICE_MAX_CONNECTIONS.
        # With arch=10, the condition `args.cp > 1 and arch < 10` is False, but `tp > 1` is still True.
        # So we still enter the block. Need CUDA_DEVICE_MAX_CONNECTIONS=1 to pass.
        with patch.dict('os.environ', {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}, clear=True):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_inference_batch_times_seqlen(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """inference_batch_times_seqlen_threshold > -1 but pp <= 1 -> AssertionError"""
        mock_args.inference_batch_times_seqlen_threshold = 0
        mock_args.pipeline_model_parallel_size = 1
        mock_args.world_size = 1
        # 0 > -1 is True, 1 <= 1 is True -> AssertionError
        with pytest.raises(
            AssertionError,
            match="--inference-batch-times-seqlen-threshold requires setting --pipeline-model-parallel-size > 1",
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_moe_upcycling_no_save(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """moe_use_upcycling without save -> AssertionError"""
        mock_args.moe_use_upcycling = True
        mock_args.num_experts = 2
        mock_args.save = None
        # Need to pass model parallel checks: ep is 1, no fsdp2
        with pytest.raises(AssertionError, match="When using upcycling, the --save option must be specified"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_optimizer_cpu_offload(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """optimizer_cpu_offload without precision_aware_optimizer -> AssertionError"""
        mock_args.optimizer_cpu_offload = True
        mock_args.use_precision_aware_optimizer = False
        with pytest.raises(AssertionError, match="optimizer cpu offload must be used in conjunction"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('importlib.util.find_spec', return_value=MagicMock())
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_non_persistent_ckpt_type_local_no_dir(
        self, mock_torch_ver, mock_print, mock_find_spec, mock_vdp, mock_args
    ):
        """local ckpt without dir -> AssertionError"""
        mock_args.non_persistent_ckpt_type = 'local'
        mock_args.non_persistent_local_ckpt_dir = None
        with pytest.raises(AssertionError, match="Tried to use local checkpointing without specifying"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_replication_no_jump(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """replication=True without replication_jump -> AssertionError"""
        mock_args.replication = True
        mock_args.replication_jump = None
        with pytest.raises(AssertionError, match="--replication requires the value of --replication-jump"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_replication_not_local_ckpt(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """replication=True with non_local ckpt_type -> AssertionError"""
        mock_args.replication = True
        mock_args.replication_jump = 1
        mock_args.non_persistent_ckpt_type = 'global'
        with pytest.raises(AssertionError, match="--replication requires args.non_persistent_ckpt_type == 'local'"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.update_use_dist_ckpt')
    def test_legacy_dist_ckpt_not_supported(
        self, mock_update_dist_ckpt, mock_torch_ver, mock_print, mock_vdp, mock_args
    ):
        """use_dist_ckpt with use_legacy_models -> RuntimeError"""
        mock_args.use_dist_ckpt = True
        mock_args.use_legacy_models = True
        # Mock update_use_dist_ckpt to preserve use_dist_ckpt=True
        # Set ckpt_format='torch' to pass the legacy model format check at line 318-319
        mock_args.ckpt_format = 'torch'
        with pytest.raises(RuntimeError, match="--use-dist-ckpt is not supported in legacy models"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_enable_gloo_no_dist_opt_use_dist_ckpt(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """gloo disabled with dist_opt but no dist_ckpt -> AssertionError"""
        # Actually the check is: not enable_gloo_process_groups, use_distributed_optimizer, not use_dist_ckpt
        # Default enable_gloo_process_groups = False
        # Default use_distributed_optimizer = False (not set)
        mock_args.enable_gloo_process_groups = False
        mock_args.use_distributed_optimizer = True
        mock_args.use_dist_ckpt = False
        with pytest.raises(AssertionError, match="use_distributed_optimizer requires use_dist_ckpt"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_device_arch_version', return_value=10)
    def test_context_parallel_gt_1_legacy_models(self, mock_dev_arch, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """context_parallel_size > 1 with use_legacy_models -> AssertionError"""
        mock_args.context_parallel_size = 2
        mock_args.use_legacy_models = True
        mock_args.world_size = 2
        # get_device_arch_version=10 bypasses CUDA_DEVICE_MAX_CONNECTIONS check (blackwell)
        with pytest.raises(AssertionError, match="Context parallelism is not supported in legacy models"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_device_arch_version', return_value=10)
    def test_encoder_tp_exceeds_decoder(self, mock_dev_arch, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """encoder_tp > decoder_tp with encoder_pp > 0 passes since tp=2 == encoder_tp=2"""
        mock_args.encoder_pipeline_model_parallel_size = 2
        mock_args.encoder_tensor_model_parallel_size = 2
        mock_args.tensor_model_parallel_size = 2
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 8
        # Line 322 skipped: encoder_pp=2, num_experts=None -> 2 != 0, skip
        # Line 326: encoder_tp=2>0, 12%2=0, pass; 2>2=False, pass
        # Line 332: encoder_pp=2>0, encoder_tp=2>0, not set
        # total = 2*2*1 = 4, decoder = 2*2*1 = 4, total = 8, 8%8=0, pass
        with patch.dict('os.environ', {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}, clear=True):
            result = _validate_args_ldt(mock_args)
        assert result is not None

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_seq_length_encoder_seq_length_none(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """both seq_length and encoder_seq_length are None -> AssertionError"""
        mock_args.seq_length = None
        mock_args.encoder_seq_length = None
        with pytest.raises(AssertionError, match="Either seq_length or encoder_seq_length must be specified"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_fp16_dynamic_loss_scale(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """fp16 with dynamic loss scale -> check_for_nan_in_loss_and_grad = False"""
        mock_args.fp16 = True
        mock_args.bf16 = False
        mock_args.loss_scale = None
        result = _validate_args_ldt(mock_args)
        assert result.check_for_nan_in_loss_and_grad is False

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_bf16_auto_accumulate_allreduce(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """bf16 with default settings -> auto set accumulate_allreduce_grads_in_fp32=True"""
        mock_args.bf16 = True
        mock_args.accumulate_allreduce_grads_in_fp32 = False
        mock_args.main_grads_dtype = 'fp32'
        result = _validate_args_ldt(mock_args)
        assert result.accumulate_allreduce_grads_in_fp32 is True

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_weight_decay_constant_style_with_start(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """weight_decay_incr_style='constant' with start_weight_decay not None -> AssertionError"""
        mock_args.weight_decay_incr_style = 'constant'
        mock_args.start_weight_decay = 0.1
        mock_args.end_weight_decay = None
        with pytest.raises(AssertionError, match="start_weight_decay must be None for constant weight decay style"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_weight_decay_constant_style_with_end(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """weight_decay_incr_style='constant' with end_weight_decay not None -> AssertionError"""
        mock_args.weight_decay_incr_style = 'constant'
        mock_args.start_weight_decay = None
        mock_args.end_weight_decay = 0.1
        with pytest.raises(AssertionError, match="end_weight_decay must be None for constant weight decay style"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_distribute_saved_recompute_method_none(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """distribute_saved_activations with recompute_method None -> AssertionError"""
        mock_args.distribute_saved_activations = True
        mock_args.tensor_model_parallel_size = 2
        mock_args.world_size = 2
        mock_args.recompute_granularity = 'full'
        mock_args.recompute_method = None
        with pytest.raises(AssertionError, match="need to use a recompute method"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_recompute_selective_with_method(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """recompute_granularity='selective' with recompute_method not None -> AssertionError"""
        mock_args.recompute_granularity = 'selective'
        mock_args.recompute_method = 'standard'
        with pytest.raises(
            AssertionError, match="recompute method is not yet supported for selective recomputing granularity"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_sequence_parallel_with_tp1(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """sequence_parallel=True with tp=1 -> sequence_parallel set to False"""
        mock_args.sequence_parallel = True
        mock_args.tensor_model_parallel_size = 1
        result = _validate_args_ldt(mock_args)
        assert result.sequence_parallel is False

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_tp_comm_overlap_no_sequence_parallel(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """tp_comm_overlap without sequence_parallel -> AssertionError"""
        mock_args.tp_comm_overlap = True
        mock_args.sequence_parallel = False
        with pytest.raises(
            AssertionError,
            match="Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled",
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_cuda_device_max_connections_ok(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """tp>1 with CUDA_DEVICE_MAX_CONNECTIONS=1 on non-blackwell -> passes"""
        mock_args.tensor_model_parallel_size = 2
        mock_args.world_size = 2
        with patch.dict('os.environ', {'CUDA_DEVICE_MAX_CONNECTIONS': '1'}, clear=True):
            result = _validate_args_ldt(mock_args)
            assert result is not None

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_decoupled_lr_legacy_models(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """decoupled_lr with use_legacy_models -> AssertionError"""
        mock_args.decoupled_lr = 0.1
        mock_args.use_legacy_models = True
        with pytest.raises(
            AssertionError, match="--decoupled-lr and --decoupled-min-lr is not supported in legacy models"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_rotary_interleaved_rope_fusion(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """rotary_interleaved + apply_rope_fusion -> RuntimeError"""
        mock_args.rotary_interleaved = True
        mock_args.apply_rope_fusion = True
        # use_rotary_position_embeddings is false by default, which at line 891 sets position_embedding_type='rope'?
        # No, line 891: if use_rotary: position_embedding_type='rope'. So since use_rotary=False, position_embedding_type stays 'rope'(default)
        # Actually default position_embedding_type='rope', so line 897 means apply_rope_fusion stays True and position_embedding_type is 'rope'
        # Then at line 893: rotary_interleaved and apply_rope_fusion -> RuntimeError
        with pytest.raises(RuntimeError, match="--rotary-interleaved does not work with rope_fusion"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_rotary_interleaved_legacy_models(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """rotary_interleaved + use_legacy_models -> RuntimeError"""
        mock_args.rotary_interleaved = True
        mock_args.apply_rope_fusion = False
        mock_args.use_legacy_models = True
        # Line 893: rotary_interleaved=True, apply_rope_fusion=False -> skip
        # Line 895: rotary_interleaved=True, use_legacy_models=True -> RuntimeError
        with pytest.raises(RuntimeError, match="--rotary-interleaved is not supported in legacy models"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_no_position_embedding_not_rope(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """no_position_embedding with non-rope position_embedding_type -> RuntimeError"""
        mock_args.add_position_embedding = False
        mock_args.position_embedding_type = 'absolute'
        with pytest.raises(RuntimeError, match="--no-position-embedding is deprecated"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_moe_spec_not_none(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """num_experts not None with spec not None -> AssertionError"""
        mock_args.num_experts = 2
        mock_args.spec = 'some_spec'
        with pytest.raises(AssertionError, match="Model Spec must be None when using MoEs"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_inference_dynamic_batching_no_buffer(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """inference_dynamic_batching without buffer_size_gb -> AssertionError"""
        mock_args.inference_dynamic_batching = True
        mock_args.inference_dynamic_batching_buffer_size_gb = None
        mock_args.inference_dynamic_batching_buffer_guaranteed_fraction = 0.5
        with pytest.raises(AssertionError, match="inference_dynamic_batching_buffer_size_gb must be specified"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_inference_dynamic_batching_no_fraction(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """inference_dynamic_batching without fraction -> AssertionError"""
        mock_args.inference_dynamic_batching = True
        mock_args.inference_dynamic_batching_buffer_size_gb = 1
        mock_args.inference_dynamic_batching_buffer_guaranteed_fraction = None
        with pytest.raises(
            AssertionError, match="inference_dynamic_batching_buffer_guaranteed_fraction must be specified"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_expert_parallel_fp16(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """expert_parallel with fp16 -> AssertionError"""
        mock_args.expert_model_parallel_size = 2
        mock_args.num_experts = 2
        mock_args.fp16 = True
        mock_args.bf16 = False
        mock_args.world_size = 2
        with pytest.raises(AssertionError, match="Expert parallelism is not supported with fp16 training"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.get_device_arch_version', return_value=10)
    def test_tp_gt_1_blackwell_fsdp2_warning(self, mock_dev_arch, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """tp>1, blackwell, use_custom_fsdp -> warning printed, no error (os.environ check skipped)"""
        mock_args.tensor_model_parallel_size = 2
        mock_args.world_size = 2
        mock_args.use_custom_fsdp = True
        mock_args.use_distributed_optimizer = True
        # Need use_dist_ckpt to skip the gloo check at line 984
        mock_args.use_dist_ckpt = True
        mock_args.ckpt_format = 'torch_dist'
        mock_dev_arch.return_value = 10
        # Should pass: use_custom_fsdp=True enters the fsdp branch which only warns, no error
        result = _validate_args_ldt(mock_args)
        assert result is not None

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_custom_fsdp_grad_accum_fusion_with_sharding(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """custom_fsdp, optim_grads_params, gradient_accumulation_fusion -> AssertionError"""
        mock_args.use_custom_fsdp = True
        mock_args.use_distributed_optimizer = True
        mock_args.data_parallel_sharding_strategy = 'optim_grads_params'
        mock_args.gradient_accumulation_fusion = True
        with pytest.raises(
            AssertionError, match="optim_grads_params optim_grads are not supported with gradient accumulation fusion"
        ):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_custom_fsdp_optim_grads_grad_accum_fusion(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """custom_fsdp, optim_grads, gradient_accumulation_fusion -> AssertionError"""
        mock_args.use_custom_fsdp = True
        mock_args.use_distributed_optimizer = True
        mock_args.data_parallel_sharding_strategy = 'optim_grads'
        mock_args.gradient_accumulation_fusion = True
        # data_parallel_sharding_strategy 'optim_grads' first sets overlap_grad_reduce=True
        # Then in use_custom_fsdp check: data_parallel_sharding_strategy in ["optim_grads_params", "optim_grads"] -> True
        # gradient_accumulation_fusion -> AssertionError
        with pytest.raises(AssertionError, match="optim_grads_params optim_grads are not supported"):
            _validate_args_ldt(mock_args)

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_apply_query_key_layer_scaling(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """apply_query_key_layer_scaling=True -> attention_softmax_in_fp32=True"""
        mock_args.apply_query_key_layer_scaling = True
        result = _validate_args_ldt(mock_args)
        assert result.attention_softmax_in_fp32 is True

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_bias_gelu_fusion_disabled(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """add_bias_linear=False -> bias_gelu_fusion=False"""
        mock_args.add_bias_linear = False
        result = _validate_args_ldt(mock_args)
        assert result.bias_gelu_fusion is False

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_add_bias_linear_sync_qkv(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """add_bias_linear=True, add_qkv_bias has default True-> stays True"""
        mock_args.add_bias_linear = True
        mock_args.add_qkv_bias = True
        result = _validate_args_ldt(mock_args)
        assert result.add_qkv_bias is True

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_validation_vp_overlap_p2p_pp_3_ok(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """vp with overlap_p2p_comm, pp=3 -> passes num_layers check, sets vp"""
        mock_args.num_layers_per_virtual_pipeline_stage = 2
        mock_args.overlap_p2p_comm = True
        mock_args.pipeline_model_parallel_size = 3
        mock_args.world_size = 3
        mock_args.num_layers = 12
        # total = 0 + 3 = 3, 3%3=0, dp=1
        # pp=3 > 1, passes overlap check
        # num_layers=12, 12%3=0, 12/3=4, 4%2=0, vp=2
        # overlap_p2p_comm stays True (block not entered)
        result = _validate_args_ldt(mock_args)
        assert result.virtual_pipeline_model_parallel_size == 2

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_validation_vp_num_virtual_stages(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """vp with num_virtual_stages_per_pipeline_rank -> vp set directly"""
        mock_args.num_virtual_stages_per_pipeline_rank = 3
        mock_args.overlap_p2p_comm = True
        # pp > 1 check: pp is set to 1, so fails. But actually MockArgs.pipeline_model_parallel_size=1
        # The check is: if overlap_p2p_comm: if pp <= 1 -> error. So need pp > 1
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        # total = 0 + 2 = 2, 2%2=0, passes
        result = _validate_args_ldt(mock_args)
        # Check vp was set
        assert result is not None

    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._get_vdp_size', return_value=1)
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize._print_args')
    @patch('mindspeed_llm.core.layerwise_disaggregated_training.initialize.is_torch_min_version', return_value=True)
    def test_validation_add_bias_linear_false_warning(self, mock_torch_ver, mock_print, mock_vdp, mock_args):
        """Test non-interleaved with PP > 1 prints warning"""
        mock_args.pipeline_model_parallel_size = 2
        mock_args.world_size = 2
        # Don't set any vp args
        # Line 510-519: vp is None, overlap_p2p_comm set to False, align_param_gather set to False
        result = _validate_args_ldt(mock_args)
        assert result.overlap_p2p_comm is False
        assert result.align_param_gather is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

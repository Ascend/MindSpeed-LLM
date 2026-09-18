# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT."""

# ruff: noqa: E402

import os
import sys
import time
import json
import warnings

_VERSION_018 = os.environ.get("MINDSPEED_LLM_VERSION", "012") == "018"
if _VERSION_018:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))

rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import Callable, Union, Any, List, Optional, Tuple

import torch

# MindSpeed patches must be applied before any Megatron modules are imported..
# isort: off
from mindspeed_llm import megatron_adaptor  # noqa: F401  # pylint: disable=ungrouped-imports
# isort: on

from megatron.training import (
    get_args,
    get_timers,
    print_rank_0,
)
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.parallel_state import (
    get_context_parallel_group,
)
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import (
    StragglerDetector,
)
from megatron.training.utils import (
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt import GPTModel

if _VERSION_018:
    from megatron.training import (
        inprocess_restart,
        set_startup_timestamps,
    )
    from megatron.core.parallel_state import get_hybrid_data_context_parallel_groups
    from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
    from megatron.training.datasets.sft_dataset import SFTDataset
    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
    from megatron.core.transformer.multi_token_prediction import get_mtp_ranks
    from megatron.core.transformer.multi_token_prediction import (
        mtp_on_this_rank as mtp_on_this_rank_func,
    )
    from megatron.core.utils import (
        get_batch_on_this_cp_rank,
        get_batch_on_this_tp_rank,
    )
    from megatron.training.utils import is_first_or_last_pipeline_stage
    from megatron.training.argument_utils import pretrain_cfg_container_from_args
    from megatron.training.arguments import parse_and_validate_args
    from megatron.core.models.hybrid.hybrid_model import HybridModel
else:
    from megatron.training import get_tokenizer
    import megatron.legacy.model
    from megatron.core.datasets.utils import get_blend_from_list
    from megatron.training.utils import (
        get_batch_on_this_cp_rank,
        get_batch_on_this_tp_rank,
    )

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    if _VERSION_018:
        from megatron.post_training.model_builder import modelopt_gpt_hybrid_builder

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from mindspeed_llm.core.models.deepseek4.deepseek4_model import DeepSeek4Model
from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc import get_mhc_spec
from mindspeed_llm.training.training import pretrain
from mindspeed_llm.training.utils import set_mtp_batch_list

_PROGRAM_START_TIME = time.time()
stimer = StragglerDetector()
# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

if _VERSION_018:

    def model_provider(
        model_builder: Callable,
        pre_process=True,
        post_process=True,
        vp_stage: Optional[int] = None,
        config=None,
        pg_collection=None,
    ) -> Union[GPTModel, HybridModel]:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            model_builder: A callable that builds the actual model, its signature is the same as model_provider's with an exception of the first argument which is a builder itself. In addition might take a config passed from outside to skip its own config loading. See gpt_builder or hybrid_builder for an example, see _gpt_model_builder in train_rl.py to see how to augment a default gpt builder and pass the config from outside
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to compute output logits/loss. Defaults to True.

        Returns:
            Union[GPTModel, HybridModel]: The returned model
        """
        args = get_args()

        if args.record_memory_history:
            torch.cuda.memory._record_memory_history(
                True,
                # keep 100,000 alloc/free events from before the snapshot
                trace_alloc_max_entries=100000,
                # record stack information for the trace events
                trace_alloc_record_context=True,
            )

            def oom_observer(device, alloc, device_alloc, device_free):
                # snapshot right after an OOM happened
                print('saving allocated state during OOM')

                filename = f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
                torch.cuda.memory._dump_snapshot(filename)

            torch._C._cuda_attach_out_of_memory_observer(oom_observer)

        if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):
            # [ModelOpt]: Use custom builder + spec when modelopt is enabled
            model_builder = modelopt_gpt_hybrid_builder

        return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)

    def gpt_builder(
        args,
        pre_process=True,
        post_process=True,
        vp_stage=None,
        config=None,
        pg_collection=None,
        use_dualpipe_mtp=False,
    ) -> Union[DeepSeek4Model, GPTModel]:
        """Builds the model.

        If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
            vp_stage (int, optional): Virtual pipeline stage index. Defaults to None.
            config (TransformerConfig, optional): Transformer config object passed from get_model. Defaults to None.
            pg_collection (ProcessGroupCollection, optional): Process groups collection for parallel communication. Defaults to None.

        Returns:
            Union[DeepSeek4Model, megatron.core.models.gpt.gpt_model]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        print_rank_0('building GPT model ...')
        # Experimental loading arguments from yaml
        if config is None:
            if args.yaml_cfg is not None:
                config = core_transformer_config_from_yaml(args, "language_model")
            else:
                config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            if args.mtp_spec is not None:
                mtp_layer_spec = import_module(args.mtp_spec)
            else:
                mtp_layer_spec = transformer_layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, mtp_layer_spec, use_transformer_engine=use_te, vp_stage=vp_stage
            )
            if use_dualpipe_mtp:
                post_process = True

        hc_head_spec = get_mhc_spec(args.enable_mhc)

        model = DeepSeek4Model(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
            hc_head_spec=hc_head_spec,
        )
        return model

    def get_batch(data_iterator, vp_stage: Optional[int] = None):
        """Generate a batch."""

        args = get_args()
        is_hybrid_cp = args.hybrid_context_parallel

        is_middle_stage = not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage())
        pretrain_not_tnd_flags = not args.is_instruction_dataset and not args.reset_attention_mask
        if pretrain_not_tnd_flags and is_middle_stage:
            return (None,) * 5

        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)

        if (
            args.return_document_ids
            and mpu.get_context_parallel_rank() == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
        ):
            print(
                "current idx: {}, current rank: {}, data_parallel_rank: {}, document_ids: {}".format(
                    batch['idx'], torch.distributed.get_rank(), mpu.get_data_parallel_rank(), batch['document_ids']
                )
            )
            batch.pop('document_ids', None)
            batch.pop('idx', None)

        # get batch_list for mtp_block
        if args.mtp_num_layers:
            mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
            set_mtp_batch_list(mtp_batch_list)

        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(
            batch,
            is_hybrid_cp=is_hybrid_cp,
            cp_group=get_context_parallel_group(),
            hybrid_cp_group_func=get_hybrid_data_context_parallel_groups,
        )
        return batch.values()

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None):
        """Loss function.

        Args:
            loss_mask (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses

        Returns:
            the loss scalar for this micro-batch
            the number of non-padded tokens in this microbatch
            a dict containing reporting metrics on the loss and number of tokens across
                the data parallel ranks
        """
        args = get_args()
        if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
            loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
        else:
            losses = output_tensor.view(-1).float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses * loss_mask)

            num_tokens = loss_mask.sum().clone().detach().to(torch.int)
            report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

        # Check individual rank losses are not NaN prior to DP all-reduce.
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss,
                rejection_func=torch.isnan,
                message="found NaN in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss,
                rejection_func=torch.isinf,
                message="found Inf in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
        # Check for spiky loss
        if args.check_for_spiky_loss:
            rerun_state_machine.validate_result(
                result=loss,
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context="loss",
                ),
                message="Spiky loss",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=False,
            )

        return loss, num_tokens, report

    def forward_step(data_iterator, model: DeepSeek4Model, return_schedule_plan: bool = False):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (DeepSeek4Model): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        global stimer
        with stimer(bdata=True):
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

        packed_seq_params = None
        timers('batch-generator').stop()

        with stimer:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, (
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                )
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=labels,
                    loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params,
                )

        # [ModelOpt]: model is needed to access ModelOpt distillation losses
        return output_tensor, partial(loss_func, loss_mask, model=model)

    def is_dataset_built_on_rank(vp_stage=None, is_packed_sequence=False):
        args = get_args()
        config = core_transformer_config_from_args(args)
        if mpu.get_tensor_model_parallel_rank() != 0:
            return False
        elif is_packed_sequence:
            return True
        return is_first_or_last_pipeline_stage(vp_stage) or mtp_on_this_rank_func(
            layout=config.pipeline_model_parallel_layout,
            mtp_num_layers=config.mtp_num_layers,
            ignore_virtual=False,
            vp_stage=vp_stage,
        )

    def core_gpt_dataset_config_from_args(args: Any) -> GPTDatasetConfig:
        tokenizer = build_tokenizer(args)

        # Sometimes --data-path is too long, instead we parse it from a file.
        blend: Optional[Tuple[List[str], Optional[List[float]]]]
        blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
        blend, blend_per_split = get_blend_and_blend_per_split(args)

        sequences_per_dataset = None
        if args.per_dataset_sequences_path is not None:
            with open(args.per_dataset_sequences_path, "r") as f:  # pylint: disable=unspecified-encoding
                sequences_per_dataset = json.load(f)

        data_args = {
            "random_seed": args.seed,
            "sequence_length": args.seq_length,
            "blend": blend,
            "blend_per_split": blend_per_split,
            "split": args.split,
            "multiple_validation_sets": args.multiple_validation_sets,
            "full_validation": args.full_validation,
            "num_dataset_builder_threads": args.num_dataset_builder_threads,
            "path_to_cache": args.data_cache_path,
            "mmap_bin_files": args.mmap_bin_files,
            "tokenizer": tokenizer,
            "reset_position_ids": args.reset_position_ids,
            "reset_attention_mask": args.reset_attention_mask,
            "eod_mask_loss": args.eod_mask_loss,
            "create_attention_mask": args.create_attention_mask_in_dataloader,
            "object_storage_cache_path": args.object_storage_cache_path,
            "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
            "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
            "fast_cache_load": args.dataloader_fast_cache_load,
            "sequences_per_dataset": sequences_per_dataset,
            "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
            "context_parallel_size": args.context_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "sequence_parallel_size": args.tensor_model_parallel_size * args.sequence_parallel,
            "hybrid_context_parallel": args.hybrid_context_parallel,
        }

        # add FIM args to the config
        if args.fim_data:
            extra_tokens = {
                "prefix": args.fim_prefix_token,
                "middle": args.fim_middle_token,
                "suffix": args.fim_suffix_token,
                "pad": args.fim_pad_token,
                "eod": args.fim_eod_token,
            }
            data_args.update(
                {
                    "fim_rate": args.fim_rate,
                    "fim_spm_rate": args.fim_spm_rate,
                    "fim_extra_tokens": extra_tokens,
                    "fim_split_sample": args.fim_split_sample,
                    "fim_fragment_rate": args.fim_fragment_rate,
                    "fim_no_prefix": args.fim_no_prefix,
                }
            )
            return GPTFIMDatasetConfig(**data_args)

        return GPTDatasetConfig(**data_args)

    def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
        """Build the train test and validation datasets.

        Args:
            train_val_test_num_samples : A list containing the number of samples in train test and validation.
        """
        args = get_args()

        config = core_gpt_dataset_config_from_args(args)

        is_packed_sequence = False
        if args.sft:
            dataset_type = SFTDataset
            is_packed_sequence = True  # SFT always uses packed sequence
        else:
            if args.mock_data:
                dataset_type = MockGPTDataset
            elif args.fim_data:
                dataset_type = GPTFIMDataset
            else:
                dataset_type = GPTDataset

        print_rank_0("> building train, validation, and test datasets for GPT ...")

        is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage, is_packed_sequence=is_packed_sequence)
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, is_dataset_built, config
        ).build()

        print_rank_0("> finished creating GPT datasets ...")

        return train_ds, valid_ds, test_ds

    def get_embedding_ranks(pp_ranks: List[int]):
        """Get the embedding ranks."""
        embedding_ranks = [pp_ranks[0]]
        if len(pp_ranks) > 1:
            args = get_args()
            if not args.untie_embeddings_and_output_weights:
                embedding_ranks.append(pp_ranks[-1])
            config = core_transformer_config_from_args(args)
            mtp_ranks = get_mtp_ranks(pp_ranks, config)
            embedding_ranks.extend(mtp_ranks)
        embedding_ranks = list(set(embedding_ranks))
        embedding_ranks = sorted(embedding_ranks)
        return embedding_ranks

    def main():
        # Timestamp right after entering __main__ block (after all imports/library setup)
        _MAIN_ENTRY_TIME = time.time()

        # Register startup timestamps for timing report in pretrain()
        set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

        # Temporary for transition to core datasets
        setattr(train_valid_test_datasets_provider, "is_distributed", True)

        wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

        args = parse_and_validate_args(
            extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        )
        full_config = pretrain_cfg_container_from_args(args)
        wrapped_pretrain(
            full_config,
            train_valid_test_datasets_provider,
            partial(model_provider, gpt_builder),
            ModelType.encoder_or_decoder,
            forward_step,
            store=store,
            get_embedding_ranks=get_embedding_ranks,
        )
else:

    def model_provider(
        pre_process=True, post_process=True, use_dualpipe_mtp=False
    ) -> Union[DeepSeek4Model, megatron.legacy.model.GPTModel]:
        """Builds the model.

        If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[DeepSeek4Model, megatron.legacy.model.DeepSeek4Model]: The returned model
        """
        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        print_rank_0('building GPT model ...')
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if not args.use_legacy_models:
            if args.spec is not None:
                transformer_layer_spec = import_module(args.spec)
            else:
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
            mtp_block_spec = None
            if args.mtp_num_layers is not None:
                if args.mtp_spec is not None:
                    mtp_layer_spec = import_module(args.mtp_spec)
                else:
                    mtp_layer_spec = transformer_layer_spec
                mtp_block_spec = get_gpt_mtp_block_spec(config, mtp_layer_spec, use_transformer_engine=use_te)
                if use_dualpipe_mtp:
                    post_process = True

            hc_head_spec = get_mhc_spec(args.enable_mhc)

            model = DeepSeek4Model(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                mtp_block_spec=mtp_block_spec,
                hc_head_spec=hc_head_spec,
            )
        else:
            raise ValueError("DeepSeek4 model is only supported with Megatron Core!")

        return model

    def get_batch(data_iterator):
        """Generate a batch."""

        args = get_args()

        is_middle_stage = not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage())
        pretrain_not_tnd_flags = not args.is_instruction_dataset and not args.reset_attention_mask
        if pretrain_not_tnd_flags and is_middle_stage:
            return (None,) * 5

        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)

        if (
            args.return_document_ids
            and mpu.get_context_parallel_rank() == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
        ):
            print(
                "current idx: {}, current rank: {}, data_parallel_rank: {}, document_ids: {}".format(
                    batch['idx'], torch.distributed.get_rank(), mpu.get_data_parallel_rank(), batch['document_ids']
                )
            )
            batch.pop('document_ids', None)
            batch.pop('idx', None)

        # get batch_list for mtp_block
        if args.mtp_num_layers:
            mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
            set_mtp_batch_list(mtp_batch_list)

        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            loss_mask (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses

        Returns:
            the loss scalar for this micro-batch
            the number of non-padded tokens in this microbatch
            a dict containing reporting metrics on the loss and number of tokens across
                the data parallel ranks
        """
        args = get_args()

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

        # Check individual rank losses are not NaN prior to DP all-reduce.
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message="found NaN in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message="found Inf in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
        # Check for spiky loss
        if args.check_for_spiky_loss:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context="loss",
                ),
                message="Spiky loss",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=False,
            )
        # Reduce loss for logging.
        reporting_loss = loss.clone().detach()
        try:
            if args.enable_elastic_training:
                from mindspeed_llm.core.high_availability import elastic_training_common

                if not elastic_training_common.zit_scale_in_running_state():
                    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
            else:
                torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
        except Exception:
            torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
        # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
        # on loss[0] fixes this
        local_num_tokens = loss[1].clone().detach().to(torch.int)
        return (
            loss[0].clone(),
            local_num_tokens,
            {'lm loss': (reporting_loss[0], reporting_loss[1])},
        )

    def forward_step(data_iterator, model: DeepSeek4Model):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (DeepSeek4Model): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
        timers('batch-generator').stop()

        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)

        return output_tensor, partial(loss_func, loss_mask)

    def is_dataset_built_on_rank():
        return mpu.get_tensor_model_parallel_rank() == 0

    def core_gpt_dataset_config_from_args(args):
        tokenizer = get_tokenizer()

        return GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=get_blend_from_list(args.data_path),
            blend_per_split=[
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path),
            ],
            split=args.split,
            path_to_cache=args.data_cache_path,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=tokenizer,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )

    def train_valid_test_datasets_provider(train_val_test_num_samples):
        """Build the train test and validation datasets.

        Args:
            train_val_test_num_samples : A list containing the number of samples in train test and validation.
        """
        args = get_args()

        config = core_gpt_dataset_config_from_args(args)

        if config.mock:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset
        print_rank_0("> building train, validation, and test datasets for GPT ...")

        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
        ).build()

        print_rank_0("> finished creating GPT datasets ...")

        return train_ds, valid_ds, test_ds

    def main():
        # Temporary for transition to core datasets
        train_valid_test_datasets_provider.is_distributed = True
        pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step)


if __name__ == "__main__":
    main()

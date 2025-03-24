from typing import Union

import torch

from megatron.core import mpu, dist_checkpointing
from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.async_utils import schedule_async_save
from megatron.training.checkpointing import get_rng_state, get_checkpoint_name, \
    get_distributed_optimizer_checkpoint_name, ensure_directory_exists, \
    generate_state_dict, get_checkpoint_tracker_filename
from megatron.training.global_vars import get_args
from megatron.training.global_vars import get_timers
from megatron.training.training import compute_throughputs_and_append_to_progress_log
from megatron.training.utils import unwrap_model, print_rank_0, append_to_progress_log
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from mindspeed_llm.tasks.posttrain.orm.orm_model import GPTRewardModel


def model_provider(is_reward_model=False, pre_process=True, post_process=True) -> Union[GPTModel]:
    """Builds the model.

    Currently supports only the mcore GPT model.

    Args:
        is_reward_model (bool, optional): Set to true if you build GPTReward model
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
        Defaults to True.

    Returns:
        Union[GPTModel, GPTRewardModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if is_reward_model:
        print_rank_0('building GPTReward model ...')
    else:
        print_rank_0('building GPT model ...')

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if not args.use_mcore_models:
        raise ValueError("Training models currently supports mcore only. Please set use_mcore_models to True.")
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                args.moe_grouped_gemm)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

    if is_reward_model:
        model = GPTRewardModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            post_layer_norm=not args.no_post_layer_norm,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True if args.sequence_parallel else False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
    else:
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True if args.sequence_parallel else False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )
    return model


def pad_to_tensor_dict(data, padding_side="right", pad_multi_of=16):
    max_length = torch.LongTensor([max(len(val) for val in data)]).cuda()
    max_length = max_length if max_length % pad_multi_of == 0 else (max_length // pad_multi_of + 1) * pad_multi_of
    torch.distributed.all_reduce(max_length, op=torch.distributed.ReduceOp.MAX)

    args = get_args()

    pad_id = args.pad_token_id if args.pad_token_id else args.eos_token_id

    context_lengths = [len(val) for val in data]
    ori_context_lengths = []
    for val in data:
        if pad_id not in val:
            ori_context_lengths.append(len(val))
        else:
            ori_context_lengths.append(torch.nonzero(torch.tensor(val) == pad_id).min().item() + 1)

    data_length = len(data)
    for i in range(data_length):
        if context_lengths[i] < max_length:
            if padding_side == "right":
                data[i].extend([pad_id] * (max_length - context_lengths[i]))
            else:
                data[i] = [pad_id] * (max_length - context_lengths[i]) + data[i]
    return ori_context_lengths, max_length


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                    num_floating_point_operations_so_far, checkpointing_context=None, save_model_type=None):
    """Save a model checkpoint.

    Checkpointing context is used to persist some checkpointing state
    throughout a single job. Must be initialized externally (not used if None).
    """
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    save_path = args.save
    # save_model_type is 'actor' or 'critic'
    if save_model_type:
        save_path = args.save + '/' + save_model_type

    ckpt_format = args.dist_ckpt_format if args.use_dist_ckpt else 'torch'
    print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
        iteration, save_path, ckpt_format))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(args.use_dist_ckpt)

    # Checkpoint name.
    print(f"save_path {save_path}, iteration {iteration}, args.use_dist_ckpt {args.use_dist_ckpt}")

    checkpoint_name = get_checkpoint_name(save_path, iteration, return_base_dir=args.use_dist_ckpt)

    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None and not args.use_dist_ckpt:
        optim_checkpoint_name = \
            get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        optimizer.save_parameter_state(optim_checkpoint_name)

    async_save_request = None
    if args.async_save:
        if not args.use_dist_ckpt:
            raise NotImplementedError('Async checkpoint save not implemented for legacy checkpoints')
        elif args.dist_ckpt_format != 'torch_dist':
            raise NotImplementedError(f'Async checkpoint save not implemented for {args.dist_ckpt_format} distributed checkpoint format')

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or mpu.get_data_modulo_expert_parallel_rank() == 0 \
            or args.use_dist_ckpt:

        optim_sd_kwargs = {}
        if args.use_dist_ckpt and args.use_distributed_optimizer:
            optim_sd_kwargs['sharding_type'] = ('fully_sharded_bucket_space'
                                                if args.ckpt_fully_parallel_save
                                                else 'dp_zero_gather_scatter')
            print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
        state_dict = generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                         args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

        state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
        if args.use_dist_ckpt:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                ensure_directory_exists(checkpoint_name, check_parent=False)
            validate_sharding_integrity = True
            save_strategy = (checkpointing_context or {}).get('save_strategy',
                                                              get_default_save_sharded_strategy(args.dist_ckpt_format))
            if args.ckpt_fully_parallel_save:
                if checkpointing_context is not None and 'save_strategy' in checkpointing_context:
                    # Already saved once before - don't need to rerun sharding validation
                    validate_sharding_integrity = not args.ckpt_assume_constant_structure
                else:
                    save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, mpu.get_data_parallel_group(with_context_parallel=True),
                                                                     args.ckpt_assume_constant_structure)
            # Store save strategy for future checkpoint saves
            if checkpointing_context is not None:
                checkpointing_context['save_strategy'] = save_strategy
            async_save_request = dist_checkpointing.save(state_dict, checkpoint_name, save_strategy,
                                                         async_sharded_save=args.async_save)
        else:
            # Save.
            ensure_directory_exists(checkpoint_name)
            torch.save(state_dict, checkpoint_name)

    if not args.async_save:
        if async_save_request is not None:
            raise ValueError("Async save request should not be None when async_save is False.")
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # And update the latest iteration
    if not torch.distributed.is_initialized() \
            or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(save_path)

        def iter_finalize_fn():
            with open(tracker_filename, 'w') as f:
                f.write(str(iteration))
            print_rank_0('  successfully saved checkpoint from iteration {:7d} to {}'
                         .format(iteration, save_path))
            if args.log_progress and args.async_save:
                append_to_progress_log(f'Saved async checkpoint\tIteration: {iteration}',
                                       barrier=False)

        if args.async_save:
            if async_save_request is None:
                raise ValueError("Async save request must not be None when async_save is True.")
            async_save_request.add_finalize_fn(iter_finalize_fn)
        else:
            iter_finalize_fn()

    if args.async_save:
        schedule_async_save(async_save_request)
        print_rank_0('  scheduled an async checkpoint save at iteration {:7d} to {}' \
                     .format(iteration, save_path))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler,
                             num_floating_point_operations_so_far, checkpointing_context, save_model_type=None):
    args = get_args()
    timers = get_timers()
    # Extra barrier is added to make sure all ranks report the max time.
    timers('save-checkpoint', log_level=0).start(barrier=True)
    print(f"save_model_type {save_model_type}")
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                    num_floating_point_operations_so_far, checkpointing_context, save_model_type)
    timers('save-checkpoint').stop(barrier=True)
    timers.log(['save-checkpoint'])

    if args.log_progress:
        compute_throughputs_and_append_to_progress_log(iteration,
                                                       num_floating_point_operations_so_far)


class MiniBatchIterator:
    def __init__(self, data, num_micro_batch, micro_batch_size):
        self.data = data
        self.index = 0
        self.num_micro_batch = num_micro_batch
        self.micro_batch_size = micro_batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_micro_batch

    def __next__(self):
        if self.index >= self.num_micro_batch:
            raise StopIteration

        mini_batch = {}
        for k, v in self.data.items():
            if torch.is_tensor(v):
                mini_batch[k] = v[self.index * self.micro_batch_size: (self.index + 1) * self.micro_batch_size]
            else:
                mini_batch[k] = v

        self.index += 1

        return mini_batch

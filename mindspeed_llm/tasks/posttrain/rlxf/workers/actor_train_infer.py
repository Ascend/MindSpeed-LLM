# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import copy
import os
from functools import partial
from typing import Iterable, Dict

import ray
import torch
import torch_npu
import torch.nn.functional as F
from tensordict import TensorDict

from megatron.training import get_args, initialize_megatron, get_timers, get_tokenizer
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core import parallel_state as mpu, tensor_parallel
from megatron.training.training import append_to_progress_log, build_train_valid_test_data_iterators, print_datetime
from megatron.training import get_model
from megatron.training.utils import unwrap_model
from megatron.training.checkpointing import save_checkpoint
from megatron.training.training import num_floating_point_operations
from mindspeed_llm.tasks.posttrain.rlxf.training.core_algos import compute_policy_loss, find_first_eos_index, compute_grpo_policy_loss
from mindspeed_llm.tasks.posttrain.rlxf.utils.torch_functional import split_dict_tensor_into_batches
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker import MegatronWorker
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.decorator import register, Dispatch, Execute
from mindspeed_llm.tasks.posttrain.rlxf.training.parameter_mapping import sync_param_nums, run_auto_mapping, \
    init_parameter_mapping_distributed
from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto, make_batch_generator
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
import mindspeed_llm.tasks.posttrain.rlxf.training.parallel_state as ps
from mindspeed_llm.tasks.inference.module import MegatronModuleForCausalLM
from mindspeed_llm.tasks.preprocess.blended_mtf_dataset import build_blended_mtf_dataset
from mindspeed_llm.training.initialize import set_jit_fusion_options
from mindspeed_llm.training.utils import get_finetune_data_on_this_tp_rank, get_tune_attention_mask
from mindspeed_llm.tasks.posttrain.utils import compute_log_probs, append_to_dict


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    print("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = build_blended_mtf_dataset(
        data_prefix=args.data_path,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.max_prompt_length,
        seed=args.seed)

    print("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


@ray.remote
class PPOActorWorker(MegatronWorker):
    """
    A basic class to launch two megatron instances with different communication groups.
    Currently assume that the first group is for training and the second group is for inference.
    """

    def __init__(self, config, role):
        super().__init__()
        self.config = config
        self.role = role
        self.IGNORE_INDEX = -100
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

        initialize_megatron(args_defaults={'no_load_rng': True, 'no_load_optim': True},
                            role=self.role,
                            config=self.config,
                            two_megatron=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self):
        init_parameter_mapping_distributed()
        self.is_inference_node = ps.in_mg2_inference_group()
        if self.is_inference_node:
            self.node = PPOActorInferWorker()
        else:
            self.node = PPOActorTrainWorker()
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def auto_mapping(self):
        if self.is_inference_node:
            run_auto_mapping(self.node.inf_model)
        else:
            run_auto_mapping(self.node.actor.model[0])
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_ALL_GATHER_INFER, execute_mode=Execute.INFER)
    def generate_sequences(self):
        args = get_args()
        output = self.node.run_inference()
        tokenizer = get_tokenizer()
        meta_info = {'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id, 'num_samples_per_step':args.num_samples_per_step}
        output.meta_info.update(meta_info)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_ALL_GATHER_TRAIN, execute_mode=Execute.TRAIN)
    def update_actor(self, data):
        device = next(self.node.actor.model[0].parameters()).device
        data = data.to(device)

        dataloader = self.node.actor.make_minibatch_iterator(data=data)

        metrics = self.node.actor.update_policy(dataloader=dataloader)

        output = DataProto(meta_info={'metrics': metrics})
        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_ALL_GATHER_TRAIN, execute_mode=Execute.TRAIN)
    def get_log_probs(self, data):
        old_log_probs = self.node.actor.compute_log_prob(data)
        if old_log_probs is not None:  # pp last stage
            data.batch['old_log_probs'] = old_log_probs
            data = data.to('cpu')
        else:  # pp intermediate stage, no useful results
            data = None
        # clear kv cache
        torch.cuda.empty_cache()
        return data

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.TRAIN)
    def save_checkpoint(self, iteration):
        self.node.actor.save_checkpoint(iteration)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.TRAIN)
    def get_iteration(self):
        return self.node.actor.get_iteration()


class PPOActorTrainWorker(BaseTrainer):
    def __init__(self):
        super().__init__()

    def initialize(self):
        self.args = get_args()
        model, optimizer, opt_param_scheduler = self._build_model_and_optimizer()
        self.actor = MegatronPPOActor(model=model, optimizer=optimizer, opt_param_scheduler=opt_param_scheduler)
        sync_param_nums(model[0])
        if self.args.stage == "ray_online_dpo":
            self.args.micro_batch_size *= 2
            self.args.ppo_mini_batch_size *= 2

    def _build_model_and_optimizer(self):
        from megatron.training.training import setup_model_and_optimizer
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            self.model_provider, self.model_type)
        return model, optimizer, opt_param_scheduler

    def get_batch(self, data_iterator):
        """
        Retrieves a batch of data from the data iterator.
        Called during each forward step.
        """
        pass

    def loss_func(self, input_tensor, output_tensor):
        """
        Computes the loss function.
        Called during each forward step.
        """
        pass

    def forward_step(self, data_iterator, model):
        """
        Performs a forward pass and computes the loss.
        Called during each training iteration.
        """
        pass


def pad_to_tensor_dict(data, padding_side="right", pad_multi_of=16):
    max_length = torch.LongTensor([max(len(val) for val in data)]).cuda()
    max_length = max_length if max_length % pad_multi_of == 0 else (max_length // pad_multi_of + 1) * pad_multi_of
    torch.distributed.all_reduce(max_length, op=torch.distributed.ReduceOp.MAX)

    tokenizer = get_tokenizer()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    context_lengths = [len(val) for val in data]

    data_length = len(data)
    for i in range(data_length):
        if context_lengths[i] < max_length:
            if padding_side == "right":
                data[i].extend([pad_id] * (max_length - context_lengths[i]))
            else:
                data[i] = [pad_id] * (max_length - context_lengths[i]) + data[i]
    return context_lengths, max_length


class PPOActorInferWorker(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.keys = None

    def model_provider(self, pre_process=True, post_process=True):
        """Builds the inference model.

        If you set the use_mcore_models to True, it will return the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModelInfer, GPTModel]: The returned model
        """

        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
            get_gpt_layer_local_spec
        from megatron.core.transformer.spec_utils import import_module
        from megatron.training import get_args, print_rank_0
        from megatron.training.arguments import core_transformer_config_from_args
        from megatron.training.yaml_arguments import core_transformer_config_from_yaml

        from mindspeed_llm.tasks.inference.module import GPTModelInfer

        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        print_rank_0('building GPT Rollout model ...')
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                    args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModelInfer(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )

        return model

    def initialize(self):
        train_valid_test_datasets_provider.is_distributed = True
        self.args = get_args()
        self.timers = get_timers()
        self.train_valid_test_datasets_provider = train_valid_test_datasets_provider

        if self.args.log_progress:
            append_to_progress_log("Starting job")
        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        self.timers('train/valid/test-data-iterators-setup', log_level=0).start(
            barrier=True)

        self.args.num_layer_list = None
        self.args.micro_batch_size = 1
        self.args.sequence_parallel = False

        self.args.model = unwrap_model(get_model(self.model_provider, wrap_with_ddp=False))
        self.inf_model = self.args.model[0]
        self.args.dataset_additional_keys = eval(self.args.dataset_additional_keys[0]) if self.args.dataset_additional_keys else []

        sync_param_nums(self.inf_model)
        true_pad_to_multiple_of = self.args.pad_to_multiple_of
        self.args.pad_to_multiple_of = 1 # we don't want to pad data here
        self.train_data_iterator, self.valid_data_iterator, self.test_data_iterator \
            = build_train_valid_test_data_iterators(
            self.train_valid_test_datasets_provider)
        self.args.pad_to_multiple_of = true_pad_to_multiple_of
        self.timers('train/valid/test-data-iterators-setup').stop()
        print_datetime('after dataloaders are built')

        # Print setup timing.
        print('done with setup ...')
        self.timers.log(['model-setup', 'train/valid/test-data-iterators-setup'], barrier=True)

    def get_batch(self, data_iterator):
        """Generate a batch identical to Llama factory"""
        args = get_args()

        self.keys = ['input_ids', *self.args.dataset_additional_keys]

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, _ = get_finetune_data_on_this_tp_rank(data_iterator)
            else:
                tokens = None
            return tokens

        # Items and their type.

        data_type = torch.int64
        cur_data = next(data_iterator)

        # add problem category for reward choosing
        if args.dataset_category is not None:
            dataset_category = [int(item) for item in args.dataset_category.split(",")]
            categories = [dataset_category[id.item()] for id in cur_data['dataset_id']]
            cur_data['categories'] = torch.tensor(categories, dtype=torch.int64)

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(self.keys, cur_data, data_type)

        # Unpack
        batch = {}
        for key in self.keys:
            batch[key] = data_b.get(key).long()

        return batch

    def run_inference(self):
        args = get_args()
        num_infer_steps = args.global_batch_size // (args.data_parallel_size * args.num_samples_per_step)
        responses = []
        idx_list = []
        idx_list_per_step = []
        additional_dict = {}
        additional_dict_per_step = {}

        for k in self.args.dataset_additional_keys:
            if not hasattr(additional_dict, k):
                additional_dict[k] = []

        max_new_tokens = args.seq_length - args.max_prompt_length
        if max_new_tokens % args.pad_to_multiple_of != 0:
            raise ValueError(f"Please adjust pad_to_multiple_of so that max_new_tokens % args.pad_to_multiple_of == 0. "
                            f"Current max_new_tokens: {max_new_tokens}, pad_to_multiple_of: {args.pad_to_multiple_of}")
        for _ in range(num_infer_steps):
            for k in self.args.dataset_additional_keys:
                if not hasattr(additional_dict_per_step, k):
                    additional_dict_per_step[k] = []

            for _ in range(args.num_samples_per_step):
                batch = self.get_batch(self.train_data_iterator)

                tokens = batch["input_ids"]
                tokens_list = tokens.view(-1).cpu().numpy().tolist()

                for additional_key in self.args.dataset_additional_keys:
                    additional_val = batch.get(additional_key).view(-1).cpu().numpy().tolist()

                    for _ in range(args.n_samples_per_prompt):
                        additional_dict_per_step.get(additional_key).append(copy.deepcopy(additional_val))

                for _ in range(args.n_samples_per_prompt):
                    idx_list_per_step.append(copy.deepcopy(tokens_list))

                if args.stage == "ray_online_dpo":
                    idx_list_per_step.append(copy.deepcopy(tokens_list))

            responses_per_step = self.inf_model.generate(
                copy.deepcopy(idx_list_per_step),
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                detokenize=False,
                broadcast=False,
                truncate=True
            )
            
            if not isinstance(responses_per_step, list):
                responses_per_step = [responses_per_step]

            responses.extend(responses_per_step)
            idx_list.extend(idx_list_per_step)
            idx_list_per_step = []

            for k in additional_dict:
                additional_dict[k].extend(additional_dict_per_step[k])

            additional_dict_per_step = {}


        responses_ori_length, responses_pad_length = pad_to_tensor_dict(
            responses,
            pad_multi_of=args.pad_to_multiple_of
        )
        prompts_ori_length, prompts_pad_length = pad_to_tensor_dict(
            idx_list, "left",
            pad_multi_of=args.pad_to_multiple_of
        )

        for additional_key in self.args.dataset_additional_keys:
            tmp_val = additional_dict.get(additional_key)
            pad_to_tensor_dict(
                tmp_val,
                pad_multi_of=args.pad_to_multiple_of
            )
            additional_dict[additional_key] = tmp_val

        input_ids = [prompt + response for prompt, response in zip(idx_list, responses)]

        attention_mask = generate_attention_mask(input_ids, prompts_ori_length, prompts_pad_length,
                                                 responses_ori_length, responses_pad_length)

        position_ids = generate_position_ids_from_attention_mask(input_ids, prompts_ori_length, prompts_pad_length)
        if self.args.stage == "ray_online_dpo":
            batch_size = args.global_batch_size // args.data_parallel_size * 2
        else:
            batch_size = args.global_batch_size // args.data_parallel_size * args.n_samples_per_prompt

        batch = TensorDict(
            dict(
                    {
                        "prompts": idx_list,
                        "responses": responses,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "responses_ori_length": responses_ori_length
                    }, **additional_dict
                ),
            batch_size=batch_size
        )

        return DataProto(batch=batch)

    def loss_func(self, input_tensor, output_tensor):
        """
        Computes the loss function.
        Called during each forward step.
        """
        pass

    def forward_step(self, data_iterator, model):
        """
        Performs a forward pass and computes the loss.
        Called during each training iteration.
        """
        pass


def generate_position_ids_from_attention_mask(input_ids_list, prompts_ori_length, prompts_pad_length):
    """
    生成与 attention_mask 对应的 position_ids 列表。

    参数:
    input_ids_list (list of lists): 包含 input_ids 的列表，每个元素是一个列表。
    prompts_ori_length (list of lists): 包含 prompt_ori_length 的列表，每个元素是int。
    prompts_pad_length int: prompts_pad_length，int。

    返回:
    list of lists: 包含 position_ids 的列表，每个元素是一个列表。
    """
    position_ids_list = []
    for idx, input_ids in enumerate(input_ids_list):
        prompt_pad_length = prompts_pad_length - prompts_ori_length[idx]
        position_ids = [0] * prompt_pad_length + list(range(len(input_ids) - prompt_pad_length))
        position_ids_list.append(position_ids)

    return position_ids_list


def generate_attention_mask(input_ids_list, prompts_ori_length, prompts_pad_length, responses_ori_length,
                            responses_pad_length):
    """
    生成与 input_ids 对应的 attention_mask 列表。

    参数:
    input_ids_list (list of lists): 包含 input_ids 的列表，每个元素是一个列表。
    prompts_ori_length (list of lists): 包含 prompt_ori_length 的列表，每个元素是int。
    prompts_pad_length int: prompts_pad_length，int。
    responses_ori_length (list of lists): 包含 response_ori_length 的列表，每个元素是int。
    responses_pad_length int: responses_pad_length，int。

    返回:
    list of lists: 包含 attention_mask 的列表，每个元素是一个列表。
    """
    attention_mask_list = []

    for idx, input_ids in enumerate(input_ids_list):
        attention_mask = torch.ones_like(torch.tensor(input_ids))
        prompt_pad_length = prompts_pad_length - prompts_ori_length[idx]
        response_pad_length = responses_pad_length - responses_ori_length[idx]
        attention_mask[:prompt_pad_length] = 0
        if response_pad_length > 0:
            attention_mask[-response_pad_length:] = 0
        attention_mask_list.append(attention_mask.numpy().tolist())

    return attention_mask_list


def split_two_prompts(origin_tensor):
    origin_tensor = origin_tensor.reshape(-1, 2)
    first_half, second_half = origin_tensor.split(1, dim=1)
    return first_half.reshape(-1), second_half.reshape(-1)



class MegatronPPOActor():

    def __init__(self, model, optimizer, opt_param_scheduler):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            megatron_config (OmegaConf): megatron configuration. It must contains

                ``sequence_parallel_enabled``: whether the sequence parallel is enabled.

                ``param_dtype``: the dtype of the parameters.

                ``virtual_pipeline_model_parallel_size``: virtual pipeline model parallel size. a.k.a number of chunks in each pp stage.
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this pp stage.
                each nn.Module in this rank holds a vpp module chunk.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation. Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron. It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        """
        self.args = get_args()
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler
        self._beta = self.args.dpo_beta
        self.num_floating_point_operations_so_far = 0

    def get_iteration(self):
        return self.args.iteration

    def save_checkpoint(self, iteration):

        save_checkpoint(iteration, self.model, self.optimizer, self.opt_param_scheduler,
                        self.num_floating_point_operations_so_far)

    def compute_log_prob(self, data) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        data.batch = data.batch.contiguous()

        def compute_logprobs_fn(output, data):
            response = data['responses']
            response_length = response.size(1)
            logits = output
            logits = logits[:, -response_length - 1:-1]
            _, _, log_probs = compute_log_probs(logits, response, per_token=True)
            return {'log_probs': log_probs}

        # We make recompute_old_log_prob by default here.
        data = data.to(next(self.model[0].parameters()).device)
        with torch.no_grad():
            output = self.forward_backward_batch(data, forward_only=True, post_process_fn=compute_logprobs_fn)
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                log_probs = torch.cat([single_output['log_probs'] for single_output in output], dim=0)  # (bs, seq_size)
                log_probs = log_probs.to(torch.float32)
            else:
                log_probs = None

        # add empty cache after each compute
        torch.cuda.empty_cache()

        return log_probs

    @property
    def beta(self):
        if isinstance(self._beta, list):
            epoch = self.state.epoch
            return self._beta[epoch] if epoch < len(self._beta) else self._beta[-1]
        else:
            return self._beta

    def make_minibatch_iterator(self, data):
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of responses.
                See PPO paper for details.

        Returns:

        """
        return data.make_iterator(mini_batch_size=self.args.ppo_mini_batch_size,
                                  epochs=self.args.ppo_epochs,
                                  dataloader_kwargs={'shuffle': self.args.shuffle_minibatch})

    def forward_backward_batch(self, data, forward_only=False, post_process_fn=None):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks

        data.batch['attention_mask'] = data.batch['attention_mask'].to(bool)

        batch_size = self.args.micro_batch_size
        batches = split_dict_tensor_into_batches(data.batch, batch_size=batch_size)

        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        forward_backward_func = get_forward_backward_func()

        def loss_func_ppo(output, data, meta_info):
            """
            This loss_func has two modes
            1. when forward_only is True: use post_process_fn to calculate the log_probs
            2. when forward_only is False: calculate the policy loss
            """
            if forward_only:
                if post_process_fn is None:
                    return 1.0, {'logits': output}
                else:
                    return 1.0, post_process_fn(output, data)

            responses = data['responses']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            old_log_prob = data['old_log_probs']
            advantages = data['advantages']

            clip_ratio = meta_info['clip_ratio']

            # compute policy loss
            logits = output
            logits = logits[:, -response_length - 1:-1]
            _, _, log_prob = compute_log_probs(logits, responses, per_token=True)
            pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss(old_log_prob=old_log_prob,
                                                                          log_prob=log_prob,
                                                                          advantages=advantages,
                                                                          eos_mask=response_mask,
                                                                          cliprange=clip_ratio)
            policy_loss = pg_loss
            # return loss and stats
            stats = {
                'actor/pg_loss': abs(pg_loss.detach().item()),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item()
            }
            return policy_loss, stats

        def loss_func_grpo(output, data, meta_info):
            """
            This loss_func has two modes
            1. when forward_only is True: use post_process_fn to calculate the log_probs
            2. when forward_only is False: calculate the policy loss
            """
            if forward_only:
                if post_process_fn is None:
                    return 1.0, {'logits': output}
                else:
                    return 1.0, post_process_fn(output, data)

            responses = data['responses']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            old_log_prob = data['old_log_probs']
            advantages = data['advantages']
            ref_log_prob = data['ref_log_prob']
            clip_ratio = meta_info['clip_ratio']

            # compute policy loss
            logits = output
            logits = logits[:, -response_length - 1:-1]
            _, _, log_prob = compute_log_probs(logits, responses, per_token=True)

            pg_loss, pg_clipfrac, ppo_kl = compute_grpo_policy_loss(old_log_prob=old_log_prob,
                                                                          log_prob=log_prob,
                                                                          ref_log_prob=ref_log_prob,
                                                                          advantages=advantages,
                                                                          eos_mask=response_mask,
                                                                          cliprange=clip_ratio,
                                                                          kl_ctrl=self.args.kl_ctrl) 
            policy_loss = pg_loss

            stats = {
                'actor/pg_loss': abs(pg_loss.detach().item()),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item()
            }
            return policy_loss, stats

        def loss_func_online_dpo(output, data, meta_info):
            """
            calculate the policy loss
            """
            args = get_args()
            scores = data['rm_scores']
            responses = data['responses']
            device = responses.device
            ref_logprobs = data['ref_log_prob']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            num_examples = responses.shape[0] // 2

            actual_start = torch.arange(responses.size(0), device=responses.device)
            tokenizer = get_tokenizer()
            score_first_eos_index, reward_first_eos_index = find_first_eos_index(responses, tokenizer.eos_token_id)

            scores = scores[[actual_start, score_first_eos_index]]
            contain_eos_token = torch.any(responses == tokenizer.eos_token_id, dim=-1)
            if args.missing_eos_penalty is not None:
                scores[~contain_eos_token] -= args.missing_eos_penalty
                data['rm_scores'] = scores
            first_half, second_half = split_two_prompts(scores)

            mask = first_half >= second_half
            num_examples_range = torch.arange(num_examples, device=device)
            chosen_indices = num_examples_range + (~mask * num_examples)
            rejected_indices = num_examples_range + (mask * num_examples)
            # Build tensor so that the first half is the chosen examples and the second half the rejected examples
            cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
            logits = output[:, -response_length - 1:-1]
            _, _, log_prob = compute_log_probs(logits, responses, per_token=True)

            cr_logprobs = log_prob[cr_indices]
            cr_ref_logprobs = ref_logprobs[cr_indices]

            # mask out the padding tokens
            padding_mask = ~response_mask.bool()
            cr_padding_mask = padding_mask[cr_indices]

            cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
            cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

            # Split the chosen and rejected examples
            chosen_logprobs_sum, rejected_logprobs_sum = split_two_prompts(cr_logprobs_sum)
            chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = split_two_prompts(cr_ref_logprobs_sum)
            pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
            ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

            logits = pi_logratios - ref_logratios

            if args.dpo_loss_type == "sigmoid":
                losses = -F.logsigmoid(self.beta * logits)
            elif args.dpo_loss_type == "ipo":
                losses = (logits - 1 / (2 * self.beta)) ** 2
            else:
                raise NotImplementedError(f"invalid loss type {self.loss_type}")

            loss = losses.mean()
            chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
            rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)

            stats = {
                'actor/pg_loss': loss.detach().item(),
                'beta': self.beta,
                'logps/chosen': chosen_logprobs_sum.mean().detach().item(),
                'logps/rejected': rejected_logprobs_sum.mean().detach().item(),
                'rewards/chosen': chosen_rewards.mean().detach().item(),
                'rewards/rejected': rejected_rewards.mean().detach().item(),
            }
            return loss, stats


        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            input_ids = batch['input_ids']
            attention_mask_1d = batch['attention_mask']
            position_ids = batch['position_ids']
            attention_mask = get_tune_attention_mask(attention_mask_1d)
            output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
            if forward_only:
                meta_info = None
            else:
                meta_info = {'clip_ratio': self.args.clip_ratio}

            loss_funcs = {
                "ray_ppo": loss_func_ppo,
                "ray_online_dpo": loss_func_online_dpo,
                "ray_grpo": loss_func_grpo
            }

            loss_func = loss_funcs.get(self.args.stage)
            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(batches, vpp_size=len(self.model))
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.model,
            num_microbatches=n_micro_batch,
            seq_length=seq_len,  # unused when variable_seq_lengths
            micro_batch_size=self.args.micro_batch_size,  # unused when variable_seq_lengths
            forward_only=forward_only
        )

        return losses_reduced

    def update_policy(self, dataloader: Iterable[DataProto]) -> Dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        model = self.model
        optimizer = self.optimizer
        opt_param_scheduler = self.opt_param_scheduler


        for model_module in self.model:
            model_module.train()

        for data in dataloader:

            for model_chunk in model:
                model_chunk.zero_grad_buffer()
            optimizer.zero_grad()
            if self.args.stage == 'ray_grpo':
                self.args.kl_ctrl = data.meta_info['kl_ctrl']
            metric_micro_batch = self.forward_backward_batch(data)

            update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

            if update_successful:
                increment = 1
                opt_param_scheduler.step(increment=increment)

            for metric in metric_micro_batch:
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

        self.args.consumed_train_samples += self.args.global_batch_size
        self.num_floating_point_operations_so_far += num_floating_point_operations(self.args, self.args.global_batch_size)

        # add empty cache after each compute
        torch.cuda.empty_cache()

        return metrics

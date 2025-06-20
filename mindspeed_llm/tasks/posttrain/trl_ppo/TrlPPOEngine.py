# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import dataclasses
from functools import partial
import torch
import torch.nn.functional as F

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.training import get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    print_rank_0,
    unwrap_model)
from megatron.training.global_vars import (
    get_args,
    get_timers,
    get_tokenizer)
from megatron.training.training import (
    get_optimizer_param_scheduler,
    build_train_valid_test_data_iterators
)
from mindspeed_llm.training.utils import get_tune_attention_mask
from mindspeed_llm.tasks.posttrain.utils import train_valid_test_datasets_provider
from mindspeed_llm.tasks.posttrain.trl_ppo.actor_model import ActorModel
from mindspeed_llm.tasks.posttrain.trl_ppo.utils import model_provider


class TrlPPOEngine():
    def __init__(self):
        self.actor_model = None
        self.ref_model = None
        self.reward_model = None
        self.critic_model = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.actor_opt_param_scheduler = None
        self.critic_opt_param_scheduler = None
        self.tokenizer = None
        self.model_type = ModelType.encoder_or_decoder
        self.process_non_loss_data_func = None
        self.train_data_iterator, self.valid_data_iterator, self.test_data_iterator = None, None, None

    @staticmethod
    def actor_model_provider(pre_process=True, post_process=True):
        return model_provider(is_reward_model=False, pre_process=pre_process, post_process=post_process)

    @staticmethod
    def critic_model_provider(pre_process=True, post_process=True):
        return model_provider(is_reward_model=True, pre_process=pre_process, post_process=post_process)

    def initialize(self):
        """
        This function will run the followings in the order provided:
            1) setup model, optimizer and lr schedule using the model_provider.
            2) call train_val_test_data_provider to get train/val/test datasets.
        """
        args = get_args()
        train_valid_test_datasets_provider.is_distributed = True

        self.actor_model = ActorModel()

        self.actor_model.model, self.actor_optimizer, self.actor_opt_param_scheduler = \
            self.setup_model_and_optimizer(self.actor_model_provider, self.model_type, load_arg='ref_model')

        self.critic_model, self.critic_optimizer, self.critic_opt_param_scheduler = \
            self.setup_model_and_optimizer(self.critic_model_provider, self.model_type, load_arg='reward_model')

        self.ref_model = self.setup_model(self.actor_model_provider, self.model_type, load_arg='ref_model')
        self.reward_model = self.setup_model(self.critic_model_provider, self.model_type, load_arg='reward_model')

        # Change micro_batch_size of data_iterator to rollout_batch_size
        origin_micro_batch_size = args.micro_batch_size
        if args.rollout_batch_size is None:
            args.rollout_batch_size = args.micro_batch_size
        else:
            args.micro_batch_size = args.rollout_batch_size

        self.train_data_iterator, self.valid_data_iterator, self.test_data_iterator \
            = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)

        args.micro_batch_size = origin_micro_batch_size

        self.tokenizer = get_tokenizer().tokenizer

    def forward_only_function(self, data_iterator, model):
        batch = next(data_iterator)

        tokens = batch["padded_query_responses"].clone()
        position_ids = batch["position_ids"]
        attention_mask_1d = batch["attention_mask"].to(bool)
        attention_mask = get_tune_attention_mask(attention_mask_1d)
        logits = model(tokens, position_ids, attention_mask)

        def loss_func(logits: torch.Tensor, **kwargs):
            args = get_args()

            if mpu.is_pipeline_last_stage():
                if args.output_type in ['actor_model', 'ref_model']:
                    if args.sequence_parallel:
                        logits = gather_from_tensor_model_parallel_region(logits)

            return logits, {'logits': logits}

        return logits, partial(loss_func)

    def get_actor_and_critic_train_data(self, data_iterator):

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            return None, None, None, None, None
        data = next(data_iterator)

        tokens = data['padded_query_responses']
        position_ids = data["position_ids"]

        attention_mask_1d = data["attention_mask"].to(bool)
        attention_mask = get_tune_attention_mask(attention_mask_1d)
        return tokens, None, None, attention_mask, position_ids, data

    def actor_loss_func(self, data, logits, **kwargs):
        args = get_args()
        context_length = data['context_length']
        query_responses = data['padded_query_responses']
        advantages = data["advantages"]
        prev_log_probs = data["actor_logprobs"]
        padding_mask = data["padding_mask"]

        curr_log_probs = logits[:, context_length - 1:-1].to(torch.float32)
        curr_log_probs = F.log_softmax(curr_log_probs, dim=-1)
        tokens_indices = torch.unsqueeze(query_responses[:, context_length:], 2)
        curr_log_probs = torch.gather(curr_log_probs, 2, index=tokens_indices).squeeze(2)

        # Calculate clipped PPO surrogate loss function.
        ratios = (curr_log_probs - prev_log_probs).exp()
        ratios_clamped = ratios.clamp(1.0 - args.clip_ratio, 1.0 + args.clip_ratio)

        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped
        actor_loss = masked_mean(torch.max(loss1, loss2), ~padding_mask)

        with torch.no_grad():
            ppo_ratio = masked_mean(ratios.detach(), ~padding_mask)
            ppo_ratio_clamped = masked_mean(ratios_clamped.detach(), ~padding_mask)

            abs_actor_loss = masked_mean(torch.max(torch.abs(loss1), torch.abs(loss2)), ~padding_mask)
            reduced_abs_actor_loss = average_losses_across_data_parallel_group([abs_actor_loss])

        reduced_actor_loss = average_losses_across_data_parallel_group([actor_loss])

        return (
            actor_loss,
            {
                "pg_loss": reduced_actor_loss,
                "abs_pg_loss": reduced_abs_actor_loss,
                "ppo_ratio": ppo_ratio,
                "ppo_ratio_clamped": ppo_ratio_clamped,
            },
        )

    def get_actor_forward_output_and_loss_func(self):

        def fwd_output_and_loss_func(data, model, **kwargs):
            args = get_args()
            tokens, labels, loss_mask, attention_mask, position_ids, data = self.get_actor_and_critic_train_data(data)
            parallel_logits = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask)

            if args.sequence_parallel:
                if mpu.is_pipeline_last_stage():
                    parallel_logits = gather_from_tensor_model_parallel_region(parallel_logits)

            return parallel_logits, partial(self.actor_loss_func, data)

        return fwd_output_and_loss_func


    def critic_loss_func(self, data, curr_values, **kwargs):
        args = get_args()
        context_length = data['context_length']
        returns = data["returns"]
        prev_values = data["values"]
        padding_mask_p1 = data["padding_mask_p1"]

        curr_values = curr_values.squeeze(-1)[:, context_length - 1:-1].to(torch.float32).contiguous()
        curr_values = torch.masked_fill(curr_values, padding_mask_p1, 0)
        curr_values_clipped = torch.clamp(
            curr_values,
            prev_values - args.cliprange_value,
            prev_values + args.cliprange_value,
        )
        vf_losses1 = torch.square(curr_values - returns)
        vf_losses2 = torch.square(curr_values_clipped - returns)

        # Critic loss
        loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), ~padding_mask_p1)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        return loss, {"vf_loss": reduced_loss}

    def get_critic_forward_output_and_loss_func(self):

        def fwd_output_and_loss_func(data, model, **kwargs):
            tokens, labels, loss_mask, attention_mask, position_ids, data = self.get_actor_and_critic_train_data(data)
            curr_values = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask)
            return curr_values, partial(self.critic_loss_func, data)

        return fwd_output_and_loss_func

    def setup_model_and_optimizer(self,
                                  model_provider_func,
                                  model_type,
                                  no_wd_decay_cond=None,
                                  scale_lr_cond=None,
                                  lr_mult=1.0,
                                  load_arg='ref_model',
                                  ):
        """Setup model and optimizer."""
        args = get_args()
        timers = get_timers()

        model = get_model(model_provider_func, model_type)
        unwrapped_model = unwrap_model(model)

        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        config.timers = timers
        optimizer = get_megatron_optimizer(config, model, no_wd_decay_cond,
                                           scale_lr_cond, lr_mult)
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler, load_arg=load_arg)

        # get model without FP16 and/or DDP wrappers
        if args.iteration == 0 and len(unwrapped_model) == 1 \
                and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                optimizer.reload_model_params()

        return model, optimizer, opt_param_scheduler

    def setup_model(self, model_provider_func,
                    model_type,
                    load_arg,
                    ):
        """Setup model."""
        model = get_model(model_provider_func, model_type)
        load_checkpoint(model, None, None, load_arg=load_arg)
        return model

    def compute_advantages_and_returns(self, rollout_batch):
        INVALID_LOGPROB = 1
        args = get_args()

        context_length = rollout_batch['context_length']
        sequence_lengths = rollout_batch['sequence_lengths']
        logprobs = rollout_batch['actor_logprobs']
        ref_logprobs = rollout_batch['ref_logprobs']
        responses = rollout_batch['padded_query_responses'][:, context_length:].contiguous()
        response_lengths = sequence_lengths - context_length - 1

        response_idxs = torch.arange(responses.shape[1], device=logprobs.device).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > response_lengths.unsqueeze(1)

        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

        response_lengths_p1 = response_lengths + 1
        padding_mask_p1 = response_idxs > (response_lengths_p1.unsqueeze(1))

        rewards = rollout_batch["scores"]
        values = rollout_batch["values"]
        values = torch.masked_fill(values, padding_mask_p1, 0)
        kl = (logprobs - ref_logprobs)

        # compute rewards_with_kl
        non_score_reward = -args.kl_coef * kl
        rewards_with_kl = non_score_reward.cpu()
        actual_start = torch.arange(rewards_with_kl.size(0), device=rewards_with_kl.device)
        actual_end = torch.where(response_lengths_p1 < rewards_with_kl.size(1), response_lengths_p1, response_lengths)
        rewards_with_kl[[actual_start.cpu(), actual_end.cpu()]] += rewards.squeeze(1).cpu()
        rewards_with_kl = rewards_with_kl.contiguous().to(torch.cuda.current_device())

        # compute advantages and returns
        lastgaelam = 0
        advantages_reversed = []
        gen_length = responses.shape[1]
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards_with_kl[:, t] + args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + args.gamma * args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, ~padding_mask)
        advantages = torch.masked_fill(advantages, padding_mask, 0)

        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        return (advantages.contiguous().to(torch.cuda.current_device()),
                returns.contiguous().to(torch.cuda.current_device()),
                padding_mask.contiguous().to(torch.cuda.current_device()),
                padding_mask_p1.contiguous().to(torch.cuda.current_device()))

    def set_model_eval(self, model):
        for model_module in model:
            model_module.eval()

    def set_model_train(self, model):
        for model_module in model:
            model_module.train()


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        elif mask_sum == 1:
            bessel_correction = mask_sum
        else:
            bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

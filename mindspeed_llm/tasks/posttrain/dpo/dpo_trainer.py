# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
from typing import Dict, Tuple
from functools import partial
import torch
import torch.nn.functional as F
from megatron.training import get_args, get_model
from megatron.core import mpu, tensor_parallel, parallel_state
from megatron.core.enums import ModelType
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.global_vars import set_args
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.tasks.posttrain.dpo.dpo_model import DPOModel
from mindspeed_llm.tasks.posttrain.utils import compute_log_probs
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank
from mindspeed_llm.training.utils import get_batch_on_this_cp_rank
from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len, get_ring_degree
from mindspeed.core.context_parallel.utils import pad_data


IGNORE_INDEX = -100
class DPOTrainer(BaseTrainer):
    """
    A trainer class for Direct Preference Optimization (DPO).

    This class provides methods for model initialize, computing losses and metrics, and training.
    """


    def __init__(self):
        """
        Initializes the DPOTrainer instance.

        Sets up the instance variables for the model provider, actual micro batch size,
        and initializes the DPO model.
        """
        super().__init__()

        self.args.actual_micro_batch_size = self.args.micro_batch_size * 4
        self.hyper_model = DPOModel(
            self.train_args[1],
            self._init_reference_model()
        )

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""

        args = get_args()

        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        if args.reset_attention_mask:
            keys += ['position_ids', 'actual_seq_len']
        data_type = torch.int64



        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.no_pad_to_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
                batch = {'attention_mask': attention_mask}
                batch = get_batch_on_this_cp_rank(batch)
                return tokens, None, batch['attention_mask'], None
            else:
                # Broadcast data.
                data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
                # Unpack
                labels = data_b.get('labels').long()
                tokens = data_b.get('input_ids').long()
                # ignored label -100
                loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)

                if args.reset_attention_mask:
                    position_ids = data_b.get('position_ids').long()
                    batch = {
                        'tokens': tokens,
                        'labels': labels,
                        'loss_mask': loss_mask,
                        'attention_mask': None,
                        'position_ids': position_ids
                    }
                    actual_seq_len = data_b['actual_seq_len'].view(-1)
                    if args.attention_mask_type == 'causal' \
                            and args.context_parallel_size > 1 \
                            and args.context_parallel_algo == 'megatron_cp_algo':
                        actual_seq_len = pad_data(data_b['actual_seq_len'].view(-1), batch, args.context_parallel_size,
                                                  args.tensor_model_parallel_size)
                        actual_seq_len /= get_ring_degree()
                    set_actual_seq_len(actual_seq_len)
                attention_mask_1d = data_b.get('attention_mask').long()
                attention_mask = get_tune_attention_mask(attention_mask_1d)
                batch = {'attention_mask': attention_mask}
                batch = get_batch_on_this_cp_rank(batch)
                return None, None, batch['attention_mask'], None

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)

        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        # ignored label -100
        loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)

        position_ids = None
        if args.reset_attention_mask:
            position_ids = data_b.get('position_ids').long()
            batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': None,
                'position_ids': position_ids
            }
            actual_seq_len = data_b['actual_seq_len'].view(-1)
            if args.attention_mask_type == 'causal' \
                    and args.context_parallel_size > 1 \
                    and args.context_parallel_algo == 'megatron_cp_algo':
                actual_seq_len = pad_data(data_b['actual_seq_len'].view(-1), batch, args.context_parallel_size,
                                          args.tensor_model_parallel_size)
                actual_seq_len /= get_ring_degree()
            set_actual_seq_len(actual_seq_len)

        attention_mask_1d = data_b.get('attention_mask').long()
        attention_mask = get_tune_attention_mask(attention_mask_1d)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    def loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """DPO Loss function.

        Args:
            input_tensor (torch.Tensor): The tensor with the labels (repeated in double)
            output_tensor (torch.Tensor): The tensor with the Policy and Reference Model's Logits
        """
        args = get_args()
        labels = input_tensor[:args.actual_micro_batch_size // 2, ...]

        all_policy_logits, all_reference_logits = torch.chunk(output_tensor, 2, dim=0)
        all_reference_logits = all_reference_logits.detach()

        loss, metrics = self.get_batch_loss_metrics(
            all_policy_logits,
            all_reference_logits,
            labels
        )

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        metrics['lm loss'] = average_losses_across_data_parallel_group([loss])
        for key in metrics.keys():
            metrics[key] = average_losses_across_data_parallel_group([metrics[key]])

        # Return the average loss, which will be used to calculate the global loss in the forward_step function.
        cp = parallel_state.get_context_parallel_world_size()
        return loss / cp, metrics

    def forward_step(self, data_iterator, model):
        """DPO Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        # Get the batch.
        self.timers('batch-generator', log_level=2).start()
        tokens, labels, attention_mask, position_ids = self.get_batch(data_iterator)

        # Temporary solution, can be optimized in get_batch func.
        if self.args.stage in ['dpo']:
            if attention_mask is not None:
                if isinstance(attention_mask, list):
                    attention_mask = [torch.cat((x, x), dim=0) for x in attention_mask]
                else:
                    attention_mask = torch.cat((attention_mask, attention_mask), dim=0)

        self.timers('batch-generator').stop()

        output_tensor = self.hyper_model(tokens, position_ids, attention_mask)

        return output_tensor, partial(self.loss_func, labels)

    def dpo_loss(
            self,
            policy_chosen_log_probs: torch.Tensor,
            policy_rejected_log_probs: torch.Tensor,
            reference_chosen_log_probs: torch.Tensor,
            reference_rejected_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_log_probs:
            Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_log_probs:
            Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_log_probs:
            Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_log_probs:
            Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
        """
        pi_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
        ref_log_ratios = reference_chosen_log_probs - reference_rejected_log_probs
        logits = pi_log_ratios - ref_log_ratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0.
        # The label_smoothing parameter encodes our uncertainty about the labels and calculates a conservative DPO loss.
        if self.args.dpo_loss_type == "sigmoid":
            losses = (
                    -F.logsigmoid(self.args.dpo_beta * logits) * (1 - self.args.dpo_label_smoothing)
                    - F.logsigmoid(-self.args.dpo_beta * logits) * self.args.dpo_label_smoothing
            )
        elif self.args.dpo_loss_type == "hinge":
            losses = torch.relu(1 - self.args.dpo_beta * logits)
        elif self.args.dpo_loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter
            # for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.args.dpo_beta)) ** 2
        elif self.args.dpo_loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_kl = (policy_chosen_log_probs - reference_chosen_log_probs).mean().clamp(min=0)
            rejected_kl = (policy_rejected_log_probs - reference_rejected_log_probs).mean().clamp(min=0)

            chosen_log_ratios = policy_chosen_log_probs - reference_chosen_log_probs
            rejected_log_ratios = policy_rejected_log_probs - reference_rejected_log_probs
            # As described in the KTO report, the KL term for chosen (rejected)
            # is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.args.dpo_beta * (chosen_log_ratios - rejected_kl)),
                    1 - F.sigmoid(self.args.dpo_beta * (chosen_kl - rejected_log_ratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.args.dpo_loss_type}."
                f" Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
                self.args.dpo_beta
                * (
                        policy_chosen_log_probs - reference_chosen_log_probs
                ).detach()
        )
        rejected_rewards = (
                self.args.dpo_beta
                * (
                        policy_rejected_log_probs - reference_rejected_log_probs
                ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def compute_preference_loss(
            self,
            policy_chosen_log_probs: torch.Tensor,
            policy_rejected_log_probs: torch.Tensor,
            reference_chosen_log_probs: torch.Tensor,
            reference_rejected_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the preference loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_log_probs: Log probabilities of the policy model for the chosen responses.
            policy_rejected_log_probs: Log probabilities of the policy model for the rejected responses.
            reference_chosen_log_probs: Log probabilities of the reference model for the chosen responses.
            reference_rejected_log_probs: Log probabilities of the reference model for the rejected responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the preference loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
        """
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            reference_chosen_log_probs,
            reference_rejected_log_probs
        )
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
            self,
            all_policy_logits,
            all_reference_logits,
            label
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Computes the sum log probabilities of the labels under the given logits.

        Otherwise, the average log probabilities.

        Args:
            all_policy_logits: Logits of the policy model.
            all_reference_logits: Logits of the reference model.
            label: The label tensor.

        Returns:
            A tuple containing the loss tensor and a dictionary of metrics.
        """
        metrics = {}

        (
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            policy_chosen_log_probs_avg,
        ) = self._compute_log_probs(all_policy_logits, label)

        reference_chosen_log_probs, reference_rejected_log_probs, *_ = self._compute_log_probs(
            all_reference_logits,
            label
        )

        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            reference_chosen_log_probs,
            reference_rejected_log_probs,
        )

        sft_loss = -policy_chosen_log_probs_avg
        if self.args.pref_ftx > 1e-6:
            losses += self.args.pref_ftx * sft_loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = ""
        metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.detach().mean()
        if self.args.dpo_loss_type == "orpo":
            metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean()
            metrics["{}odds_ratio_loss".format(prefix)] = (
                    (losses - sft_loss) / self.args.dpo_beta).detach().mean()

        return losses.mean(), metrics

    def _init_reference_model(self):
        """
        Initializes the reference model frozen.

        Returns:
            The initialized reference model.
        """
        model = get_model(self.model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

        self.args.load = self.args.ref_model if self.args.ref_model is not None else self.args.load
        if self.args.load:
            strict = False if self.args.lora_load else True
            # to avoid assert error
            consumed_train_samples = self.args.consumed_train_samples
            self.args.consumed_train_samples = 0
            args_ = get_args()
            if not args_.finetune:
                args_.is_load_refer = True
                args_.no_load_rng = True
                args_.no_load_optim = True
                set_args(args_)
            load_checkpoint(model, None, None, load_arg='load', strict=strict)
            self.args.consumed_train_samples = consumed_train_samples

        return [model[k].eval() for k in range(len(model))]

    def _compute_log_probs(self, all_logits, label) -> Tuple[torch.Tensor, ...]:
        """
        Computes the sum log probabilities of the labels under given logits if loss_type.
        Otherwise, the average log probabilities.
        Assuming IGNORE_INDEX is all negative numbers, the default is -100.

        Args:
            all_logits: The logits tensor.
            label: The label tensor.

        Returns:
            A tuple containing the log probabilities and other tensors.
        """
        label = label[:, 1:].clone()
        all_logits = all_logits[:, :-1, :]
        batch_size = all_logits.size(0) // 2

        all_log_probs, valid_length, _ = compute_log_probs(
            all_logits,
            label
        )

        if self.args.dpo_loss_type in ["ipo", "orpo"]:
            all_log_probs = all_log_probs / torch.clamp(valid_length, min=1)

        chosen_log_probs, rejected_log_probs = all_log_probs.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        all_results = (chosen_log_probs, rejected_log_probs, chosen_log_probs / chosen_length)

        return all_results

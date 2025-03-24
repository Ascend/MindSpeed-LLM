# Copyright 2022 The HuggingFace Team. All rights reserved.
#
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from copy import deepcopy

import numpy as np
import torch
from transformers import AutoTokenizer

import mindspeed_llm.tasks.posttrain.rlxf.utils.torch_functional as F
from mindspeed_llm.tasks.posttrain.rlxf.utils.loggers import Loggers
from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto
from mindspeed_llm.tasks.posttrain.verifier.rule_verifier import preprocess_box_response_for_qwen_prompt, format_reward, reasoning_steps_reward, strict_format_reward, \
base_model_accuracy_reward

logger = Loggers()


class AdaptiveKLController:
    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = F.masked_whiten(advantages, eos_mask)
        advantages = torch.masked_fill(advantages, ~eos_mask, 0)
    return advantages, returns


def compute_group_norm_advantage_return(
        token_level_rewards: torch.Tensor,
        eos_mask: torch.Tensor,
        gamma: torch.Tensor,
        lam: torch.Tensor,
        config
):
    """
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation


    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    response_length = token_level_rewards.size(1)
    returns = torch.zeros_like(token_level_rewards)
    cumulative_return = torch.zeros(token_level_rewards.size(0), device=token_level_rewards.device)

    # Calculate returns by accumulating discounted rewards
    for t in reversed(range(response_length)):
        cumulative_return = token_level_rewards[:, t] + gamma * cumulative_return
        returns[:, t] = cumulative_return
    advantages = deepcopy(returns)
    if not hasattr(config.algorithm, "advantage_whiten") or config.algorithm.advantage_whiten:
        advantages = F.masked_whiten(advantages, eos_mask)
    else:
        advantages = advantages * eos_mask
    return advantages, returns


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO.

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_grpo_policy_loss(old_log_prob, log_prob, ref_log_prob, advantages, eos_mask, cliprange, kl_ctrl):
    """
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        ref_log_prob `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO.

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GRPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    
    ref_approx_kl = ref_log_prob - log_prob
    ratio_kl = torch.exp(ref_approx_kl)
    kl_losses = ratio_kl - ref_approx_kl - 1
    kl_loss = F.masked_mean(kl_losses, eos_mask)
    pg_loss = pg_loss + kl_loss * kl_ctrl.value
    return pg_loss, pg_clipfrac, ppo_kl


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if penalty == "kl":
        return logprob - ref_logprob

    if penalty == "abs":
        return (logprob - ref_logprob).abs()

    if penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    if penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def find_first_eos_index(tensor, eos_token_id):
    """
    找到张量中每一行第一个等于 eos_token_id 的索引。

    Args:
        tensor (torch.Tensor): 输入的张量，形状为 (batch_size, seq_len)。

    Returns:
        torch.Tensor: 每一行中每一行第一个等于 eos_token_id 的索引，形状为 (batch_size,)。
                     如果没有找到，返回 -1。
    """

    is_eos = (tensor == eos_token_id)

    # 使用 torch.argmax 找到第一个等于 eos_id 的索引
    score_first_eos_index = torch.argmax(is_eos.int(), dim=1)
    reward_first_eos_index = torch.argmax(is_eos.int(), dim=1) + 1
    max_id = is_eos.shape[1] - 1
    reward_first_eos_index = torch.min(reward_first_eos_index, torch.tensor(max_id, device=reward_first_eos_index.device))
    has_eos = is_eos.any(dim=1)
    score_first_eos_index[~has_eos] = -1
    reward_first_eos_index[~has_eos] = -1

    return score_first_eos_index, reward_first_eos_index


def apply_kl_penalty(config, data: DataProto, kl_ctrl: AdaptiveKLController):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['rm_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    eos_token_id = data.meta_info['eos_token_id']
    contain_eos_token = torch.any(responses == eos_token_id, dim=-1)
    if config.algorithm.missing_eos_penalty is not None:
        token_level_scores[~contain_eos_token] -= config.algorithm.missing_eos_penalty
        data.batch['rm_scores'] = token_level_scores
    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    penalty=config.algorithm.kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)
    actual_start = torch.arange(token_level_scores.size(0), device=token_level_scores.device)
    score_first_eos_index, reward_first_eos_index = find_first_eos_index(responses, eos_token_id)
    token_level_rewards = - beta * kld
    token_level_rewards[[actual_start, reward_first_eos_index]] += token_level_scores[[actual_start, score_first_eos_index]]

    current_kl = F.masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, config):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    token_level_rewards = data.batch['token_level_rewards']

    if config.algorithm.adv_estimator == 'gae':
        values = data.batch['values']
        advantages, returns = compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                           values=values,
                                                           eos_mask=response_mask,
                                                           gamma=config.algorithm.gamma,
                                                           lam=config.algorithm.lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif config.algorithm.adv_estimator == 'group_norm':
        advantages, returns = compute_group_norm_advantage_return(token_level_rewards=token_level_rewards,
                                                                  eos_mask=response_mask,
                                                                  gamma=config.algorithm.gamma,
                                                                  lam=config.algorithm.lam,
                                                                  config=config)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def compute_score(reward_wg, batch, metrics, config):
    token_level_rewards = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)

    assert reward_wg is not None or config.reward.verifier, "At least one reward should be provided for score computing."

    # 0 for default/general problems, 1 for math problems
    if "categories" in batch.batch.keys():
        use_verifier_mask = batch.batch["categories"][:, 0].squeeze().bool()
    elif hasattr(config.reward, "verifier") and config.reward.verifier:
        use_verifier_mask = torch.ones(len(batch.batch["input_ids"]), dtype=torch.bool)
    else:
        use_verifier_mask = torch.zeros(len(batch.batch["input_ids"]), dtype=torch.bool)

    if reward_wg and (~use_verifier_mask).sum():
        score_tensor = reward_wg.compute_rm_score(batch).batch['rm_scores']

        rm_token_level_rewards = get_last_reward(
            batch,
            rm_scores=score_tensor,
            n_sample_batch=config.actor_rollout_ref.actor_rollout.n_samples_per_prompt,
            metrics=metrics,
            valid_mask=~use_verifier_mask
        )
        token_level_rewards[~use_verifier_mask] += rm_token_level_rewards


    if hasattr(config.reward, "verifier") and config.reward.verifier and use_verifier_mask.sum():
        verifier_token_level_rewards = compute_verifier_score(batch, metrics, config, use_verifier_mask)
        token_level_rewards[use_verifier_mask] += verifier_token_level_rewards

    rewards = DataProto.from_dict(
        tensors={
            'token_level_rewards': token_level_rewards,
            'rm_scores': token_level_rewards
        }
    )

    return batch.union(rewards)


def compute_verifier_score(batch, metrics, config, valid_mask):
    tokenizer = AutoTokenizer.from_pretrained(config.training.tokenizer_name_or_path, trust_remote_code=True)

    responses = batch.batch["responses"][valid_mask]
    str_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
    question = batch.batch["prompts"][valid_mask]
    str_question = tokenizer.batch_decode(question, skip_special_tokens=True)

    reward_index = batch.batch["responses_ori_length"].unsqueeze(1) - 1
    reward_index = reward_index[valid_mask]

    logger.logger.info("=" * 50)
    logger.logger.info(">>>>>>>>>> User:\n")
    logger.logger.info(str_question[0])
    logger.logger.info(">>>>>>>>>> Assistant:\n")
    logger.logger.info(str_responses[0])

    extra_data = {}

    if hasattr(config.training, "dataset_additional_keys"):
        for k in config.training.dataset_additional_keys:
            extra_data[k] = tokenizer.batch_decode(batch.batch[k], skip_special_tokens=True)
            if k == "categories":
                continue
            logger.logger.info(f">>>>>>>>>> {k}")
            logger.logger.info(extra_data[k][valid_mask.nonzero()[0]])

    logger.logger.info("=" * 50)

    labels = [label for label, mask in zip(extra_data.get("labels"), valid_mask) if mask]
    scores = verifier(str_responses, labels, config, metrics, infos=None)

    scores = torch.tensor(
        scores,
        dtype=torch.float32,
        device=reward_index.device
    )

    scores = scores.reshape(-1, config.actor_rollout_ref.actor_rollout.n_samples_per_prompt)
    scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-8)
    scores = scores.reshape(-1).unsqueeze(1)

    token_level_rewards = torch.zeros_like(responses, dtype=torch.float32)
    token_level_rewards.scatter_(1, reward_index, scores)

    return token_level_rewards


def verifier(responses, labels, config, metrics, infos=None):
    """
    User-defined verifier scoring process.

    Parameters:
    ----------
    responses(List[`str`]):
        Actor rollout answers.
    labels(List[`str`]):
        Ground Truth.
    infos(List[`str`], *optional*):
         Additional usable information loaded from the dataset.

    Return:
        scores(List[`float`]): Final scores.
    """
    rule_verifier_function = {
        "acc": preprocess_box_response_for_qwen_prompt,
        "format": format_reward,
        "step": reasoning_steps_reward,
        "strict_format": strict_format_reward,
        "base_acc": base_model_accuracy_reward
    }

    scores = [0.0] * len(labels)

    verifier_function = config.algorithm.verifier_function if hasattr(
        config.algorithm, "verifier_function") else ["acc"]
    verifier_weight = config.algorithm.verifier_weight if hasattr(
        config.algorithm, "verifier_weight") else [1.0]

    for idx, fun_verifier in enumerate(verifier_function):
        if fun_verifier not in rule_verifier_function:
            continue
        score = rule_verifier_function[fun_verifier](sequences=responses, answers=labels)
        metrics[f"grpo/{fun_verifier}_rewards/mean"] = sum(score) / max(len(score), 1)
        scores = [all_score + tmp_score * verifier_weight[idx]
                  for all_score, tmp_score in zip(scores, score)]

    return scores


def get_last_reward(data, rm_scores, n_sample_batch, metrics, valid_mask):
    eos_indices = data.batch["responses_ori_length"].unsqueeze(1) - 1

    # gather reward from eos position
    rm_scores = rm_scores[valid_mask]
    eos_indices = eos_indices[valid_mask]
    reward = rm_scores.gather(dim=1, index=eos_indices).squeeze(1)

    # record raw reward
    metrics[f"grpo/reward_model_rewards/mean"] = sum(reward) / max(len(reward), 1)

    # calculate group norm
    reward = reward.reshape(-1, n_sample_batch)
    reward = (reward - reward.mean(dim=1, keepdim=True)) / (reward.std(dim=1, keepdim=True) + 1e-8)
    reward = reward.reshape(-1)
    token_level_rewards = torch.zeros_like(rm_scores).scatter_(dim=1, index=eos_indices, src=reward.unsqueeze(1).to(rm_scores.dtype))
    return token_level_rewards


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch):
    sequence_score = batch.batch['rm_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    response_length = batch.batch['responses'].shape[-1]

    advantages = batch.batch['advantages']
    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    returns = batch.batch['returns']
    values = batch.batch['values']

    metrics = {
        # score
        'critic/score/mean': torch.mean(sequence_score).detach().item(),
        'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean': F.masked_mean(advantages, response_mask).detach().item(),
        'critic/advantages/max': torch.max(advantages[response_mask]).detach().item(),
        'critic/advantages/min': torch.min(advantages[response_mask]).detach().item(),
        # returns
        'critic/returns/mean': F.masked_mean(returns, response_mask).detach().item(),
        'critic/returns/max': torch.max(returns[response_mask]).detach().item(),
        'critic/returns/min': torch.min(returns[response_mask]).detach().item(),
        # values
        'critic/values/mean': F.masked_mean(values, response_mask).detach().item(),
        'critic/values/max': torch.max(values[response_mask]).detach().item(),
        'critic/values/min': torch.min(values[response_mask]).detach().item(),
        # response length
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
    }
    return metrics


def compute_data_online_dpo_metrics(batch):
    sequence_score = batch.batch['rm_scores'].sum(-1)
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    metrics = {
        # score
        'reward/score/mean': torch.mean(sequence_score).detach().item(),
        'reward/score/max': torch.max(sequence_score).detach().item(),
        'reward/score/min': torch.min(sequence_score).detach().item(),
        # response length
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
    }
    return metrics


def compute_grpo_data_metrics(batch):
    sequence_score = batch.batch['rm_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    response_length = batch.batch['responses'].shape[-1]
    advantages = batch.batch['advantages']
    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()
    returns = batch.batch['returns']
    metrics = {
        # score
        'grpo/score/mean': torch.mean(sequence_score).detach().item(),
        'grpo/score/max': torch.max(sequence_score).detach().item(),
        'grpo/score/min': torch.min(sequence_score).detach().item(),
        # reward
        'grpo/rewards/mean': torch.mean(sequence_reward).detach().item(),
        'grpo/rewards/max': torch.max(sequence_reward).detach().item(),
        'grpo/rewards/min': torch.min(sequence_reward).detach().item(),
        # adv
        'grpo/advantages/mean': F.masked_mean(advantages, response_mask).detach().item(),
        'grpo/advantages/max': torch.max(advantages[response_mask]).detach().item(),
        'grpo/advantages/min': torch.min(advantages[response_mask]).detach().item(),
        'grpo/returns/mean': F.masked_mean(returns, response_mask).detach().item(),
        'grpo/returns/max': torch.max(returns[response_mask]).detach().item(),
        'grpo/returns/min': torch.min(returns[response_mask]).detach().item(),
        # response length
        'response_length/mean': torch.mean(response_length).detach().item(),
        'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(),
        'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(),
    }
    return metrics

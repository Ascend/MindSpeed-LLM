# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

import os
import abc
import logging
from typing import Optional, Union

import torch
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.training import get_args, global_vars
from megatron.core import parallel_state, ModelParallelConfig


class MegatronModuleForCausalLMABC(torch.nn.Module, abc.ABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """

    def __init__(self):
        super(MegatronModuleForCausalLMABC, self).__init__()
        self.top_k = 0
        self.top_p = 1.0
        self.do_sample = False
        self.num_beams = 1
        self.temperature = 1.0
        self.max_length = 128
        self.max_new_tokens = 0
        self.eos_token_id = None
        self.bos_token_id = None
        self.pad_token_id = None
        self.num_return_sequences = 1
        self.length_penalty = 1.0
        self.tokenizer_new = None
        self.detokenize = True
        self.include_input = False
        self.stream = False
        self.return_output_log_probs = False
        self.truncate = False

    @classmethod
    def from_pretrained(
            cls,
            model_provider,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike, None]] = None,
            **kwargs
    ):
        """
        This is an API for initializing model and loading weight.

        Parameters:
        ----------
        model_provider(`func`):
            Function used to generate model objects which is similar to the training define.
        pretrained_model_name_or_path(`str`, *optional*, defaults to None):
           File path of Model weight in megatron format (TP, PP may be used).
           If it is None, the random initialized weights will be used.
        """

    def generate(self, input_ids=None, **kwargs):
        """
        This is an API for text generation which complies with most huggingface definition.

        - *greedy decoding* if `do_sample=False`
        - *top-k decoding* if `top_k>0`
        - *top-p decoding* if `top_p>0.0`
        - *beam-search decoding* if `num_beams>1`

        Parameters:
        ----------
        input_ids(str | list | LongTensor):
            The text entered by the user, e.g. 'Hello!'
            Or
            The text, which encoded by tokenizer, entered by the user, e.g. [0, 13, 5, ...]
            Or
            The List, which includes multi texts or tokens,
            e.g. [['Hello!'], ["How are you?"]] | [[0, 13, 5, ...], [0, 21, ...]].
            Notice that in beam-search mode multi texts or tokens is forbidden.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling ; use greedy decoding otherwise.
        top_k (`int`, *optional*, defaults to 0):
            The number of the highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to modulate the next token probabilities.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token. Optionally,
            use a list to set multiple *end-of-sequence* tokens.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token. Optionally,
            use a list to set multiple *beginning-of-sequence* tokens.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        tokenizer (`obj`, *optional*, defaults to None):
            If you don't want to use the tokenizer initialized by megatron, you can pass it in HF format here.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences. Only activate in beam search mode.
        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch. Only activate
            in beam search mode.
        detokenize (`bool`, *optional*, defaults to True):
            Whether to detokenize tokens into characters.
        include_input (`bool`, *optional*, defaults to False):
            Whether the output contains the context instruction.
        stream (`bool`, *optional*, defaults to False):
            Whether the output is streamed one by one.
        return_output_log_probs(`bool`, *optional*, defaults to False):
            Whether to return a probability distribution for each token.
            Note that the accumulated probability (i.e. Score) of the whole sentence will be return in beam search mode.
        """
        self.top_k = kwargs.pop("top_k", 0)
        self.top_p = kwargs.pop("top_p", 0.0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.max_length = kwargs.pop("max_length", 128)
        self.max_new_tokens = kwargs.pop("max_new_tokens", 0)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.tokenizer_new = kwargs.pop("tokenizer", None)
        self.detokenize = kwargs.pop("detokenize", True)
        self.include_input = kwargs.pop("include_input", False)
        self.stream = kwargs.pop("stream", False)
        self.return_output_log_probs = kwargs.pop("return_output_log_probs", False)
        self.truncate = kwargs.pop("truncate", False)


class MegatronModuleForCausalLM(MegatronModuleForCausalLMABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """

    def __init__(self, *args, **kwargs):
        super(MegatronModuleForCausalLM, self).__init__()
        from megatron.training import get_tokenizer
        from megatron.inference.text_generation.generation import generate_tokens_probs_and_return_on_first_stage, \
            beam_search_and_return_on_first_stage
        from megatron.inference.text_generation.tokenization import tokenize_prompts
        from megatron.inference.text_generation.communication import broadcast_float_list
        from megatron.inference.text_generation.forward_step import ForwardStep
        from megatron.inference.text_generation.generation import _build_attention_mask_and_position_ids

        args = get_args()
        args.max_tokens_to_oom = args.max_tokens_to_oom if hasattr(args, "max_tokens_to_oom") else 4096
        
        # use megatron/p2p for inference p2p comm, so need to assign dtype for pp.
        config = ModelParallelConfig
        if args.bf16:
            config.pipeline_dtype = torch.bfloat16
        elif args.fp16:
            config.pipeline_dtype = torch.float16
        else:
            config.pipeline_dtype = torch.float32
        try:
            self.tokenizer = get_tokenizer().tokenizer
        except AssertionError:
            self.tokenizer = None

        # import module to avoid error of circular import
        self.greedy_search_or_sampling = generate_tokens_probs_and_return_on_first_stage
        self.beam_search_or_sampling = beam_search_and_return_on_first_stage
        self.tokenize_prompts = tokenize_prompts
        self.broadcast_float_list = broadcast_float_list
        self.ForwardStep = ForwardStep
        self.build_attention_mask_and_position_ids = _build_attention_mask_and_position_ids

    @staticmethod
    def _ids_check(ids, tokenizer):
        checked_ids = []
        for per_ids in ids:
            if per_ids == torch.Size([]) and torch.max(per_ids) >= len(tokenizer):
                warning_info = "The output ids exceeds the tokenizer length, " \
                               "the clamp operation is enforced, please check!!"
                logging.warning(warning_info)
                checked_ids.append(torch.clamp(per_ids, min=0, max=len(tokenizer)) - 1)
            else:
                checked_ids.append(per_ids)
        return checked_ids

    @classmethod
    def from_pretrained(
            cls,
            model_provider, pretrained_model_name_or_path: Optional[Union[str, os.PathLike, None]] = None,
            **kwargs
    ) -> MegatronModuleForCausalLMABC:
        from megatron.training import get_model
        from megatron.training.checkpointing import load_checkpoint
        from megatron.core.distributed import DistributedDataParallel as LocalDDP
        from megatron.core.transformer.module import Float16Module as MegatronFloat16Module
        from megatron.training.utils import unwrap_model

        args = get_args()

        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        args.model = get_model(model_provider, wrap_with_ddp=False)

        if pretrained_model_name_or_path:
            args.load = pretrained_model_name_or_path

        if args.load:
            load_checkpoint(args.model, None, None, 'load')

        unwrap_classes = (torchDDP, LocalDDP, MegatronFloat16Module)

        return unwrap_model(args.model, unwrap_classes)[0]

    def generate(self, input_ids=None, broadcast=False, **kwargs):
        args = get_args()

        super(MegatronModuleForCausalLM, self).generate(input_ids=input_ids, **kwargs)

        # =======================================
        # Add additional parameters to args which
        # may be used in original logic of codes
        # =======================================
        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        # =======================================
        # Initialize the tokenizer to choose
        # whether to use customizing tokenizer
        # =======================================
        self._init_tokenizer(args)

        # =======================================
        # Tokenize the prompts
        # =======================================
        context_tokens_tensor, context_length_tensor = self.tokenize_prompts(tokenizer=self.tokenizer,
                                                                             prompts=input_ids,
                                                                             tokens_to_generate=self.max_new_tokens,
                                                                             max_generate_length=self.max_length,
                                                                             add_BOS=False,
                                                                             broadcast=broadcast)
        if not args.use_mcore_models:
            args.seq_length = context_tokens_tensor.shape[1]
            args.max_position_embeddings = args.seq_length

        # =======================================
        # Get the streaming tokens generator
        # =======================================
        if self.num_beams > 1:
            token_stream = self.beam_search_or_sampling(
                args.model[0],
                tokens=context_tokens_tensor,
                lengths=context_length_tensor,
                beam_size=self.num_beams,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                length_penalty=self.length_penalty,
                num_return_gen=self.num_return_sequences
            )
        else:
            token_stream = self.greedy_search_or_sampling(
                args.model[0],
                tokens=context_tokens_tensor,
                lengths=context_length_tensor,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                return_output_log_probs=self.return_output_log_probs
            )

        # =======================================
        # Post processions in order to get final
        # output texts/tokens
        # =======================================
        return self._token_generator(token_stream)

    def score(self, input_ids=None, placeholder_token=None, reward_tokens=None, broadcast=False, **kwargs):
        """
        This function is used in the inference of PRM to perform a single forward pass and extract the scores
        at the corresponding positions, instead of generating tokens one by one.

        Parameters:
        ----------
        placeholder_token(str):
            The delimiter used in the sentence to indicate the end of the reasoning step.
        reward_tokens([str, str, ...]):
            The identifier used to distinguish between correct reasoning steps and incorrect reasoning steps.
        """

        args = get_args()

        if placeholder_token is None:
            placeholder_token = args.placeholder_token
            if placeholder_token is None:
                raise ValueError("placeholder_token cannot be None")

        if reward_tokens is None:
            reward_tokens = args.reward_tokens
            if reward_tokens is None:
                raise ValueError("reward_tokens cannot be None")

        # =======================================
        # Add additional parameters to args which
        # may be used in original logic of codes
        # =======================================
        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        # =======================================
        # Initialize the tokenizer to choose
        # whether to use customizing tokenizer
        # =======================================
        self._init_tokenizer(args)

        # =======================================
        # Tokenize the prompts
        # =======================================
        context_tokens_tensor, context_length_tensor = self.tokenize_prompts(
            tokenizer=self.tokenizer,
            prompts=input_ids,
            tokens_to_generate=1,
            max_generate_length=None,
            add_BOS=False,
            broadcast=broadcast
        )

        # =======================================
        # Deal with PRM score process
        # =======================================
        placeholder_token = self.tokenizer.encode(placeholder_token, add_special_tokens=False)[0]

        # If there is no placeholder_token, it should automatically add one at the end of the token.
        need_to_add = ~torch.any(context_tokens_tensor == placeholder_token, dim=1).squeeze(-1)
        context_tokens_tensor[need_to_add, context_length_tensor[need_to_add]] = placeholder_token

        reward_tokens = [self.tokenizer.encode(val, add_special_tokens=False)[0] for val in reward_tokens]
        placeholder_indices = torch.nonzero(context_tokens_tensor == placeholder_token)

        if not args.use_mcore_models:
            args.seq_length = context_tokens_tensor.shape[1]
            args.max_position_embeddings = args.seq_length

        logits = self._forward_step(context_tokens_tensor)

        scores = []
        for idx, _ in enumerate(range(logits.size(0))):
            scores_per_sentence = []
            for placeholder_idx in placeholder_indices:
                if placeholder_idx[0] == idx:
                    scores_per_sentence.append(logits[placeholder_idx[0], placeholder_idx[1] - 1, reward_tokens])

            scores_per_sentence = torch.nn.functional.log_softmax(torch.stack(scores_per_sentence), dim=-1)
            scores.append(torch.exp(scores_per_sentence[:, 0]))  # only select score in positive token

        return scores

    def _forward_step(self, tokens):
        args = get_args()
        model = args.model[0]

        batch_size = tokens.size(0)
        max_sequence_length = tokens.size(1)

        if max_sequence_length > args.max_position_embeddings:
            raise ValueError("Length of prompt + tokens_to_generate longer than allowed")

        if max_sequence_length * batch_size > args.max_tokens_to_oom:
            raise ValueError("Too many tokens.  " + str(max_sequence_length * batch_size) + " is greater than " + str(
                args.max_tokens_to_oom))

        forward_step = self.ForwardStep(model, batch_size, max_sequence_length)

        with torch.no_grad():
            attention_mask, position_ids = self.build_attention_mask_and_position_ids(tokens)
            logits = forward_step(tokens, position_ids, attention_mask)

        return logits

    def _init_tokenizer(self, args):
        self.tokenizer = self.tokenizer if self.tokenizer_new is None else self.tokenizer_new
        global_vars._GLOBAL_TOKENIZER = self.tokenizer

        if self.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.pad_token_id
        if self.eos_token_id is not None:
            self.tokenizer.eos_token_id = self.eos_token_id
        if self.bos_token_id is not None:
            self.tokenizer.bos_token_id = self.bos_token_id

        if self.tokenizer.eos_token_id is not None:
            args.eos_id = self.tokenizer.eos_token_id
            args.eod_id = self.tokenizer.eos_token_id
        else:
            raise ValueError("Your tokenizer doesn't include eos_token.")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _post_processing(self, output, context_lengths, log_probs):
        if not self.include_input:
            output = [val[context_lengths[i]:] for i, val in enumerate(output)]

        # When batch size > 1, you need truncate the tokens after eos_token_id
        output = self._truncate_in_multi_batch(output)

        if self.detokenize:
            try:
                output_checked = self._ids_check(output, self.tokenizer)
                output = self.tokenizer.batch_decode(output_checked, skip_special_tokens=True)
            except Exception as e:
                error_info = "Meet errors when trying to decode the tokens. "\
                             "Please handle it by yourself."
                logging.error(error_info)
                logging.error(e)

        output = output[0] if len(output) == 1 else output

        if not self.return_output_log_probs:
            res = output
        else:
            if self.num_beams == 1:
                log_probs = [val[context_lengths[i] - 1:, :] for i, val in enumerate(log_probs)] \
                    if log_probs is not None else None

            res = output, log_probs[0] if len(log_probs) == 1 else log_probs

        return res

    def _truncate_in_multi_batch(self, output):
        if len(output) > 1:
            truncated_output = []
            for idx, batch in enumerate(output):
                output[idx] = output[idx][:self.max_new_tokens] if self.max_new_tokens else output[idx]
                trunc_index = torch.nonzero(batch == self.tokenizer.eos_token_id)

                if min(trunc_index.shape):
                    if self.truncate:
                        truncated_output.append(output[idx][:trunc_index.min() + 1])
                    else:
                        output[idx][trunc_index.min():] = self.tokenizer.eos_token_id
                else:
                    truncated_output.append(output[idx])

            if self.truncate:
                output = [val.tolist() if torch.is_tensor(val) else val for val in truncated_output]

        return output

    def _yield(self, token_stream):
        output, context_lengths, log_probs = None, None, None
        for output, context_lengths, log_probs in token_stream:
            if self.stream:
                res = self._post_processing(output, context_lengths, log_probs)
                yield res

        if not self.stream:
            yield self._post_processing(output, context_lengths, log_probs)

    def _token_generator(self, token_stream):
        token_stream = self._yield(token_stream)
        if not self.stream:
            full_output = None
            for tmp in token_stream:
                full_output = tmp
            return full_output
        else:
            return token_stream


class GPTModelInfer(GPTModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer_model = MegatronModuleForCausalLM()

    def generate(self, input_ids=None, **kwargs):
        return self.infer_model.generate(input_ids=input_ids, **kwargs)
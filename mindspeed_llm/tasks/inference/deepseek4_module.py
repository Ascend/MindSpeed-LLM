# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
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
"""DeepSeek4 inference classes built on top of common inference modules."""

import torch
from megatron.training import get_args

from mindspeed_llm.core.models.deepseek4.deepseek4_model import DeepSeek4Model
from mindspeed_llm.tasks.inference.module import GPTModelInfer, MegatronModuleForCausalLM


class DeepSeek4MegatronModuleForCausalLM(MegatronModuleForCausalLM):
    """DeepSeek4 generation facade that keeps common inference behavior isolated."""

    def _truncate_in_multi_batch(self, output):
        truncated_output = []

        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = list(eos_token_id)
        else:
            eos_ids = [eos_token_id]

        for batch in output:
            seq = batch[: self.max_new_tokens] if self.max_new_tokens else batch

            if torch.is_tensor(seq):
                eos_mask = torch.zeros_like(seq, dtype=torch.bool)
                for eos_id in eos_ids:
                    if eos_id is not None:
                        eos_mask |= seq == eos_id

                eos_pos = torch.nonzero(eos_mask, as_tuple=False)
                if eos_pos.numel() > 0:
                    cut = eos_pos[0].item()
                    seq = seq[: cut + 1] if self.truncate else seq[:cut]
            else:
                cut = None
                for i, token_id in enumerate(seq):
                    if token_id in eos_ids:
                        cut = i
                        break

                if cut is not None:
                    seq = seq[: cut + 1] if self.truncate else seq[:cut]

            truncated_output.append(seq)

        return truncated_output

    def _yield(self, token_stream):
        output, context_lengths, log_probs = None, None, None
        last_text = ""

        for output, context_lengths, log_probs in token_stream:
            if self.stream:
                res = self._post_processing(output, context_lengths, log_probs)

                if isinstance(res, str):
                    if res.startswith(last_text):
                        delta = res[len(last_text) :]
                    else:
                        delta = res

                    last_text = res
                    yield delta
                else:
                    yield res

        if not self.stream and output is not None:
            yield self._post_processing(output, context_lengths, log_probs)


class DeepSeek4ModelInfer(GPTModelInfer, DeepSeek4Model):
    """DeepSeek4 model that reuses common GPTModelInfer and overrides only generation details."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer_model = DeepSeek4MegatronModuleForCausalLM()

    def generate(self, input_ids=None, **kwargs):
        original_mtp_states = self._set_mtp_process_for_generation(False)
        result = None
        try:
            result = super().generate(input_ids=input_ids, **kwargs)
            if hasattr(result, "__next__"):
                return self._restore_mtp_after_stream(result, original_mtp_states)
            return result
        finally:
            if not hasattr(result, "__next__"):
                self._restore_mtp_process(original_mtp_states)

    @staticmethod
    def _restore_mtp_process(states):
        for module, mtp_process in states:
            module.mtp_process = mtp_process

    def _set_mtp_process_for_generation(self, mtp_process):
        args = get_args()
        candidates = [self]
        if hasattr(args, "model"):
            candidates.extend(args.model if isinstance(args.model, list) else [args.model])

        states = []
        visited = set()
        try:
            while candidates:
                module = candidates.pop()
                if module is None or id(module) in visited:
                    continue
                visited.add(id(module))
                if hasattr(module, "mtp_process"):
                    states.append((module, module.mtp_process))
                    module.mtp_process = mtp_process
                if hasattr(module, "module"):
                    candidates.append(module.module)
        except Exception:
            self._restore_mtp_process(states)
            raise
        return states

    def _restore_mtp_after_stream(self, token_stream, original_mtp_states):
        try:
            yield from token_stream
        finally:
            self._restore_mtp_process(original_mtp_states)

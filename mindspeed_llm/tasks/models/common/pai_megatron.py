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
from functools import partial

import torch
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import switch_load_balancing_loss_func
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity

def pai_megatron_aux_loss(self, logits: torch.Tensor):
    probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
        logits,
        self.topk,
        use_pre_softmax=True
    )
    
    if self.training and torch.is_grad_enabled():
        # Apply load balancing loss
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        aux_loss_func = partial(
            switch_load_balancing_loss_func,
            probs=scores,
            tokens_per_expert=tokens_per_expert,
            topk=self.topk,
        )
        probs = self.apply_load_balancing_loss(activation=probs, load_balancing_loss_func=aux_loss_func)

    args = get_args()
    if args.moe_token_dispatcher_type == "allgather":
        if args.moe_permutation_async_comm and (
                self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1)):
            from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async
            with torch.no_grad():
                routing_map = gather_from_sequence_parallel_region_to_moe_async(routing_map)
    return probs, routing_map


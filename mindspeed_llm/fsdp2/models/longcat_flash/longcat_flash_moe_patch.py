import torch
from mindspeed_llm.fsdp2.distributed.expert_parallel.dispatcher import dispatch_mlp_combine


def split_experts_by_id(top_k_index: torch.Tensor, top_k_weights: torch.Tensor, split_index: int):
    """
    Split top_k_index / top_k_weights into two groups by expert ID
    Each group maintains the same shape as the input. Fill invalid value 0 for weights.
        group0: experts where expert_id < n
        group1: experts where expert_id >= n
    Return:
        (index0, weight0), (index1, weight1)

    """

    mask_less = top_k_index < split_index

    # load balance
    index0 = top_k_index % split_index
    weight0 = torch.where(mask_less, top_k_weights, 0.0)

    index1 = torch.where(~mask_less, top_k_index, 0)
    weight1 = torch.where(~mask_less, top_k_weights, 0.0)

    return (index0, weight0), (index1, weight1)


def zero_experts_init(module, ep_size, ep_rank):
    if module.num_routed_experts % ep_size != 0:
        raise AssertionError(f'Number of experts({module.num_routed_experts}) is not divisible by ep size({ep_size}).')
    module.num_local_experts = module.num_routed_experts // ep_size
    local_expert_indices_offset = ep_rank * module.num_local_experts
    module.local_expert_indices = [local_expert_indices_offset + i for i in range(module.num_local_experts)]
    if module.num_local_experts > 1:
        module.expert_ids_per_ep_rank = torch.tensor(
            [i % module.num_local_experts for i in range(module.num_routed_experts)],
            dtype=torch.int32,
            device=torch.accelerator.current_device_index(),
        )


def get_zero_experts_forward_fn(ep_group, fused):
    def zero_experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        # normal experts need to be split by EP
        def normal_experts_part_forward(
            hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
        ):
            weights = (self.gate_up_proj.to_local(), self.down_proj.to_local())

            act_fn = getattr(self, 'act_fn', None)

            if top_k_weights.shape[-1] > top_k_index.shape[-1]:
                real_top_k_weights = torch.gather(top_k_weights, -1, top_k_index.long())
            else:
                # top_k_weights are already [B, S, K], use directly.
                real_top_k_weights = top_k_weights

            return dispatch_mlp_combine(
                ep_group,
                fused,
                hidden_states,
                top_k_index,
                real_top_k_weights,
                weights,
                act_fn,
                self.num_routed_experts,
                self.expert_ids_per_ep_rank,
                expert_module=self,
            )

        # zero experts is no need to be split by EP
        def zero_experts_part_forward(
            hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
        ):
            return hidden_states * top_k_weights.sum(dim=-1, keepdim=True)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states_shape[-1])
        (experts_topk_index, experts_topk_weights), (zero_experts_topk_index, zero_experts_topk_weights) = (
            split_experts_by_id(top_k_index, top_k_weights, self.num_routed_experts)
        )

        normal_experts_output = normal_experts_part_forward(hidden_states, experts_topk_index, experts_topk_weights)

        zero_experts_output = zero_experts_part_forward(
            hidden_states, zero_experts_topk_index, zero_experts_topk_weights
        )

        outputs = normal_experts_output + zero_experts_output

        return outputs.view(*hidden_states_shape)

    return zero_experts_forward

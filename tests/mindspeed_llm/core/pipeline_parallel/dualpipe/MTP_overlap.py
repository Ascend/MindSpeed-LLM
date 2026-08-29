#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from contextlib import nullcontext

import torch
from torch import Tensor
from megatron.training import get_args
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, make_viewless_tensor
from megatron.core.tensor_parallel import (
    all_gather_last_dim_from_tensor_parallel_region,
    scatter_to_sequence_parallel_region,
)

from mindspeed.core.pipeline_parallel.fb_overlap.transformer_layer import (
    transformer_layer_forward,
    transformer_layer_forward_moe,
    transformer_layer_backward
)

from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDelayedScaling,
        TENorm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


class TransformerMTPoverlap(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                layer,
                hidden_states,
                attention_mask,
                context=None,
                context_mask=None,
                rotary_pos_emb=None,
                inference_params=None,
                inference_context=None,
                packed_seq_params=None, ):
        with torch.enable_grad():
            output, context_out, graph = transformer_layer_forward_moe(layer,
                                                                   hidden_states,
                                                                   attention_mask,
                                                                   context,
                                                                   context_mask,
                                                                   rotary_pos_emb,
                                                                   inference_params,
                                                                   packed_seq_params)
        args = get_args()
        if args.recompute_mtp_layer:
            graph.deallocate_graph()
            graph.record_layer_inputs(
                attention_mask, context, context_mask, rotary_pos_emb,
                inference_params, packed_seq_params
            )
        ctx.graph = graph

        return output.detach(), context_out

    @staticmethod
    def backward(ctx, *args):
        layer_graph = ctx.graph
        if layer_graph.checkpointed:
            with torch.enable_grad():
                _, _, restored_layer_graph = transformer_layer_forward(
                    layer_graph.layer, layer_graph.layer_input, *layer_graph.layer_inputs
                )
                restored_layer_graph.unperm2_graph = (
                restored_layer_graph.unperm2_graph[0], layer_graph.unperm2_graph[1])
                layer_graph = restored_layer_graph

        transformer_layer_backward(args[0], layer_graph)

        return None, layer_graph.layer_input.grad, None, None, None, None, None, None


def forward_overlap(self,
                    decoder_input: Tensor,
                    hidden_states: Tensor,
                    attention_mask: Tensor,
                    context: Tensor = None,
                    context_mask: Tensor = None,
                    rotary_pos_emb: Tensor = None,
                    attention_bias: Tensor = None,
                    inference_params: InferenceParams = None,
                    packed_seq_params: PackedSeqParams = None,):
    if context is not None:
        raise NotImplementedError(f"multi token prediction + cross attention is not yet supported.")

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    if self.config.fp8:
        import transformer_engine  # To keep out TE dependency when not training in fp8

        if self.config.fp8 == "e4m3":
            fp8_format = transformer_engine.common.recipe.Format.E4M3
        elif self.config.fp8 == "hybrid":
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

        fp8_recipe = TEDelayedScaling(
            config=self.config,
            fp8_format=fp8_format,
            override_linear_precision=(False, False, not self.config.fp8_wgrad),
        )
        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
            )
        fp8_context = transformer_engine.pytorch.fp8_autocast(
            enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
        )
    else:
        fp8_context = nullcontext()

    with rng_context, fp8_context:

        def enorm(tensor):
            tensor = self.enorm(tensor)
            tensor = make_viewless_tensor(
                inp=tensor, requires_grad=True, keep_graph=True
            )
            return tensor

        def hnorm(tensor):
            tensor = self.hnorm(tensor)
            tensor = make_viewless_tensor(
                inp=tensor, requires_grad=True, keep_graph=True
            )
            return tensor

        if self.recompute_mtp_norm:
            self.enorm_ckpt = CheckpointWithoutOutput()
            enorm_output = self.enorm_ckpt.checkpoint(enorm, False, decoder_input)
            self.hnorm_ckpt = CheckpointWithoutOutput()
            hnorm_output = self.hnorm_ckpt.checkpoint(hnorm, False, hidden_states)
        else:
            enorm_output = enorm(decoder_input)
            hnorm_output = hnorm(hidden_states)
        # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
        # and the (i + K)-th tocken's embedding, and combine them with linear projection.
        hidden_states = torch.cat((enorm_output, hnorm_output), -1)
        if self.recompute_mtp_norm:
            self.enorm_ckpt.discard_output()
            self.hnorm_ckpt.discard_output()
            hidden_states.register_hook(self.enorm_ckpt.recompute)
            hidden_states.register_hook(self.hnorm_ckpt.recompute)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel, all gather after linear_fc.
        hidden_states = all_gather_last_dim_from_tensor_parallel_region(hidden_states)
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        trans = TransformerMTPoverlap.apply
        if self.recompute_mtp_layer:
            hidden_states, _ = tensor_parallel.checkpoint(
                trans,
                self.config.distribute_saved_activations,
                self.transformer_layer,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                inference_params,
                packed_seq_params,
            )
        else:
            hidden_states, _ = trans(
                self.transformer_layer,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                inference_params,
                packed_seq_params,
            )

        return hidden_states
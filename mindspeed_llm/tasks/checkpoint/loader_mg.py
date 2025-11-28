# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import copy
import json
import os
import sys
import types
import logging as logger
import torch
from .models import get_megatron_model

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of megatron repository')
    parser.add_argument('--add-qkv-bias', action='store_true',
                       help='Add bias for attention qkv', default=False,
    )
    parser.add_argument('--add-dense-bias', action='store_true',
                       help='Add bias for attention dense', default=False,
    )
    parser.add_argument('--embed-layernorm', action='store_true',
                       help='Add embed layernorm for word embedding', default=False,
    )
    parser.add_argument('--params-dtype', type=str,
                       help='Set weight dtype', default='fp16',
    )
    group.add_argument('--post-norm', action='store_true',
                       help='post norm after attention or mlp.', default=False)
    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Lora target modules.')
    group.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    group.add_argument('--lora-r', type=int, default=16,
                       help='Lora r.')
    group.add_argument('--lora-alpha', type=int, default=32,
                       help='Lora alpha.')
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='Usr moe grouped gemm.')
    group.add_argument("--moe-tp-extend-ep", action='store_true',
                       help="use tp group to extend experts parallelism instead of sharding weight tensor of experts in tp group")
    group.add_argument('--load-from-legacy', action='store_true',
                       help='Is loader legacy')
    group.add_argument('--spec', type=str, default=None, nargs='*',
                        help='Specify the <module_location function_name> pair '
                             'that returns a spec to customize transformer layer, depending on the use case.')
    group.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    group.add_argument("--add-output-layer-bias", action="store_true", default=False,
                       help='Configuration for the output layer bias.')


def build_metadata(args, margs):
    # Metadata.

    # Layernorm has bias; RMSNorm does not.
    if hasattr(margs, 'normalization'):
        norm_has_bias = margs.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.spec = args.spec
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = norm_has_bias
    md.add_output_layer_bias = getattr(margs, "add_output_layer_bias", False)
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = None
    md.checkpoint_args = margs
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.embed_layernorm = margs.embed_layernorm
    md.moe_grouped_gemm = margs.moe_grouped_gemm
    md.moe_tp_extend_ep = margs.moe_tp_extend_ep
    md.spec = margs.spec
    md.num_experts = getattr(margs, "num_experts", None)
    md.num_layer_list = getattr(margs, "num_layer_list", None)
    md.n_shared_experts = getattr(margs, "n_shared_experts", None)
    md.qk_layernorm = getattr(margs, "qk_layernorm", False)
    md.moe_intermediate_size = getattr(margs, "moe_intermediate_size", None)
    md.first_k_dense_replace = getattr(margs, "first_k_dense_replace", None)
    md.moe_layer_freq = getattr(margs, "moe_layer_freq", None)
    md.q_lora_rank = getattr(margs, "q_lora_rank", None)
    md.multi_latent_attention = getattr(margs, "multi_latent_attention", False)
    if md.multi_latent_attention:
        md.qk_pos_emb_head_dim = getattr(margs, "qk_pos_emb_head_dim", None)
        md.qk_head_dim = getattr(margs, "qk_head_dim", None)
        md.q_lora_rank = getattr(margs, "q_lora_rank", None)
        md.kv_lora_rank = getattr(margs, "kv_lora_rank", None)
        md.v_head_dim = getattr(margs, "v_head_dim", None)

    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0

    return md


def get_message_preprocess(model, args):
    # Send embeddings.
    tp_size = args.tensor_model_parallel_size
    message = {
        "word embeddings": torch.cat(
            [model.get_embedding_word_embeddings_weight(tp_rank=tp_rank) for tp_rank in range(tp_size)], dim=0
    )
    }
    if args.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.get_embedding_position_embeddings_weight()
    if args.embed_layernorm:
        message["word embeddings norm_w"] = model.get_embedding_word_embeddings_norm_weight()
        message["word embeddings norm_b"] = model.get_embedding_word_embeddings_norm_bias()

    return message


def get_message_layer_norm(message, model, md, **kwargs):
    # Get non-parallel tensors from tp_rank 0.
    mg_args = model.get_args()
    message["input norm weight"] = model.get_layers_input_layernorm_weight(**kwargs)

    if mg_args.post_norm:
        message["post norm weight"] = model.get_layers_self_attention_post_attention_layernorm_weight(**kwargs)
        message["pre mlp norm weight"] = model.get_layers_self_attention_pre_mlp_layernorm_weight(**kwargs)
        message["post mlp norm weight"] = model.get_layers_self_attention_post_mlp_layernorm_weight(**kwargs)
    else:
        message["post norm weight"] = model.get_layers_self_attention_pre_mlp_layernorm_weight(**kwargs)

    if md.norm_has_bias:
        message["input norm bias"] = model.get_layers_input_layernorm_bias(**kwargs)
        message["post norm bias"] = model.get_layers_self_attention_pre_mlp_layernorm_bias(**kwargs)

    return message


def get_message_layer_attn(message, model, md=None, **kwargs):
    # Grab all parallel tensors for this layer
    qkv_weight = []
    qb_weight = []
    kvb_weight = []
    qkv_bias = []
    dense_weight = []
    proj_lora_A_weight = []
    qkv_lora_B_weight = []
    margs = model.get_args()

    for tp_rank in range(margs.tensor_model_parallel_size):
        kwargs["tp_rank"] = tp_rank
        qkv_weight.append(model.get_layers_self_attention_linear_qkv_weight(**kwargs))
        dense_weight.append(model.get_layers_self_attention_linear_proj_weight(**kwargs))

        if md.linear_bias or margs.add_qkv_bias:
            qkv_bias.append(model.get_layers_self_attention_linear_qkv_bias(**kwargs))
        if getattr(model.get_args(), "multi_latent_attention", False):
            if getattr(model.get_args(), "q_lora_rank", None):
                qb_weight.append(model.get_layers_self_attention_linear_q_up_proj_weight(**kwargs))
            kvb_weight.append(model.get_layers_self_attention_linear_kv_up_proj_weight(**kwargs))

        if margs.save_lora_to_hf:
            proj_lora_A_weight.append(model.get_layers_self_attention_linear_proj_lora_A_default_weight(**kwargs))
            qkv_lora_B_weight.append(model.get_layers_self_attention_linear_qkv_lora_B_default_weight(**kwargs))

    # Handle gated linear units
    # simple concat of the rest
    if getattr(model.get_args(), "qk_layernorm", False):
        if getattr(model.get_args(), "multi_latent_attention", False):
            if getattr(model.get_args(), "q_lora_rank", None):
                message["q layernorm"] = model.get_layers_self_attention_q_layernorm_weight(**kwargs)
            message["kv layernorm"] = model.get_layers_self_attention_kv_layernorm_weight(**kwargs)
        else:
            message["q layernorm"] = model.get_layers_self_attention_q_layernorm_weight(**kwargs)           
            message["k layernorm"] = model.get_layers_self_attention_k_layernorm_weight(**kwargs)
    if getattr(model.get_args(), "multi_latent_attention", False):
        if getattr(model.get_args(), "q_lora_rank", None):
            message["linear qb weight"] = torch.cat(qb_weight, dim=0)
        message["linear kvb weight"] = torch.cat(kvb_weight, dim=0)
    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
    message["dense weight"] = torch.cat(dense_weight, dim=1)
    if md.linear_bias or margs.add_qkv_bias:
        message["qkv bias"] = torch.cat(qkv_bias, dim=0)

    if md.linear_bias or margs.add_dense_bias:
        message["dense bias"] = model.get_layers_self_attention_linear_proj_bias(**kwargs)

    if margs.save_lora_to_hf:
        message["proj lora A"] = torch.cat(proj_lora_A_weight, dim=1)
        message["proj lora B"] = model.get_layers_self_attention_linear_proj_lora_B_default_weight(**kwargs)
        message["qkv lora A"] = model.get_layers_self_attention_linear_qkv_lora_A_default_weight(**kwargs)
        message["qkv lora B"] = torch.cat(qkv_lora_B_weight, dim=0)

    return message


def _get_message_layer_mlp(message, model, md=None, is_moe_mlp=False, **kwargs):
    margs = model.get_args()
    mlp_l0_weight = []
    mlp_l1_weight = []
    mlp_l0_bias = []
    fc1_lora_B = []
    fc2_lora_A = []
    for tp_rank in range(margs.tensor_model_parallel_size):
        kwargs['tp_rank'] = tp_rank
        if is_moe_mlp:
            mlp_l0_weight.append(model.get_layers_mlp_experts_linear_fc1_weight(**kwargs))
            mlp_l1_weight.append(model.get_layers_mlp_experts_linear_fc2_weight(**kwargs))
            if margs.save_lora_to_hf:
                fc1_lora_A = model.get_layers_mlp_experts_linear_fc1_lora_A_default_weight(**kwargs)
                fc1_lora_B.append(model.get_layers_mlp_experts_linear_fc1_lora_B_default_weight(**kwargs))
                fc2_lora_A.append(model.get_layers_mlp_experts_linear_fc2_lora_A_default_weight(**kwargs))
                fc2_lora_B = model.get_layers_mlp_experts_linear_fc2_lora_B_default_weight(**kwargs)
        else:
            mlp_l0_weight.append(model.get_layers_mlp_linear_fc1_weight(**kwargs))
            mlp_l1_weight.append(model.get_layers_mlp_linear_fc2_weight(**kwargs))
            if margs.save_lora_to_hf:
                fc1_lora_A = model.get_layers_mlp_linear_fc1_lora_A_default_weight(**kwargs)
                fc1_lora_B.append(model.get_layers_mlp_linear_fc1_lora_B_default_weight(**kwargs))
                fc2_lora_A.append(model.get_layers_mlp_linear_fc2_lora_A_default_weight(**kwargs))
                fc2_lora_B = model.get_layers_mlp_linear_fc2_lora_B_default_weight(**kwargs)
        if md.linear_bias:
            if is_moe_mlp:
                mlp_l0_bias.append(model.get_layers_mlp_experts_linear_fc1_bias(**kwargs))
            else:
                mlp_l0_bias.append(model.get_layers_mlp_linear_fc1_bias(**kwargs))

    # Handle gated linear units
    if md.swiglu:
        # concat all the first halves ('W's) and all the second halves ('V's)
        for tp_rank in range(margs.tensor_model_parallel_size):
            mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
        message[f"mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
        message[f"mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
    else:
        message[f"mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

    # simple concat of the rest
    message[f"mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
    if md.linear_bias:
        if md.swiglu:
            for tp_rank in range(margs.tensor_model_parallel_size):
                mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
            message[f"mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
            message[f"mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
        else:
            message[f"mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
        message[f"mlp l1 bias"] = model.get_layers_mlp_linear_fc2_bias(**kwargs)
    if margs.save_lora_to_hf:
        message[f"fc1 lora A"] = fc1_lora_A
        message[f"fc1 lora B"] = torch.cat(fc1_lora_B, dim=0)
        message[f"fc2 lora A"] = torch.cat(fc2_lora_A, dim=1)
        message[f"fc2 lora B"] = fc2_lora_B


def get_message_layer_mlp(message, model, md=None, **kwargs):
    # Grab all parallel tensors for this layer
    margs = model.get_args()
    if margs.num_layer_list:
        num_layer_list = [int(x) for x in margs.num_layer_list.split(',')]
        layer_idx = sum(num_layer_list[:kwargs["pp_rank"]]) + kwargs["layer_idx"]
    else:
        layer_idx = kwargs["layer_idx"] + kwargs["pp_rank"] * len(model.get_layers_module(**kwargs)) + kwargs['vp_rank']
    first_k_dense_replace = model.get_first_k_dense_replace()
    moe_layer_freq = model.get_moe_layer_freq()
    shared_expert_gate = getattr(margs, 'shared_expert_gate', None)

    if layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0:
        message["mlp_moe"] = {}
        mlp_router_weight = model.get_layers_mlp_router_weight(**kwargs)
        num_experts_local = margs.num_experts // margs.expert_model_parallel_size
        message["mlp_moe"]["mlp router weight"] = mlp_router_weight
        if shared_expert_gate:
            shared_expert_gate_weight = model.get_layers_mlp_shared_experts_gate_weight_module(**kwargs)
            message["mlp_moe"]["mlp shared_expert_gate weight"] = shared_expert_gate_weight
        weight1 = []
        weight2 = []
        if getattr(margs, "n_shared_experts", None) is not None:
            shared_experts_fc1_weight_list = []
            shared_experts_fc2_weight_list = []
            for tp_rank in range(margs.tensor_model_parallel_size):
                kwargs['tp_rank'] = tp_rank
                if getattr(margs, "n_shared_experts", None) is not None:
                    shared_experts_fc1_weight_list.append(
                        model.get_layers_mlp_shared_experts_linear_fc1_weight(**kwargs))
                    shared_experts_fc2_weight_list.append(
                        model.get_layers_mlp_shared_experts_linear_fc2_weight(**kwargs))
            if md.swiglu:
                for tp_rank in range(margs.tensor_model_parallel_size):
                    shared_experts_fc1_weight_list[tp_rank] = torch.chunk(shared_experts_fc1_weight_list[tp_rank], 2,
                                                                          dim=0)
                message["mlp_moe"]["mlp shared experts linear fc1 weight W"] = torch.cat(
                    [w[0] for w in shared_experts_fc1_weight_list], dim=0
                )
                message["mlp_moe"]["mlp shared experts linear fc1 weight V"] = torch.cat(
                    [w[1] for w in shared_experts_fc1_weight_list], dim=0
                )
            else:
                message["mlp_moe"]["mlp shared experts linear fc1 weight"] = torch.cat(shared_experts_fc1_weight_list,
                                                                                       dim=0)
            message["mlp_moe"]["mlp shared experts linear fc2 weight"] = torch.cat(shared_experts_fc2_weight_list,
                                                                                   dim=1)
        for ep_rank in range(margs.expert_model_parallel_size):
            kwargs["ep_rank"] = ep_rank
            weight1_tp = []
            weight2_tp = []
            for tp_rank in range(margs.tensor_model_parallel_size):
                kwargs['tp_rank'] = tp_rank
                model.get_layers_mlp_router_weight(**kwargs, data=mlp_router_weight)
                if shared_expert_gate:
                    model.get_layers_mlp_shared_experts_gate_weight_module(**kwargs)
                if margs.moe_grouped_gemm:
                    weight1_tp.append(model.get_layers_mlp_experts_weight1_module(**kwargs))
                    weight2_tp.append(model.get_layers_mlp_experts_weight2_module(**kwargs))
            weight1.append(weight1_tp)
            weight2.append(weight2_tp)
            if not margs.moe_grouped_gemm:
                for expert_idx in range(num_experts_local):
                    kwargs["expert_idx"] = expert_idx
                    global_expert_idx = expert_idx + ep_rank * num_experts_local
                    message["mlp_moe"][f"expert {global_expert_idx}"] = {}
                    expert = message["mlp_moe"][f"expert {global_expert_idx}"]
                    _get_message_layer_mlp(expert, model, md, is_moe_mlp=True, **kwargs)
        if margs.moe_grouped_gemm:
            if margs.moe_tp_extend_ep:
                bucket_num = margs.expert_model_parallel_size * margs.tensor_model_parallel_size
                bucket_expert_num = margs.num_experts // bucket_num
                weight2_list = []
                gate_tp_list = [[] for _ in range(margs.num_experts)]
                up_tp_list = [[] for _ in range(margs.num_experts)]
                down_list = [[] for _ in range(margs.num_experts)]
                weight1_list = []
                for ep_rank in range(margs.expert_model_parallel_size):
                    for tp_rank in range(margs.tensor_model_parallel_size):
                        cur_weight1_bucket = weight1[ep_rank][tp_rank]
                        cur_weight2_bucket = weight2[ep_rank][tp_rank]
                        cur_w1_list = torch.chunk(cur_weight1_bucket.view(-1), bucket_expert_num, dim=0)
                        cur_w2_list = torch.chunk(cur_weight2_bucket, bucket_expert_num, dim=0)
                        global_expert_idx = ep_rank * margs.tensor_model_parallel_size + tp_rank
                        for idx in range(bucket_expert_num):
                            local_w1 = cur_w1_list[idx].reshape(margs.hidden_size, -1)
                            local_w2 = cur_w2_list[idx].reshape(-1, margs.hidden_size)
                            # global expert idx
                            expert_idx = global_expert_idx * bucket_expert_num + idx
                            gate, up = torch.chunk(local_w1.view(margs.hidden_size, -1).t(), 2, dim=0)
                            gate_tp_list[expert_idx].append(gate)
                            up_tp_list[expert_idx].append(up)
                            down = local_w2
                            down_list[expert_idx].append(down)

                for expert_idx in range(margs.num_experts):
                    gate = torch.cat(gate_tp_list[expert_idx], dim=0)
                    up = torch.cat(up_tp_list[expert_idx], dim=0)
                    fc1_weight = torch.cat([gate, up], dim=0)
                    weight1_list.append(fc1_weight.t().view(-1))
                    weight2_list.append(torch.cat([tp_weight2 for tp_weight2 in down_list[expert_idx]], dim=1))

            else:
                weight2_list = []
                gate_tp_list = [[] for _ in range(margs.num_experts)]
                up_tp_list = [[] for _ in range(margs.num_experts)]
                weight1_list = []
                for ep_rank in range(margs.expert_model_parallel_size):
                    for tp_rank in range(margs.tensor_model_parallel_size):
                        ep_expert_list = torch.chunk(weight1[ep_rank][tp_rank].view(-1), num_experts_local)
                        for i in range(num_experts_local):
                            gate, up = torch.chunk(ep_expert_list[i].view(margs.hidden_size, -1).t(), 2, dim=0)
                            gate_tp_list[ep_rank * num_experts_local + i].append(gate)
                            up_tp_list[ep_rank * num_experts_local + i].append(up)
                for expert_idx in range(margs.num_experts):
                    gate = torch.cat(gate_tp_list[expert_idx], dim=0)
                    up = torch.cat(up_tp_list[expert_idx], dim=0)
                    fc1_weight = torch.cat([gate, up], dim=0)
                    weight1_list.append(fc1_weight.t().view(-1))
                for ep_rank in range(margs.expert_model_parallel_size):
                    weight2_list.append(torch.cat(
                        [tp_weight2.view(num_experts_local, -1, margs.hidden_size) for tp_weight2 in weight2[ep_rank]],
                        dim=1))
            message["mlp_moe"]["mlp experts weight1 module"] = torch.cat(weight1_list, dim=0).view(margs.hidden_size, -1)
            message["mlp_moe"]["mlp experts weight2 module"] = torch.cat(weight2_list, dim=0).view(margs.num_experts, -1)
    else:
        _get_message_layer_mlp(message, model, md, **kwargs)

    return message


def get_message_postprocess(model, md, **kwargs):
    # Send final norm from tp_rank 0.
    message = {}
    message[f"weight"] = model.get_final_layernorm_weight(**kwargs)
    if md.norm_has_bias:
        message[f"bias"] = model.get_final_layernorm_bias(**kwargs)

    return message


def get_message_output_layer(model, md, **kwargs):
    # Send final norm from tp_rank 0.
    margs = model.get_args()
    tp_size = margs.tensor_model_parallel_size
    message = {}
    if md.output_layer:
        get_output_layer_weight_list = []
        get_output_layer_bias_list = []
        for tp_rank in range(tp_size):
            kwargs["tp_rank"] = tp_rank
            get_output_layer_weight_list.append(
                model.get_output_layer_weight(**kwargs)
            )
            if md.add_output_layer_bias:
                get_output_layer_bias_list.append(
                    model.get_output_layer_bias(**kwargs)
                )
        message[f"weight"] = torch.cat(get_output_layer_weight_list, dim=0)
        if md.add_output_layer_bias:
            message[f"bias"] = torch.cat(get_output_layer_bias_list, dim=0)

    return message


def to_detach(message):
    for key, value in message.items():
        if isinstance(message[key], dict):
            to_detach(value)
        else:
            message[key] = value.detach()


def _load_checkpoint(model_provider, queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    if args.use_mcore_models and args.load_from_legacy:
        args.use_mcore_models = False

    model_mg = get_megatron_model(model_provider, args_cmd=args)
    model_mg.initialize_megatron_args(queue=queue, loader_megatron=True)

    model_mg.set_tensor_model_parallel_world_size(model_mg.args.tensor_model_parallel_size)
    model_mg.set_expert_model_parallel_world_size(model_mg.args.expert_model_parallel_size)
    model_mg.set_pipeline_model_parallel_world_size(model_mg.args.pipeline_model_parallel_size)
    model_mg.set_virtual_pipeline_model_parallel_world_size(model_mg.args.virtual_pipeline_model_parallel_size)

    # Get first pipe stage.
    model_mg.set_tensor_model_parallel_rank(0)
    model_mg.set_pipeline_model_parallel_rank(0)

    margs = model_mg.get_args()
    margs.add_output_layer_bias = getattr(args, "add_output_layer_bias", False)
    margs.moe_grouped_gemm = args.moe_grouped_gemm
    margs.moe_tp_extend_ep = args.moe_tp_extend_ep
    margs.spec = args.spec

    md = build_metadata(args, margs)
    queue.put(md)
    model_mg.get_modules_from_pretrained(pp_stage_cache_flag=True)

    def queue_put(name, msg):
        logger.info(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = get_message_preprocess(model_mg, margs)
    queue_put("embeddings", message)

    pp_size = margs.pipeline_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    total_layer_num = 0
    for vp_rank in range(vp_size):
        for pp_rank in range(pp_size):
            model_mg.set_pipeline_model_parallel_rank(pp_rank)
            model_mg.get_modules_from_pretrained(pp_stage_cache_flag=True)
            kwargs = {"vp_rank": vp_rank, 'pp_rank': pp_rank}
            for layer_idx in range(len(model_mg.get_layers_module(**kwargs))):
                kwargs["layer_idx"] = layer_idx
                message = {}
                message = get_message_layer_norm(message, model_mg, md, **kwargs)
                message = get_message_layer_attn(message, model_mg, md, **kwargs)
                message = get_message_layer_mlp(message, model_mg, md, **kwargs)
                to_detach(message)
                queue_put(f"transformer layer {total_layer_num}", message)
                total_layer_num = total_layer_num + 1

    kwargs = {"pp_rank": pp_size - 1, "vp_rank": vp_size - 1}
    message = get_message_postprocess(model_mg, md, **kwargs)
    queue_put("final norm", message)

    message = get_message_output_layer(model_mg, md, **kwargs)
    if message:
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(model_provider, queue, args):
    try:
        _load_checkpoint(model_provider, queue, args)
    except:
        queue.put("exit")
        raise

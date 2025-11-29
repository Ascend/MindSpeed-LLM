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
import argparse
from pathlib import Path
from functools import wraps
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
from mindspeed_llm.features_manager import FEATURES_LIST


cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir.parent.parent, "configs/finetune/templates.json")


def print_rank0_by_args(args, message):
    """Before initialization of distributed, we only print on rank 0."""
    if args.rank == 0:
        print(message, flush=True)
   
        
def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args_v2(parser)
        return parser

    return wrapper


def parse_args_decorator(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'
    parser = _add_fusion_op_args(parser)
    parser = _add_network_size_args(parser)
    parser = _add_lora_args(parser)
    parser = _add_lu_lora_args(parser)
    parser = _add_data_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_num_layer_allocation(parser)
    parser = _add_profile_args(parser)
    parser = _add_network_args(parser)
    parser = _add_training_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_algorithm_args(parser)
    parser = _add_alibi_args(parser)
    parser = _add_dataset_args(parser)
    parser = _add_cp_args(parser)
    parser = _add_mla_args(parser)
    parser = _add_yarn_args(parser)
    parser = _add_deepseek_moe_args(parser)
    parser = _add_rl_args(parser)
    parser = _add_ndmm_args(parser)
    parser = _add_2d_tp_args(parser)
    parser = _add_hccl_group_buffer_args(parser)
    parser = _add_default_model_args(parser)
    parser = _add_megatron2_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_dualpipe_args(parser)
    parser = _add_ai_framework_args(parser)
    parser = _add_ckpt_args(parser)
    parser = _add_zerc_args(parser)

    for feature in FEATURES_LIST:
        feature.register_args(parser)

    return parser


def process_args_v2(parser):

    MindSpeedFeaturesManager.register_features_args(parser)

    return parser


def _add_zerc_args(parser):
    group = parser.add_argument_group(title='zerc')
    group.add_argument('--moe-zerc', action='store_true', default=False)
    return parser


def _add_ai_framework_args(parser):
    group = parser.add_argument_group(title='ai framework')

    group.add_argument('--ai-framework', type=str, choices=['pytorch', 'mindspore'], default='pytorch', help='support pytorch and mindspore')
    return parser


def _add_default_model_args(parser):
    group = parser.add_argument_group(title='default model mode')

    group.add_argument('--use-mcore-models', action='store_true', dest='use_mcore_models',
                       help='Use Megatron-Core models, will be DEPRECATED in future')
    return parser
    
    
def _add_ckpt_args(parser):
    group = parser.add_argument_group(title='ckpt arguments')
    
    group.add_argument('--ckpt-format', default='torch',
                        choices=['torch', 'torch_dist', 'zarr'],
                        help='Checkpoint format to use.')
    group.add_argument('--use-mp-args-from-checkpoint-args', action='store_true', default=True,
                        help='Copy model parallelism command-line arguments from checkpoint')
    return parser


def _add_mla_args(parser):
    group = parser.add_argument_group(title='multi-head latent attention')

    group.add_argument('--padded-base-length', type=int, default=128,
                       help='Fill Q K V of multi-latent-attention to an integer multiple of this parameter.')
    group.add_argument('--mla-fa-without-pad', action='store_true', default=False, help='Do not pad v_head_dim to q_head_dim in MLA')
    group.add_argument('--mla-mm-split', action='store_true', default=False, help='Split 2 up-proj matmul into 4 in MLA')
    group.add_argument("--mla-zero-memory", action='store_true', default=False, help="Save activation memory in multi-latent-attention.")
    group.add_argument("--mla-up-proj-tp-overlap", action='store_true', default=False, help='overlap up proj tp comm')
    group.add_argument("--recompute-mla-up-proj", action='store_true', default=False, help='recompute up projection in mla')
    group.add_argument('--mla-swap-core-attn-out', action='store_true', default=False, help='swap core_attn_out only in mla.')
    group.add_argument('--mla-fa-divide-qk', action='store_true', default=False,
                       help='Flash attn support mla with seperate q and k.')

    return parser


def _add_yarn_args(parser):
    group = parser.add_argument_group(title='yarn')

    group.add_argument('--beta-fast', type=int, default=32, help='Yarn rope: rope beta fast')
    group.add_argument('--beta-slow', type=int, default=1, help='Yarn rope: rope beta slow')
    group.add_argument('--rope-scaling-factor', type=float, default=1.0, help='Yarn rope: rope factor')
    group.add_argument('--rope-scaling-mscale', type=float, default=1.0, help='Yarn rope: rope mscale')
    group.add_argument('--rope-scaling-mscale-all-dim', type=float, default=0.0, help='Yarn rope: rope mscale all dim')
    group.add_argument('--rope-scaling-original-max-position-embeddings', type=int, default=None,
                       help='Yarn rope: rope original max position embeddings')
    return parser


def _add_deepseek_moe_args(parser):
    group = parser.add_argument_group(title='deepseek moe')

    group.add_argument('--moe-intermediate-size', type=int, default=None, help='The ffn hidden size of MoE layer')
    group.add_argument('--n-shared-experts', type=int, default=None,
                       help='This value is the number of shared experts, which is equal to the intermediate_size '
                            'of the shared experts divided by the moe_intermediate_size.')
    group.add_argument('--first-k-dense-replace', type=int, default=None, help='Set first k layer as dense layer')
    group.add_argument('--moe-layer-freq', type=int, default=None, help='Set the occurrence frequency of the moe layer')
    return parser


def _add_dualpipe_args(parser):
    group = parser.add_argument_group(title='dualpipe')
    group.add_argument('--moe-fb-overlap', action='store_true', default=False)
    group.add_argument('--schedules-method', type=str, default=None, choices=['dualpipev'])
    group.add_argument('--dualpipev-dw-detach', action='store_true', help='detach dw in cooldown to reduce bubble')
    group.add_argument('--moe-unperm2-mem-optim', action='store_true', default=False,
                       help='deallocate unperm2 activation memory by multiplying prob after act func.')
    group.add_argument('--moe-unperm2-mem-optim-swap', action='store_true', default=False)
    return parser


def _add_profile_args(parser):
    group = parser.add_argument_group(title='profiler')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[-1],
                       help='Global ranks to profile.The default value of -1 means to profile all ranks')
    group.add_argument('--profile-export-type', type=str, default='text',
                       choices=['text', 'db'], help='choose the export mode as text or db.')
    group.add_argument('--profile-level', type=str, default='level0',
                       choices=['level_none', 'level0', 'level1', 'level2'], help='profiling level_none, level0, level1, level2')
    group.add_argument('--profile-data-simplification', action='store_true', help='use data simplification mode')
    group.add_argument('--profile-with-stack', action='store_true', help='profiling with stack info')
    group.add_argument('--profile-with-memory', action='store_true', help='profiling with memory info')
    group.add_argument('--profile-record-shapes', action='store_true', help='profiling with shape info')
    group.add_argument('--profile-with-cpu', action='store_true', help='profiling with cpu info')
    group.add_argument('--profile-save-path', type=str, default='./profile_dir',
                       help='path to save profiling files')

    return parser


def _add_cp_args(parser):
    group = parser.add_argument_group(title='cp parallel')
    group.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')
    group.add_argument('--context-parallel-algo', type=str, default='ulysses_cp_algo',
                       choices=['ulysses_cp_algo', 'megatron_cp_algo', 'hybrid_cp_algo', 'mamba_cp_algo'], help='context parallel algorithm')
    group.add_argument('--ulysses-degree-in-cp', type=int, default=None)
    group.add_argument('--attention-mask-type', type=str, default='causal',
                       choices=['causal', 'general'], help='context parallel attention mask type')
    group.add_argument('--attention-mask-type', type=str, default='causal',
                       choices=['causal', 'general'], help='context parallel attention mask type')
    group.add_argument('--use-cp-send-recv-overlap', action='store_true',
                       help='use it to enable cp send-recv-overlap.')
    group.add_argument('--cp-window-size', type=int, default=1,
                       help='inner window size of double ring attention')
    group.add_argument('--attention-mask-on-cpu', action='store_true',
                       help='store full attention mask on CPU instead of NPU')
    group.add_argument('--adaptive-cp-without-coarse', action='store_true',
                       help='does not coarse the attention mask in adaptive_cp feature, only recommended when full'
                            'sequence length is less than 8K and dynamic attention mask is not feasible')
    group.add_argument('--adaptive-cp-dynamic-attn-mask', action='store_true',
                       help='if the attention mask is dynamic across batches')
    group.add_argument('--adaptive-cp-only-reschedule', action='store_true',
                       help='not apply remapping but only rescheduling process in adaptive-cp feature')
    group.add_argument('--adaptive-cp-manually-set-mask-list', action='store_true',
                       help='manually set pre-cooked attention mask list')
    group.add_argument('--kv-head-repeat-before-uly-alltoall', action='store_true', default=True,
                       help='use it to expand key and value for ulysses when GQA/MQA is used.')
    return parser


def _validate_varlen_fa_args(args):
    # varlen FA layout must be TND
    if args.reset_attention_mask:
        args.shape_order = 'TND'
        print_rank0_by_args(args, f"When reset_attention_mask is enabled, shape_order should be TND.")


def _validate_cp_args(args):
    def _check_attention_head(args, uly_size):
        """
        check GQA & ulysses
        """
        head, remainder = divmod(args.num_attention_heads, uly_size * args.tensor_model_parallel_size)
        if not (head >= 1 and remainder == 0):
            raise ValueError(f"num_attention_heads must be divisible by ulysses_size * tensor_model_parallel_size")
        if args.group_query_attention and args.num_query_groups >= 1:
            head_split_by_tp, remainder = divmod(args.num_query_groups, args.tensor_model_parallel_size)
            if not (head_split_by_tp >= 1 and remainder == 0):
                raise ValueError(f"num_query_groups must be divisible by tensor_model_parallel_size")

            if not args.kv_head_repeat_before_uly_alltoall:
                head_split_by_tp_cp, remainder = divmod(head_split_by_tp, uly_size)
                if not (head_split_by_tp_cp >= 1 and remainder == 0):
                    raise AssertionError(
                        'num_query_groups must be divisible by ulysses_size * tensor_model_parallel_size.\n'
                        'Solution 1. adjust the ulysses_size\n'
                        'Solution 2. You can enable --kv-head-repeat-before-uly-alltoall to roll on.\n'
                        'However, performance would be affected since it would increase communication volume \n'
                        'for ulysses alltoall as well as memory usage.')

    if args.context_parallel_size <= 1:
        if args.kv_head_repeat_before_uly_alltoall:
            args.kv_head_repeat_before_uly_alltoall = False
            print_rank0_by_args(args, f"When context_parallel is not activated, kv_head_repeat_before_uly_alltoall would be set to False for reducing memory usage.")
        if args.use_fused_ring_attention_update:
            raise AssertionError(f"fused_ring_attention_update only works when context parallel is activated.")
        return

    # In context parallel we use FA
    args.use_flash_attn = True
    print_rank0_by_args(args, f"[INFO] Setting args.use_flash_attn={args.use_flash_attn} since context parallel is enabled.")
    if not args.use_mcore_models:
        raise AssertionError(f"Context parallel is only supported in Mcore.")

    if args.context_parallel_algo == 'ulysses_cp_algo' or args.context_parallel_algo == 'mamba_cp_algo':
        if args.seq_length % args.context_parallel_size != 0:
            raise ValueError(f"sequence length must be divisible by context_parallel_size")
        _check_attention_head(args, args.context_parallel_size)

    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.seq_length % (2 * args.context_parallel_size) != 0:
            raise ValueError(f"sequence length must be divisible by 2 * context_parallel_size")
        if not (args.cp_window_size >= 1 and args.cp_window_size < args.context_parallel_size):
            raise ValueError(f'cp_window_size should in range [1, context_parallel_size) when using double_ring_attention.')
        n_window, remainder = divmod(args.context_parallel_size, args.cp_window_size)
        if not (n_window >= 1 and remainder == 0):
            raise ValueError(f'context parallel size must be divisible by cp_window_size when using double ring attention.')
        if not (args.cp_window_size >= 1 and args.cp_window_size < args.context_parallel_size):
            raise ValueError(f'cp_window_size should in range [1, context_parallel_size) when using double_ring_attention.')
        n_window, remainder = divmod(args.context_parallel_size, args.cp_window_size)
        if not (n_window >= 1 and remainder == 0):
            raise ValueError(f'context parallel size must be divisible by cp_window_size when using double ring attention.')
        if args.attention_mask_type == 'general':
            if args.micro_batch_size != 1:
                raise ValueError(f'When attention_mask_type is set to general, the value of mbs can only be 1.')

    if args.context_parallel_algo == 'hybrid_cp_algo':
        if args.ulysses_degree_in_cp is None:
            raise ValueError("--ulysses-degree-in-cp must be specified in hybrid_cp_algo")
        ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
        if not (ring_degree > 1 and remainder == 0):
            raise ValueError("--ulysses-degree-in-cp must be divisible by --context-parallel-size")
        if args.seq_length % (2 * args.context_parallel_size) != 0:
            raise ValueError(f"sequence length must be divisible by 2 * context_parallel_size in hybrid cp")
        if not (args.cp_window_size >= 1 and args.cp_window_size < ring_degree):
            raise ValueError(f'cp_window_size should be in range [1, ring_degree) when using double ring attention with hybrid context parallelism.')
        n_window, remainder = divmod(ring_degree, args.cp_window_size)
        if not (n_window >= 1 and remainder == 0):
            raise ValueError(f'ring_degree should be divisible by cp_window_size when using double ring with hybrid context parallelism.')
        _check_attention_head(args, args.ulysses_degree_in_cp)
        if args.attention_mask_type == 'general':
            if args.micro_batch_size != 1:
                raise ValueError(f'When attention_mask_type is set to general, the value of mbs can only be 1.')

    if args.sliding_window:
        raise AssertionError("sliding window is not supported in context parallel.")


def _validate_tocken(args):
    """To avoid invalid tocken configration."""
    if args.pre_tocken > args.seq_length:
        print_rank0_by_args(args, f"[INFO] pre_tocken={args.pre_tocken} would be adjusted to {args.seq_length} for better performance.")
    if args.next_tocken > args.seq_length:
        print_rank0_by_args(args, f"[INFO] next_tocken={args.next_tocken} would be adjusted to {args.seq_length} for better performance.")


def _add_lu_lora_args(parser):
    group = parser.add_argument_group(title='LU-LoRA arguments')
    group.add_argument('--lu-lora-final-layer-index', type=int, default=None,
                       help='Index of last transformer block trained with LU-LoRA.')
    group.add_argument('--lu-lora-lr', type=float, default=1.25e-6,
                        help='Initial learning rate for LU-LoRA layers.')
    group.add_argument('--lu-lora-lr-ratio', type=float, default=1.,
                       help='Ratio of learning rates of LU-LoRA B and LU-LoRA A adapters.')
    return parser


def _add_lora_args(parser):
    group = parser.add_argument_group(title='lora')

    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Lora target modules.')
    group.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    group.add_argument('--lora-r', type=int, default=16,
                       help='Lora r.')
    group.add_argument('--lora-alpha', type=int, default=32,
                       help='Lora alpha.')
    group.add_argument('--lora-modules-to-save', nargs='+', type=str, default=None,
                       help='Lora modules to save.')
    group.add_argument('--lora-register-forward-hook', nargs='+', type=str,
                       default=['word_embeddings', 'input_layernorm'],
                       help='Lora register forward hook.')
    group.add_argument('--lora-fusion', action='store_true',
                       help='use fusion to accelerate lora.')
    group.add_argument('--lora-ckpt-filter', action='store_true', default=False,
                       help='Enable only saving lora checkpoint.')
    group.add_argument('--qlora', action='store_true', default=False,
                        help='Enable QLoRA for fine-tuning with reduced memory usage.')
    group.add_argument('--qlora-save-dequantize', action='store_true', default=False,
                        help='Dequantize weights to original precision when saving in QLoRA tuning.')
    return parser


def _add_moe_args(parser):
    group = parser.add_argument_group(title='moe')
    group.add_argument('--expert-interval', type=int, default=1,
                       help='Use experts in every "expert-interval" layers')
    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time used in legacy moe layer called SwitchMLP.')
    group.add_argument("--use-fused-moe-token-permute-and-unpermute", action='store_true',
                       help="Use fused moe permute and unpermute.")
    group.add_argument("--gemm-gradient-accumulation-fusion", action='store_true',
                       help="Use gradient-accumulation-fusion in gemm.")

    # For megatron_moe drop
    group.add_argument('--moe-token-dispatcher-type', type=str, choices=['allgather', 'alltoall', 'alltoall_seq'], default='allgather',
                       help='The dispatcher type for moe token dispatching.')
    group.add_argument('--noisy-gate-policy', type=str, default=None,
                       help="noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.")
    group.add_argument('--enable-token-rearrange-opt', action='store_true',
                       help="Use this flag to enable token rearrange optimize")
    group.add_argument('--embedding-multiplier-scale', type=float, default=1.0,
                       help='add scale for embedding.')
    group.add_argument('--input-jitter', action='store_false', help='Add noise to the input tensor.')
    group.add_argument('--post-norm', action='store_true', help='post norm after attention or mlp.')
    group.add_argument('--output-multiplier-scale', type=float, default=None, help='Add scale for logits output.')
    group.add_argument("--moe-permutation-async-comm", action='store_true',
                       help="overlap moe permutation 3 all gather communications")
    group.add_argument("--shared-expert-gate", action='store_true',
                       help="moe model has shared expert gate")
    group.add_argument("--shared-expert-gate-output-dimension", type=int, default=1,
                       help="moe model shared expert gate output dimension for qwen2 moe, this parameter can only configured with"
                            "1 or hidden_state")
    group.add_argument('--moe-alltoall-overlap-comm', action='store_true', default=False,
                       help='moe_alltoall_overlap_comm')
    group.add_argument("--cla-share-factor", type=int, default=1,
                       help="Cross-Layer Attention share kv between cla-share-factor layers")
    group.add_argument("--moe-tp-extend-ep", action='store_true',
                    help="use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group")
    group.add_argument("--moe-zero-memory", type=str, default='disable',
                       choices=['disable', 'level0', 'level1'],
                       help="Save activation memory in moe layer.")
    group.add_argument('--moe-zero-memory-num-layers', type=int, default=None,
                       help='the number of layers using moe-zero-memory level1'
                            'in each pp stage.')
    group.add_argument('--moe-allgather-overlap-comm', action='store_true', default=False,
                       help='moe_allgather_overlap_comm')
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data')
    group.add_argument('--is-instruction-dataset', action='store_true', help='use instruction dataset or not')
    group.add_argument('--full-shuffle-instruction-dataset', action='store_true',
                       help='full shuffle instruction dataset or not')
    group.add_argument('--variable-seq-lengths', action='store_true', help='Use variable seq lengths or not.')
    group.add_argument("--tokenizer-kwargs", type=str, nargs='+', default=None,
                       help="Kwargs of the huggingface tokenizer.")
    group.add_argument('--tokenizer-padding-side', type=str, default='right',
            help="tokenizer padding side")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'Llama2Tokenizer',
                                'PretrainedFromHF',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")
    group.add_argument("--input-layernorm-in-fp32", action='store_true',
                       help="Convert input-layernorm to fp32")
    group.add_argument("--no-shuffle", action='store_true',
                       help="Disable data shuffling, mainly for loss comparison.")
    group.add_argument('--neat-pack', action='store_true',
                       help='Use a zigzag attention mask.')
    group.add_argument('--padded-samples', action='store_true',
                       help='fill in the missing samples within an epoch, '
                            'starting at index 0, aligned with the LlamaFatory.')
    group.add_argument("--enable-thinking", type=lambda x: {"true": True, "false": False, "none": None}[x.lower()], default=None,
                       help="Whether or not to enable thinking mode for reasoning models.")

    return parser


def _add_num_layer_allocation(parser):
    group = parser.add_argument_group(title='num_layer_allocation')
    group.add_argument('--num-layer-list',
                       type=str, help='a list of number of layers, '
                                'seperated by comma; e.g., 4,4,4,4')
    return parser


def _add_fusion_op_args(parser):
    group = parser.add_argument_group(title='fusion_op_args')
    group.add_argument("--use-fused-rmsnorm", action='store_true',
                       help="Use fused rmsnorm.")
    group.add_argument("--use-fused-swiglu", action='store_true',
                       help="Use fused swiglu.")
    group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                       help="Use fused rotary-pos-emb.")
    group.add_argument("--use-fused-ring-attention-update", action='store_true',
                       help="Use fused ring attention update.")
    group.add_argument("--use-ascend-mc2", action='store_true',
                       help="Use mc2 for compute-comm overlap in tp.")
    group.add_argument("--use-fused-mlp", action='store_true',
                       help="Use fused mlp.")
    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network_size_args')
    group.add_argument('--padded-vocab-size',
                       type=int,
                       default=None,
                       help='set padded vocab size')
    group.add_argument('--embed-layernorm',
                       action='store_true',
                       default=False,
                       help='set padded vocab size'
                       )
    group.add_argument('--use-glm-rope',
                       action='store_true',
                       help='use custom partial rope in glm model.'
                       )

    group.add_argument('--sliding-window', type=int, default=None,
                       help='Window size when use sliding window attention.')
    group.add_argument('--output-layer-slice-num', type=int, default=1,
                       help='Set the number of slices for the weight of the output_layer')
    return parser


def _add_algorithm_args(parser):
    group = parser.add_argument_group(title='algorithm')

    group.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    group.add_argument('--reuse-fp32-param', action='store_true',
                       help='The distributed training optimizer frees up '
                            'param copies of FP32 to save memory.')
    group.add_argument('--recompute-activation-function', action='store_true',
                       help='Recompute the activation function in MLP layers.')
    group.add_argument('--recompute-activation-function-num-layers', type=int, default=None,
                       help='Can be used together with "--recompute-method block." '
                            'and "--recompute-num-layers". ')
    group.add_argument('--recompute-in-advance', action='store_true',
                       help='recompute early to reduce bubble and improve training.')
    group.add_argument('--recompute-norm', action='store_true',
                       help='Recompute norm in Transformer Layers')
    group.add_argument('--recompute-norm-num-layers', type=int, default=None,
                       help='Recompute norm num layers, can be used together with activation function recompute. ')
    group.add_argument('--o2-optimizer', action='store_true',
                       help='use bf16 exponential moving average to greatly save up memory.')
    group.add_argument('--o2-gradient', action='store_true',
                       help='use bf16 gradient accumulation to greatly save up memory.')
    group.add_argument('--share-kvstates', action='store_true',
                       help='CLA share kv states.')

    return parser


def _add_network_args(parser):
    group = parser.add_argument_group(title='network')

    group.add_argument("--add-qkv-bias", action="store_true", default=False,
                       help='Configuration for the qkv bias.')
    group.add_argument("--add-dense-bias", action="store_true", default=False,
                       help='Configuration for the dense bias.')
    group.add_argument("--add-output-layer-bias", action="store_true", default=False,
                       help='Configuration for the output layer bias.')
    group.add_argument("--skip-bias-add", action="store_false", default=True,
                       help='Configuration for the skip bias.')
    group.add_argument('--add-rmsnorm-offset', action='store_true', default=False,
                       help='RMSNorm unit offset.')
    group.add_argument('--geglu', action='store_true', default=False,
                       help='Geglu activate function.')
    group.add_argument('--input-embeds-norm', action='store_true', default=False,
                       help='input normalization.')
    group.add_argument('--gelu-tanh', action='store_true', default=False,
                       help='Tanh Geglu activate function.')
    group.add_argument('--output-logit-softcapping', type=float, help='output logit softcapping.')
    group.add_argument('--attn-logit-softcapping', type=float, help='attention logit softcapping.')
    group.add_argument('--query-pre-attn-scalar', type=int, help='attention scalar.')
    group.add_argument('--interleave-sliding-window', type=int,
                       help='Window size when use interleave sliding window attention.')
    group.add_argument('--fc-type', type=str, default=None,
                       help='Specifies the internal structure of the MLP module.')
    group.add_argument(
        '--stage',
        default=None,
        choices=["sft", "dpo", "ray_ppo", "ray_online_dpo", "ray_grpo"],
        help='Determine training mode'
    )
    group.add_argument('--cut-max-seqlen', action="store_true", help='Determine training mode')

    return parser


def _add_rl_args(parser):
    group = parser.add_argument_group(title='reinforce learning in dpo')
    group.add_argument(
        '--dpo-beta',
        default=0.1,
        type=float,
        help='The beta parameter for the DPO loss.'
    )
    group.add_argument(
        '--dpo-loss-type',
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo"],
        help='The type of DPO loss to use.'
    )
    group.add_argument(
        '--ref-model',
        default=None,
        type=str,
        help='Path to the reference model used for the PPO or DPO training.'
    )
    group.add_argument(
        '--refer-model-iter',
        type=int,
        default=1,
        help='iteration of the reference model used for the PPO or DPO training.'
    )
    group.add_argument(
        '--dpo-label-smoothing',
        default=0.0,
        type=float,
        help="The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5.",
    )
    group.add_argument(
        '--pref-ftx',
        default=0.0,
        type=float,
        help="The supervised fine-tuning loss coefficient in DPO training.",
    )
    group.add_argument(
        "--is-pairwise-dataset", action='store_true',
        help="Whether the dataset is pairwise format that has a chosen sequence and rejected "
             "sequence, which usually used in reinforce learning.")
    group.add_argument(
        '--reward-tokens',
        nargs='+',
        type=str,
        default=[],
        help="The labels represent the correctness of each reasoning step in the entire reasoning process.",
    )
    group.add_argument(
        "--md5-validate", action='store_true',
        help="Enable md5 validate."
    )
    group.add_argument(
        "--max-prompt-length", default=512, type=int,
        help="Max prompt length in ppo."
    )
    group.add_argument(
        "--num-samples-per-step", default=1, type=int,
        help="Number of samples per step in generation."
    )
    group.add_argument(
        "--rollout-batch-size", default=None, type=int,
        help="actor rollout batch size."
    )
    group.add_argument(
        "--cliprange-value", default=0.2, type=float,
        help="Clip range value."
    )
    group.add_argument(
        "--critic-mini-batch-size", default=1, type=int,
        help="critic mini batch size."
    )
    group.add_argument(
        "--critic-update-epochs", default=1, type=int,
        help="critic update epochs."
    )
    group.add_argument(
        "--ppo-mini-batch-size", default=1, type=int,
        help="ppo mini batch size."
    )
    group.add_argument(
        "--clip-ratio", default=0.2, type=float,
        help="ppo loss clip ratio."
    )
    group.add_argument(
        "--entropy-coeff", default=0.001, type=float,
        help="entropy coefficient."
    )
    group.add_argument(
        "--ppo-epochs", default=1, type=int,
        help="ppo epochs."
    )
    group.add_argument(
        "--shuffle-minibatch", action='store_true',
        help="Enable shuffle minibatches in PPO."
    )
    group.add_argument(
        "--do-sample", action='store_true',
        help="Enable doing sample in actor generations."
    )
    group.add_argument(
        "--missing-eos-penalty", default=0.0, type=float,
        help="eos penalty."
    )
    group.add_argument(
        "--n-samples-per-prompt",
        default=1,
        type=int,
        help="Number of samples per prompt in GRPO."
    )
    group.add_argument(
        '--reward-tokens',
        nargs='+',
        type=str,
        default=[],
        help="The labels represent the correctness of each reasoning step in the entire reasoning process.",
    )
    group.add_argument(
        '--reward-model',
        default=False,
        help="Path to the reference model used for the PPO training."
    )
    group.add_argument(
        '--kl-coef',
        default=0.3,
        type=float,
        help="KL coefficient in PPO training.",
    )
    group.add_argument(
        '--gamma',
        default=1.0,
        type=float,
        help="Discount factor in PPO training.",
    )
    group.add_argument(
        '--lam',
        default=0.95,
        type=float,
        help="Lambda value for GAE in PPO training.",
    )
    group.add_argument(
        "--advantage-whiten",
        default=True,
        help="advantage whiten in GRPO."
    )
    group.add_argument(
        '--dataset-category',
        type=str,
        default=None,
        help='Comma-separated list of dataset category for training,'
             ' 0 for general problems with no accurate answer, 1 for math problems'
    )
    group.add_argument(
        "--extract-content-for-reward",
        default=False,
        help="Extract content in answer tag for reward model to judge."
    )

    return parser


def _add_megatron2_args(parser):
    group = parser.add_argument_group(title='run two megatrons at once')
    group.add_argument('--num-gpus-for-train', type=int, default=None,
                       help='num of GPUs for train in two megatrons, training is set to the first group.')
    group.add_argument('--num-gpus-for-infer', type=int, default=None,
                       help='num of GPUs for inference in two megatrons, inference is set to the first group.')
    group.add_argument('--inference-tensor-model-parallel-size', type=int, default=1,
                       help='TP size for inference group in two megatron')
    return parser


def _add_inference_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--task",
                       nargs='*',
                       default=None, help='The task id to run.')
    group.add_argument("--top-p", type=float, default=0.95, help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=50, help='Top k sampling.')
    group.add_argument("--temperature", type=float, default=0.7, help='Sampling temperature.')
    group.add_argument("--max-length", type=int, default=256, help='Total length of text.')
    group.add_argument("--max-new-tokens", type=int, default=128, help='Size of the output generated text.')
    group.add_argument('--hf-chat-template', action='store_true', default=False,
                       help="Using Huggingface chat template")
    group.add_argument('--add-eos-token', nargs='+', type=str, default=[],
                       help="Use additional eos tokens")
    group.add_argument('--use-kv-cache', action="store_true", default=False,
                       help="Use kv cache to accelerate inference")
    group.add_argument('--history-turns', type=int, default=3, help='Chat turns of histories.')
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    # transformer-impl保持local
    group.add_argument('--transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--enable-recompute-layers-per-pp-rank',
                       action='store_true', default=False,
                       help='If enabled, --recompute-num-layers will mean the number of '
                            'layers recomputed in each pp rank. Otherwise it means the number '
                            'of layers recomputed in each vpp rank.')
    group.add_argument('--pre-tockens', type=int, default=65536,
                       help='pre-tockens is used by Flash attention')
    group.add_argument('--next-tockens', type=int, default=0,
                       help='next-tockens is used by Flash attention')
    group.add_argument('--sparse-mode', type=int, default=0,
                       help='different modes of flash attention mask')
    group.add_argument('--shape-order', type=str, default='SBH',
                       choices=['SBH', 'BSH', 'BSND', 'BNSD'],
                       help='input shape order used by Flash attention')
    group.add_argument('--use-deter-comp',
                       action='store_true',
                       default=False,
                       help='enable deterministic computing for npu')
    group.add_argument('--jit-compile', action='store_true', default=False,
                       help='Setting jit compile mode to True')
    group.add_argument('--prompt-type', type=str, default=None,
                       choices=['default', 'empty', 'trl', 'chatglm2', 'chatglm3', 'chatglm3_system', 'glm4', 'chatml', 'bailing_mini',
                                'chatml_de', 'qwen', 'qwen_r1', "qwen_math_r1", 'llama3', 'llama2', 'mistral', 'mixtral', 'gemma', 'alpaca',
                                'deepseek2', 'deepseek2-lite', 'minicpm3', 'cpm', 'baichuan2', 'deepseek3', 'intern2', 'hunyuan', 'qwen3', 'magistral', 'plm'],
                       help='Which template to use for constructing prompts in training/inference.'  'e.g., "qwen"')
    group.add_argument('--prompt-type-path', type=str, default=TEMPLATES_DIR,
                       help='Path to the json file of templates.')
    group.add_argument('--pad-to-multiple-of', type=int, default=8,
                       help='Used for Padding multiple in finetune. The default is 8.')
    group.add_argument('--scale-emb', type=float, default=None,
                       help='scale embed tokens')
    group.add_argument('--dim-model-base', type=float, default=None,
                       help='dim-model-base')
    group.add_argument('--no-cut-token', action='store_true', default=False,
                       help='Used for not cut token in finetune.')
    group.add_argument('--scale-depth', type=float, default=None,
                       help='scale-depth')
    group.add_argument('--swap-attention', action='store_true', default=False,
                       help='switch to open swap-attention feature.'
                            'The default is False.')
    group.add_argument('--swap-modules', type=str, default=None,
                       help='Swap modules for model. Should be used together with "--swap-attention."')
    group.add_argument('--load-checkpoint-loosely', action='store_true', default=False,
                       help='Enable loading checkpoint not strictly.')
    group.add_argument('--no-post-layer-norm', action='store_true', default=False,
                       help='Disable final layer norm.')
    group.add_argument('--return-document-ids', action='store_true', default=False,
                       help='Return document ids when get batch.')
    group.add_argument('--reset-attention-mask', action='store_true', default=False,
                       help='Return document ids when get batch.')

    # for swap-optimizer
    group.add_argument('--swap-optimizer', action='store_true', default=False,
                       help='swap optimizer to cpu.')
    group.add_argument('--swap-optimizer-times', type=int, default=16,
                       help='Each swap will be moved (len(shard_fp32_from_float16) // swap_optimizer_times) elements')
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--local-rank', type=int, default=int(os.getenv('LOCAL_RANK', '0')),
                       help='Local rank passed from distributed launcher for torch2.x.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=30,
                       help='Timeout minutes for torch.distributed.')
    return parser


def _add_ndmm_args(parser):
    group = parser.add_argument_group(title='ndmm')
    group.add_argument('--use-nd-matmul', action='store_true', default=False,
                       help='use use-nd-matmul to replace megatron-style tensor parallel')
    group.add_argument('--nd1-dim1-size', type=int, default=1,
                       help='Dim1 of the first nd matmul when use-3d-matmul is True')
    group.add_argument('--nd2-dim1-size', type=int, default=1,
                       help='Dim1 of the second nd matmul when use-3d-matmul is True')
    return parser


def _add_2d_tp_args(parser):
    group = parser.add_argument_group(title='2d-tp')
    group.add_argument('--tp-2d', action='store_true', default=False,
                       help='use use-2d-tp to replace megatron-style tensor parallel')
    group.add_argument('--tp-x', type=int, default=1,
                       help='the fist dim tensor parallel size for Linear')
    group.add_argument('--tp-y', type=int, default=1,
                       help='the second dim tensor parallel size for Linear')
    group.add_argument('--enable-overlap-ag-with-matmul', action='store_true', default=False,
                       help='use enable-overlap-ag-with-matmul to overlap all-gather with matmul')
    group.add_argument('--enable-overlap-matmul-with-rs', action='store_true', default=False,
                       help='use enable-overlap-matmul-with-rs to overlap matmul with reduce-scatter')
    group.add_argument('--enable-backward-overlap-ag-with-matmul', action='store_true', default=False,
                       help='use enable-backward-overlap-ag-with-matmul to overlap all-gather  with matmul in backward')
    return parser


def add_parser_argument_choices_value(parser, argument_name, value):
    if parser._actions:
        for action in parser._actions:
            if isinstance(action, argparse._ArgumentGroup):
                add_parser_argument_choices_value(action, argument_name)
            elif isinstance(action, argparse.Action) and argument_name in action.option_strings:
                action.choices.append(value)


def _add_alibi_args(parser):
    add_parser_argument_choices_value(parser, "--position-embedding-type", 'alibi')

    group = parser.add_argument_group(title='alibi')
    group.add_argument('--square-alibi-mask',
                       action='store_true',
                       default=False,
                       help='attention mask of alibi is squared')
    group.add_argument('--fill-neg-inf',
                       action='store_true',
                       default=False,
                       help='fill alibi with negative inf')

    return parser


def _add_dataset_args(parser):
    group = parser.add_argument_group(title='dataset_args')
    group.add_argument('--no-shared-storage',
                       action='store_true',
                       default=False,
                       help='if no shared storage, set it'
                       )
    group.add_argument('--dataset-additional-keys',
                       nargs='*',
                       default=[],
                       help='Additional keys need to be add from dataset.'
                      )
 
    return parser


def _add_hccl_group_buffer_args(parser):
    group = parser.add_argument_group(title='hccl-group-buffer')
    group.add_argument('--hccl-group-buffer', type=str, default=None,
                       help='the hccl buffer for group')

    return parser


def _validate_create_attention_mask_in_dataloader(args):
    args.create_attention_mask_in_dataloader = False
    reset_data = args.reset_attention_mask
    alibi_without_flash_attn = args.position_embedding_type == 'alibi' and not args.use_flash_attn
    if reset_data or alibi_without_flash_attn or args.tokenizer_padding_side == "left":
        args.create_attention_mask_in_dataloader = True
    if reset_data and args.attention_mask_type == 'causal':
        args.create_attention_mask_in_dataloader = False
    print_rank0_by_args(args, f"[INFO] Setting args.create_attention_mask_in_dataloader to {args.create_attention_mask_in_dataloader} "
                 f"since reset_data={reset_data} or alibi_without_flash_attn={alibi_without_flash_attn} or "
                 f"args.tokenizer_padding_side={args.tokenizer_padding_side}")

    if not args.reset_attention_mask and args.neat_pack:
        raise ValueError("Require set `--reset-attention-mask` when `--neat-pack` is set.")

    if args.context_parallel_size > 1 and args.reset_attention_mask and args.attention_mask_type == 'causal':
        if args.context_parallel_algo != 'megatron_cp_algo':
            raise ValueError('accelerated eod reset mode only support ring attention')


def _validate_position_embedding(args):
    """
    validate position embedding arguments.
    """
    if args.use_glm_rope and args.use_fused_rotary_pos_emb:
        raise AssertionError('Fused rotary embedding is not supported in glm rope.')
    if args.position_embedding_type == 'alibi' and args.sliding_window is not None:
        raise AssertionError('Sliding Window Attention is forbidden when use alibi.')
    if args.tokenizer_padding_side == 'left' and args.position_embedding_type == 'alibi':
        raise AssertionError('Alibi is not support tokenizer-padding-side left now.')
    if not args.use_fused_rmsnorm:
        if args.swap_attention and len(args.lora_target_modules) != 0:
            raise AssertionError('When not use_fused_rmsnorm, swap_attention cannot be used in lora fune-tuning.')


def _validate_recompute_args(args):
    """
    validate re-computation arguments.
    """
    enable_pp_vpp = args.num_layers_per_virtual_pipeline_stage
    enable_vanilla_recomputation = args.recompute_granularity is not None and args.recompute_method == 'block'
    enable_swap = args.swap_attention
    enable_recompute_activation = args.recompute_activation_function
    enable_recomputation = enable_vanilla_recomputation or enable_swap or enable_recompute_activation
    if args.enable_recompute_layers_per_pp_rank and not (enable_pp_vpp and enable_recomputation):
        raise AssertionError("enable-recompute-layers-per-pp-rank should be works with pipeline and virtual pipeline, when enabling re-computation.")

    if args.recompute_activation_function:
        if args.recompute_method == "uniform":
            raise AssertionError('uniform recomputation is not compatible with activation function recomputation.')
        if args.recompute_granularity == "selective":
            raise AssertionError('--recompute-activation-function is not compatible with selective recomputation.')
        
    if args.recompute_norm:
        if args.recompute_method == "uniform":
            raise AssertionError('uniform recomputation is not compatible with norm recomputation.')
        if args.recompute_granularity == "selective":
            raise AssertionError('--recompute-norm is not compatible with selective recomputation')
        if not args.use_mcore_models:
            raise AssertionError('--recompute-norm is only supported with mcore models')
        
    if args.swap_attention and args.swap_modules is None:
        if args.use_mcore_models:
            args.swap_modules = "input_layernorm,self_attention,pre_cross_attn_layernorm"
        else:
            args.swap_modules = "input_norm,self_attention,post_attention_norm"


def _validate_instruction_finetune(args):
    if args.variable_seq_lengths:
        if args.context_parallel_size > 1:
            raise AssertionError('Context parallelism is forbidden when use variable seq lengths.')
        if args.num_experts is not None and args.moe_token_dispatcher_type == "allgather":
            raise AssertionError('moe_token_dispatcher_type "allgather" is forbidden when use variable seq lengths. you can choose "alltoall_seq"')


def _validate_train_mode_conflict(args):
    if args.stage == 'sft':
        if args.variable_seq_lengths:
            if args.neat_pack or args.reset_position_ids:
                raise AssertionError('pack mode is forbidden when using variable seq lengths.')


def _validate_inference_args(args):
    if args.prompt_type is not None and hasattr(args, "hf_chat_template") and args.hf_chat_template:
        raise AssertionError('Prompt-type is forbidden when use huggingface chat template.')

    if hasattr(args, "history_turns") and args.history_turns < 0:
        raise AssertionError('History turns of chat must greater than 0.')


def _validate_evaluation_args(args):
    # five shot only supported on mmlu and ceval now
    if args.prompt_type is not None and hasattr(args, "task") and (args.task == "mmlu" or args.task == "ceval"):
        train_dir = os.path.join(os.path.dirname(args.task_data_path), "dev")
        if not os.path.isdir(train_dir) or not os.path.isdir(args.task_data_path):
            raise ValueError(f"Test and dev directory must exists when specify prompt_type in evaluation")


def _validate_moe_args(args):
    if not args.use_mcore_models and args.num_experts and args.num_experts > 1:
        raise ValueError(f'MOE is not supported in legacy model. Please activate `--use-mcore-models` to enable moe features.')
    if args.moe_alltoall_overlap_comm or args.moe_allgather_overlap_comm:
        if not args.moe_permutation_async_comm or not args.moe_grouped_gemm:
            raise AssertionError(
                '`--moe-alltoall-overlap-comm` or `--moe-allgather-overlap-comm` only support with `--moe-permutation-async-comm` and `--moe-grouped-gemm`.')
    if args.moe_alltoall_overlap_comm and not args.moe_token_dispatcher_type == 'alltoall_seq':
        raise AssertionError('`--moe-alltoall-overlap-comm` only support with `--moe-token-dispatcher-type alltoall_seq`.')
    if not args.moe_tp_extend_ep and args.moe_alltoall_overlap_comm and args.tensor_model_parallel_size > 1:
        raise AssertionError(
            '`--moe-alltoall-overlap-comm` do not support tp for now. only support with moe_tp_extend_ep when tp > 1.')
    if args.moe_zero_memory_num_layers is not None:
        num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
        if args.moe_zero_memory_num_layers < 0 or args.moe_zero_memory_num_layers > num_layers_per_pipeline_stage:
            raise AssertionError('`--moe-zero-memory-num-layers` must be between 0 and num layers per pipeline stage')
        if args.moe_zero_memory == "disable":
            raise AssertionError('`--moe-zero-memory` must be enabled when using `--moe-zero-memory-num-layers`')
    if args.moe_zero_memory != "disable" and not (args.moe_alltoall_overlap_comm or args.moe_fb_overlap):
        raise AssertionError('`--moe-zero-memory` only supports `--moe-alltoall-overlap-comm` and `--moe-fb-overlap` for now.')
    if args.moe_zero_memory != "disable" and args.recompute_method is not None:
        raise AssertionError('`--moe-zero-memory` does not support full recomputation for now.')
    if args.moe_allgather_overlap_comm and not args.moe_token_dispatcher_type == 'allgather':
        raise AssertionError('`--moe-allgather-overlap-comm` only support with `--moe-token-dispatcher-type allgather`.')
    if args.moe_allgather_overlap_comm and not args.tensor_model_parallel_size > 1 and not args.expert_model_parallel_size > 1:
        raise AssertionError('`--moe_allgather_overlap_comm` requires enabling tp or ep.')
    if args.shared_expert_gate and args.gradient_accumulation_fusion:
        raise AssertionError('args.shared_expert_gate does not support gradient_accumulation_fusion.')
    if args.moe_tp_extend_ep and args.moe_token_dispatcher_type == 'allgather':
        raise AssertionError('moe_tp_extend_ep is not supported when moe_token_dispatcher_type = allgather')


def _validate_mla(args):
    if args.multi_latent_attention:
        if args.padded_base_length < 1:
            raise AssertionError('The value of padded_base_length cannot be less than 1.')
        if args.mla_up_proj_tp_overlap:
            if not args.mla_mm_split:
                raise ValueError('--mla-up-proj-tp-overlap can only be used with mla-mm-split by now')
            if not args.sequence_parallel:
                raise ValueError('--mla-up-proj-tp-overlap should be used with sequence parallel')
        if args.recompute_mla_up_proj:
            if not args.mla_up_proj_tp_overlap:
                raise ValueError('--recompute-mla-up-proj can only be used with --mla-up-proj-tp-overlap')
            if args.mla_zero_memory:
                raise ValueError('--recompute-mla-up-proj is incompatible with --mla-zero-memory')
        if args.mla_swap_core_attn_out:
            if args.schedules_method != "dualpipev":
                raise AssertionError('--mla-swap-core-attn-out can only be used with dualpipev by now.')
            if not args.moe_fb_overlap:
                raise AssertionError('--mla-swap-core-attn-out can only be used with --moe-fb-overlap by now.')


def _validate_yarn(args):
    if args.rope_scaling_type == "yarn":
        if args.rope_scaling_original_max_position_embeddings is None:
            raise AssertionError('The parameter rope_scaling_original_max_position_embeddings should be set '
                                 'when use yarn.')
        if args.multi_latent_attention and not args.rope_scaling_factor:
             raise AssertionError('The parameter rope_scaling_factor should be set when use yarn.')
        if args.max_position_embeddings / args.rope_scaling_original_max_position_embeddings != args.rope_scaling_factor:
            raise AssertionError('The parameter rope_scaling_factor should be equal to max_position_embeddings'
                                 'divided by rope_scaling_original_max_position_embeddings.')


def _validate_transformer_block_build_layers(args):
    if args.num_experts is None:
        if args.first_k_dense_replace is not None or args.moe_layer_freq is not None:
            raise AssertionError('First-k-dense-replace and moe-layer-freq must be None when not using MoEs')
    else:
        if args.first_k_dense_replace and args.num_layers <= args.first_k_dense_replace:
            raise AssertionError('Num-layer ({}) must be greater than first-k-dense-replace ({}) when first-k-dense-replace is set.'.format(args.num_layers,
            args.first_k_dense_replace))
        if args.first_k_dense_replace and args.pipeline_model_parallel_size > 1:
            if args.first_k_dense_replace >= args.num_layers // args.pipeline_model_parallel_size:
                raise AssertionError('When using first-k-dense-replace, it is not allowed for all layers within a pp stage to be dense layers.')
    if args.num_experts is not None and args.use_ascend_mc2 and args.moe_grouped_gemm:
        raise AssertionError('Moe Grouped Gemm is not supported with mc2 in MOE model.')

    if args.num_layer_list:
        if len(args.num_layer_list.split(',')) != args.pipeline_model_parallel_size:
            raise ValueError("len(args.num_layer_list) != args.pipeline_model_parallel_size")
        if not args.pipeline_model_parallel_size > 1:
            raise ValueError("Dynamic pipeline model should work with pipeline parallel.")
        if args.num_layers_per_virtual_pipeline_stage:
            raise ValueError("Dynamic pipeline model and virtual pipeline cannot be enabled at the same time.")

    if args.use_ascend_mc2 and args.use_ascend_coc:
        raise AssertionError('--mc2 and coc can not be used together')




def get_layer_offset(pp_size, num_layer_list):
    """
    Get layer number offset for pp stage. global_layer_number = local_layer_number + layer_number_offset
    For instance, num-layer-list=1,3,3,1,
    (1,123,123,1) + (0,1,4,7) = (1,234,567,8)
    For each pp_stage, we have layer_number_offset = prefix_sum[pp_stage + 1]
    """
    prefix_sum = [0] * (pp_size + 1)
    # take prefix_sum[0] as sentinel
    for index, num_layers in enumerate(num_layer_list):
        prefix_sum[index + 1] = prefix_sum[index] + num_layers
    return prefix_sum


def _validate_output_layer_slice_num(args):
    if args.output_layer_slice_num < 1:
        raise AssertionError('Output_layer_slice_num must be greater than 0.')
    elif args.output_layer_slice_num > 1:
        if args.tensor_model_parallel_size > 1:
            raise AssertionError('When output_layer_slice_num is greater than 1, only support TP size is 1.')
        if (args.padded_vocab_size is not None) and (args.padded_vocab_size % args.output_layer_slice_num != 0):
            raise AssertionError('Output_layer_slice_num needs to be divisible by padded_vocab_size.')
        elif (args.vocab_size is not None) and (args.vocab_size % args.output_layer_slice_num != 0):
            raise AssertionError('Output_layer_slice_num needs to be divisible by vocab_size.')

        if args.gradient_accumulation_fusion:
            args.gradient_accumulation_fusion = False
            print_rank0_by_args(f"gradient_accumulation_fusion would be set to {args.gradient_accumulation_fusion} "
                                f"since args.output_layer_slice_num > 1")


def core_transformer_config_from_args_wrapper(fn):
    @wraps(fn)
    def wrapper(args, config_class=None):
        config = fn(args, config_class)
        # Turn down batch_p2p_comm only when pp2vpp
        if args.pipeline_model_parallel_size == 2 and args.num_layers_per_virtual_pipeline_stage is not None:
            config.batch_p2p_comm = False

        if args.moe_expert_capacity_factor:
            # moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token will be dropped. The default is None.
            config.moe_expert_capacity_factor = args.moe_expert_capacity_factor
            # moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False.
            config.moe_pad_expert_input_to_capacity = args.moe_pad_expert_input_to_capacity
            # The policy to drop tokens. Can be either "prob" or "position". If "prob", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.
            config.moe_token_drop_policy = args.moe_token_drop_policy

        if args.num_layer_list:
            # For num layer list, we turn string into int list and store it in transformer config.
            config.num_layer_list = list(map(int, args.num_layer_list.split(',')))
            config.layer_offset = get_layer_offset(args.pipeline_model_parallel_size, config.num_layer_list)
            # validate num_layer_list
            if config.layer_offset[args.pipeline_model_parallel_size] != args.num_layers:
                raise ValueError(f"Incorrect num_layer_list config since its sum({config.layer_offset[args.pipeline_model_parallel_size]} is unequal to total num layers({args.num_layers}).")
        else:
            config.num_layer_list = None

        return config

    return wrapper


def _validate_optimizer(args):
    if args.reuse_fp32_param and not args.bf16:
        raise AssertionError('--reuse-fp32-param only support for `bf16`')
    if args.reuse_fp32_param and args.swap_optimizer:
        raise AssertionError('--swap-optimizer dose not support `--reuse-fp32-param`')


def _store_variables(args):
    """
    To bypass megatron validation, we store variables and restore them shortly afterward.
    """
    variable_dict = dict()
    variable_dict["variable_seq_lengths"] = args.variable_seq_lengths

    # to bypass megatron assertion of moe+spec
    variable_dict["spec"] = args.spec
    args.spec = None

    # Bypass megatron validation when pp == 2 and vpp is enabled.
    if args.pipeline_model_parallel_size == 2 and args.num_layers_per_virtual_pipeline_stage is not None:
        variable_dict["num_layers_per_virtual_pipeline_stage"] = args.num_layers_per_virtual_pipeline_stage
        variable_dict["overlap_p2p_comm"] = args.overlap_p2p_comm
        args.num_layers_per_virtual_pipeline_stage = None
        args.overlap_p2p_comm = None

    # Bypass megatron validation for gradient reduce in fp32
    if args.o2_gradient:
        args.accumulate_allreduce_grads_in_fp32 = True

    # Bypass megatron validation for MTP and CP
    if args.mtp_num_layers is not None and args.mtp_num_layers >= 1 and args.context_parallel_size > 1:
        variable_dict["mtp_num_layers"] = args.mtp_num_layers
        args.mtp_num_layers = 0

    # for 2 megatron
    if hasattr(args, "role") and args.role == "actor_rollout":
        variable_dict["true_world_size"] = args.world_size
        args.world_size = args.num_gpus_for_train

    if not args.data_path:
        args.data_path = True
    args.mock_data = 0

    return variable_dict


def _restore_variables(args, variable_dict):
    args.variable_seq_lengths = variable_dict["variable_seq_lengths"]

    # Bypass megatron validation when pp == 2 and vpp is enabled.
    if variable_dict.get("num_layers_per_virtual_pipeline_stage") and args.pipeline_model_parallel_size == 2:
        args.num_layers_per_virtual_pipeline_stage = variable_dict["num_layers_per_virtual_pipeline_stage"]
        args.overlap_p2p_comm = variable_dict["overlap_p2p_comm"]

    # Bypass megatron validation for gradient reduce in fp32
    if args.o2_gradient:
        args.accumulate_allreduce_grads_in_fp32 = False

    # Bypass megatron validation for MTP and CP
    if variable_dict.get("mtp_num_layers"):
        args.mtp_num_layers = variable_dict["mtp_num_layers"]     

    # for 2 megatron
    if hasattr(args, "role") and args.role == "actor_rollout":
        args.world_size = variable_dict["true_world_size"]
        if args.rank >= args.num_gpus_for_train:  # inference groups, different data_parallel_size
            args.data_parallel_size = args.num_gpus_for_infer // args.tensor_model_parallel_size
            args.micro_batch_size = args.num_samples_per_step  # for passing num_microbatches_calculator check

    # to bypass megatron assertion of moe+spec
    args.spec = variable_dict["spec"]


def _add_dummy_args(args):
    """
    For arguments in mindspeed-core which is currently unsupported in mindspeed-llm.
    """
    # reduce_recompute_for_last_chunk would be registered if recompute-in-advance is supported.
    args.adaptive_recompute_device_swap = False
    args.adaptive_recompute_device_size = -1
    args.adaptive_recompute_profiling_step = 10
    args.recompute_in_bubble = False
    args.use_nanopipe = False
    args.moe_without_activation = False
    args.ampipe_degree = 0
    args.hccl_group_buffer_adaptive = False
    args.moe_bmm_mc2 = False
    args.moe_hierarchical_alltoallv = False
    args.moe_experts_pipeline_degree = 0
    args.context_parallel_kv_cache_policy = None
    args.context_parallel_cache_interval = 0
    args.use_ulysses_allgather_kv = False
    args.use_pipe_experts = False
    args.megatron_cp_in_bnsd = False
    args.use_fusion_attn_v2 = False
    args.npu_deterministic = False
    args.prof_file = False
    args.moe_adaptive_recompute_activation = False
    args.op_cal_tflops = False
    args.profile_operator = False
    args.delay_grad_reduce = False
    args.delay_param_gather = False
    args.alibi_fusion_attn_type = None
    args.moe_adaptive_recompute_activation_scale = 2.0
    args.yarn_scaling_factor = 1.0
    args.moe_intermediate_size = args.moe_ffn_hidden_size


def _add_dummy_args_v2(args):
    """
    For arguments in feature_list which is currently unsupported in mindspeed-llm.
    """
    args.unaligned_linear = False
    args.embed_layernorm = False
    args.enable_share_memory = False
    args.return_document_ids = False
    args.attention_mask_on_cpu = False
    args.output_layer_slice_num = 1
    args.use_fused_mlp = False
    if not args.fp8:
        args.transformer_impl = 'local'
    args.ckpt_format = 'torch'
    args.use_dist_ckpt = False


def _validate_noop_layer(args):
    if isinstance(args.noop_layers, str):
        noop_layers = set()
        for x in args.noop_layers.split(','):
            if int(x) >= args.num_layers or int(x) < 0:
                raise AssertionError(f'each element in args.noop_layers({args.noop_layers}) should bigger or equal '
                                     f'to 0 and smaller than args.num_layers({args.num_layers})')
            noop_layers.add(int(x))
        args.noop_layers = noop_layers
        if args.num_layer_list:
            print_rank0_by_args("num layer list would be disabled when noop-layer is activated.")
            args.num_layer_list = None


def _valid_tp_2d_args(args):
    if args.tp_2d:
        if args.sequence_parallel:
            raise AssertionError('2d tp does not support sequence parallel')
        if args.use_fused_rmsnorm:
            raise AssertionError('2d tp does not support fused rmsnorm')
        if hasattr(args, "use_nanopipe") and args.use_nanopipe:
            raise AssertionError('tp-2d does not support nano-pipe')
        if hasattr(args, "ampipe_degree") and args.ampipe_degree > 1:
            raise AssertionError('tp-2d does not support ampipe')
        if hasattr(args, "context_parallel_algo") and args.context_parallel_algo not in ['megatron_cp_algo', 'ulysses_cp_algo']:
            raise AssertionError('tp-2d now only support megatron_cp_algo or ulysses_cp_algo')
        if hasattr(args, "use_ascend_coc") and args.use_ascend_coc:
            raise AssertionError('tp-2d does not support ascend coc')
        if args.tensor_model_parallel_size // args.tp_x != args.tp_y:
            raise AssertionError('need satisfy tp = tp_x * tp_y')
        if args.expert_model_parallel_size > 1:
            raise AssertionError('2d tp does not support moe')


def _valid_fa_div_args(args):
    if args.mla_fa_divide_qk:
        if args.context_parallel_size > 1:
            raise AssertionError('MLA FA currently not support CP>1.')
        if not args.reset_attention_mask:
            raise AssertionError('MLA FA currently only support TND.')


def _validate_vpp(args):
    """validate scenario that vpp is enabled when pp=2."""
    if args.pipeline_model_parallel_size != 2 or args.num_layers_per_virtual_pipeline_stage is None:
        return

    # VPP enabled when pp == 2, do check.
    num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
    if num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage != 0:
        raise ValueError('number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage')

    pp_stage_layers = args.num_layers / args.pipeline_model_parallel_size
    if args.num_layers_per_virtual_pipeline_stage and args.num_layers_per_virtual_pipeline_stage >= pp_stage_layers:
        raise ValueError("Num of layers in vpp stage should be less than pp stage, "
                         "please turn down args.num_layers_per_virtual_pipeline_stage.")

    args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                                                args.num_layers_per_virtual_pipeline_stage

    print_rank0_by_args(args, f'vpp_size would be {args.virtual_pipeline_model_parallel_size} since '
                        f'num_layers_per_virtual_pipeline_stage is {args.num_layers_per_virtual_pipeline_stage}')


def _validate_recompute_in_advance(args):
    args.reduce_recompute_for_last_chunk = False
    if args.recompute_in_advance:
        args.reduce_recompute_for_last_chunk = True
        if args.recompute_method == "uniform":
            raise AssertionError('recompute_in_advance does not support uniform recompute_method')
        if args.recompute_granularity == 'selective':
            raise AssertionError('recompute_in_advance does not support vanilla recompute_activations.')
        if not args.recompute_num_layers:
            raise AssertionError('recompute_num_layers must be configured when using recompute_in_advance')
        if args.pipeline_model_parallel_size <= 1 or args.num_layers_per_virtual_pipeline_stage != 1:
            raise AssertionError('recompute_in_advance only support pipelining with interleaving and vpp stage should be 1.')


def _validate_mlp_fusion(args):
    if args.use_fused_mlp:
        if not args.swiglu:
            raise AssertionError(
                'use_fused_mlp only support activation func swiglu')
        if not args.sequence_parallel and not args.lora_fusion:
            raise AssertionError(
                'use_fused_mlp only support sequence_parallel with cclora')


def _validate_long_rope(args):
    if args.rope_scaling_type == "longrope":
        if args.rope_scaling_original_max_position_embeddings is None:
            raise AssertionError('The parameter rope_scaling_original_max_position_embeddings should be set '
                                 'when use longrope.')
        if args.long_factor is None:
            raise AssertionError('The parameter long_factor should be set when use longrope.')
        else:
            args.long_factor = list(map(float, args.long_factor.split(',')))

        if args.short_factor is None:
            raise AssertionError('The parameter short_factor should be set when use longrope.')
        else:
            args.short_factor = list(map(float, args.short_factor.split(',')))

        if bool(args.short_mscale) ^ bool(args.long_mscale):
            raise AssertionError('The parameter short_mscale and long_mscale must be set at the same time')


def _validate_o2(args):
    if args.o2_gradient or args.o2_optimizer:
        print_rank0_by_args(args, "[WARNING] Using half precision gradient or optimizer would definitely impact precision.")
        if not args.bf16:
            raise ValueError("[ERROR] Should only use o2 feature during bf16 mix-precision training.")
    if args.o2_gradient:
        if args.gradient_accumulation_fusion:
            raise ValueError("gradient_accumulation_fusion only works with fp32 gradient.")


def _validate_fused_opts(args):
    if args.use_fused_rmsnorm:
        if args.normalization != "RMSNorm":
            raise AssertionError(
                '--use-fused-rmsnorm must enable with '
                '--normalization=RMSNorm, but got normalization'
                '={}.'.format(args.normalization))
    if args.use_fused_swiglu:
        if not args.swiglu:
            raise AssertionError(
                '--use-fused-swiglu must enable with --swiglu, '
                'but --swiglu={}.'.format(args.swiglu))
    if args.use_fused_rotary_pos_emb:
        if args.position_embedding_type != 'rope':
            raise AssertionError(
                '--use-fused-rotary-pos-emb must enable with'
                '--position-embedding-type=rope')


def validate_args_decorator(megatron_validate_args):
    @wraps(megatron_validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}

        variable_dict = _store_variables(args)
        if args.num_layer_list:
            _num_layer = args.num_layers
            args.num_layers = len(args.num_layer_list.split(','))
            megatron_validate_args(args, defaults)
            args.num_layers = _num_layer
            args.encoder_num_layers = args.num_layers
        else:
            megatron_validate_args(args, defaults)
        _restore_variables(args, variable_dict)

        args.use_ascend_mc2 = False
        args.use_legacy_models = not args.use_mcore_models


        _validate_o2(args)
        _validate_varlen_fa_args(args)
        _validate_cp_args(args)
        _validate_vpp(args)
        _validate_recompute_args(args)
        _validate_recompute_in_advance(args)
        _validate_create_attention_mask_in_dataloader(args)
        _validate_instruction_finetune(args)
        _validate_train_mode_conflict(args)
        _validate_position_embedding(args)
        _validate_inference_args(args)
        _validate_moe_args(args)
        _validate_mla(args)
        _validate_yarn(args)
        _validate_transformer_block_build_layers(args)
        _validate_evaluation_args(args)
        _validate_output_layer_slice_num(args)
        _validate_optimizer(args)
        _validate_long_rope(args)
        _validate_mlp_fusion(args)
        _validate_fused_opts(args)
        _validate_noop_layer(args)
        _valid_tp_2d_args(args)
        _valid_fa_div_args(args)
        _add_dummy_args(args)

        for feature in FEATURES_LIST:
            if (getattr(args, feature.feature_name, None) and feature.optimization_level > 0) or feature.optimization_level == 0:
                feature.pre_validate_args(args)
                feature.validate_args(args)
                feature.post_validate_args(args)

        
        from mindspeed_llm.training.utils import print_args
        print_args('MindSpeed-LLM Arguments', args)
        return args

    return wrapper


def validate_args_v2_decorator(megatron_validate_args):
    """A decorator for megatron arguments validation function."""

    @wraps(megatron_validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}
                      
        # make prev validation and copy some args.
        MindSpeedFeaturesManager.pre_validate_features_args(args)

        # make megatron args validation then restore args thar are copied.
        args = megatron_validate_args(args, defaults)

        # make post validation after megatron validation.
        MindSpeedFeaturesManager.post_validate_features_args(args=args)

        _add_dummy_args_v2(args)
        MindSpeedFeaturesManager.validate_features_args(args=args)

        from mindspeed_llm.training.utils import print_args
        print_args('MindSpeed-LLM Arguments', args)
        return args

    return wrapper

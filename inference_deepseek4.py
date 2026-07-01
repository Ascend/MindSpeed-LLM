# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from mindspeed_llm.tasks.inference.infer_deepseek4 import task_factory
from mindspeed_llm.tasks.inference.deepseek4_module import DeepSeek4MegatronModuleForCausalLM, DeepSeek4ModelInfer
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.mhc import get_mhc_spec
from mindspeed_llm.training.utils import auto_coverage


def model_provider(pre_process=True, post_process=True) -> DeepSeek4ModelInfer:
    """Build DeepSeek4 for inference."""
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.sequence_parallel and getattr(args, "use_kv_cache", False):
        raise ValueError('Use_kv_cache can not be true in sequence_parallel mode.')

    if args.num_layers_per_virtual_pipeline_stage is not None:
        raise ValueError('VPP is not supported for inference.')

    print_rank_0('building DeepSeek4 model ...')
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if getattr(args, "use_legacy_models", False):
        raise ValueError("DeepSeek4 model is only supported with Megatron Core!")

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    elif use_te:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
    else:
        transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

    mtp_block_spec = None
    if getattr(args, "mtp_num_layers", None) is not None:
        if getattr(args, "mtp_spec", None) is not None:
            mtp_layer_spec = import_module(args.mtp_spec)
        else:
            mtp_layer_spec = transformer_layer_spec
        mtp_block_spec = get_gpt_mtp_block_spec(config, mtp_layer_spec, use_transformer_engine=use_te)

    hc_head_spec = get_mhc_spec(getattr(args, "enable_mhc", False))

    model = DeepSeek4ModelInfer(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        hc_head_spec=hc_head_spec,
    )
    return model


@auto_coverage
def main():
    initialize_megatron(args_defaults={'no_load_rng': True, 'no_load_optim': True})

    args = get_args()

    model = DeepSeek4MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider, pretrained_model_name_or_path=args.load
    )

    task_factory(args, model)


if __name__ == "__main__":
    main()

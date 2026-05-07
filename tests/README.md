# MindSpeed-LLM 测试用例贡献说明

所有测试用例仅支持Megatron-Mcore模型结构。

## CI门禁看护列表

CI门禁用例看护仓库重点模型和基本特性，覆盖冒烟测试场景，PR合入前都须通过全量CI门禁用例测试。

<table>
    <tr>
        <th>Tests</th>
        <th>Module</th>
        <th>Features</th>
        <th>Scripts</th>
        <th>Acc.</th>
        <th>Throu.</th>
        <th>Mem.</th>
    </tr>
    <tr>
        <td rowspan="16">ST</td>
        <td rowspan="12">Pretrain</td>
        <td>TP，PP，VPP，distributed_optimizer，o2_gradient，o2_optimizer，重计算，enable_recompute_layers_per_pp_rank，FA_TND，use_fused_rotary_pos_emb</td>
        <td><a href="st/shell_scripts/llama2_tp2_pp4_vpp2_ptd.sh">llama2_tp2_pp4_vpp2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>swap_attention，recompute_activation_function，enable_recompute_layers_per_pp_rank，reuse_fp32_param</td>
        <td><a href="st/shell_scripts/llama2_tp2_pp4_vpp2_swap.sh">llama2_tp2_pp4_vpp2_swap.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>cp_ring，general_cp，分布式优化器，reuse_fp32_param，fused_rmsnorm，fused_swiglu，fused_rope，overlap_grad_reduce, overlap_param_gather</td>
        <td><a href="st/shell_scripts/pretrain_qwen3_8b_general_cp_ring.sh">pretrain_qwen3_8b_general_cp_ring.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>noop_layers， recompute_norm</td>
        <td><a href="st/shell_scripts/llama3_mcore_tp2_pp2_vpp2_noop_layer.sh">llama3_mcore_tp2_pp2_vpp2_noop_layer.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>cp_hybrid，gqa</td>
        <td><a href="st/shell_scripts/chatglm3_gqa_cp4.sh">chatglm3_gqa_cp4.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>mla_attention，moe_grouped_gemm，EP，allgather_dispatcher，moe_allgather_overlap_comm，use_fused_rotary_pos_emb，recompute_norm</td>
        <td><a href="st/shell_scripts/deepseek_v2_mcore_tp1_pp1_ep8.sh">deepseek_v2_mcore_tp1_pp1_ep8.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>n_group，seq_aux，gradient_accumulation_fusion，recompute_mtp_layer，recompute_mtp_norm</td>
        <td><a href="st/shell_scripts/deepseek_v3_mcore_tp1_pp2_ep4.sh">deepseek_v3_mcore_tp1_pp2_ep4.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>moe_alltoall_overlap_comm，moe-zero-memory，swap-attention，reuse_fp32_param，fused_rmsnorm，fused_swiglu</td>
        <td><a href="st/shell_scripts/deepseek_500b_tp1_pp2_ep2_cp2_overlap.sh">deepseek_500b_tp1_pp2_ep2_cp2_overlap.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>moe-fb-overlap, mtp-mem-efficient-logits, mla-mm-split, mla-fa-without-pad</td>
        <td><a href="st/shell_scripts/deepseek32_tp1_pp2_vpp1_ep4.sh">deepseek32_tp1_pp2_vpp1_ep4.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>EP，CP，num_experts，moe_router_topk，aux_loss，moe_allgather，group_query_attention，rotary_base</td>
        <td><a href="st/shell_scripts/mixtral_mcore_tp4_cp2_ep2_ptd.sh">mixtral_mcore_tp4_cp2_ep2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>mamba_cp_algo，分布式优化器，reuse_fp32_param，recompute-granularity，enable-recompute-layers-per-pp-rank</td>
        <td><a href="st/shell_scripts/mamba2_8b_tp4_pp1_cp2_recompute_4k_ptd.sh">mamba2_8b_tp4_pp1_cp2_recompute_4k_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>triton, topk-softmax-in-fp32, moe-router-pre-softmax</td>
        <td><a href="st/shell_scripts/qwen3_next_80b_4K_A3_ptd.sh">qwen3_next_80b_4K_A3_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2">Finetune</td>
        <td>CCLoRA, QLoRA</td>
        <td><a href="st/shell_scripts/tune_llama2_tp1_pp1_qlora_ptd.sh">tune_llama2_tp1_pp1_qlora_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>LoRA, lora-fusion, llama3-rope, no-pad-to-seq-lengths, enable-hf2mg-convert, auto_data_process</td>
        <td><a href="st/shell_scripts/tune_llama3_8b_lora_tp1pp8.sh">tune_llama3_8b_lora_tp1pp8.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">DPO</td>
        <td>is_pairwise_dataset, cyclic</td>
        <td><a href="st/shell_scripts/dpo_llama2_tp1_pp1_cyclic_pairwise.sh">dpo_llama2_tp1_pp1_cyclic_pairwise.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2">FSDP</td>
        <td>fsdp pretrain</td>
        <td><a href="st/shell_scripts/pretrain_qwen3_8b_4k_fsdp2.sh">pretrain_qwen3_8b_4k_fsdp2.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>fsdp sft</td>
        <td><a href="st/shell_scripts/tune_gpt_oss_20b_a3b_4k_fsdp2.sh">tune_gpt_oss_20b_a3b_4k_fsdp2.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="4">UT</td>
        <td>Inference</td>
        <td>greedy_search, lora_inference, deterministic_computation</td>
        <td><a href="ut/inference/test_inference.py">test_inference.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Evaluation</td>
        <td>mmlu, prompt_mmlu, qwen2_mmlu, agieval, bbh</td>
        <td><a href="ut/evaluation/test_evaluate.py">test_evaluate.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="1">Checkpoint</td>
        <td>hf2mcore, mcore2hf, TP, PP, EP, DPP, VPP, moe, noop_layers, lora, deepseek2, deepseek2_lite, llama2, llama3, qwen2</td>
        <td><a href="ut/checkpoint/test_checkpoint.py">test_checkpoint.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="1">ProcessData</td>
        <td>pretrain_data_alpaca, pretrain_merge_datasets, instruction_data_alpaca, instruction_merge_datasets</td>
        <td><a href="ut/process_data/test_preprocess_data.py">test_preprocess_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>

</table>

## Pipeline看护列表

Pipeline用例看护全量覆盖仓库所有模型和所有特性，每天夜里拉起运行，次日输出测试报告。

<table>
    <tr>
        <th>Tests</th>
        <th>Module</th>
        <th>Features</th>
        <th>Scripts</th>
        <th>Acc.</th>
        <th>Throu.</th>
        <th>Mem.</th>
    </tr>
    <tr>
        <td rowspan="31">ST</td>
        <td rowspan="1">baichuan2-13B</td>
        <td>baichuan2_13b, no-gradient-accumulation-fusion</td>
        <td><a href="pipeline/st/baichuan2-13B/baichuan2_13b_tp8_pp1_mcore.sh">baichuan2_13b_tp8_pp1_mcore.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">chatglm3-6B</td>
        <td>chatglm3-6B, use-glm-rope, overlap-grad-reduce, overlap-param-gather</td>
        <td><a href="pipeline/st/chatglm3-6B/chatglm3_tp1_pp2_rope.sh">chatglm3_tp1_pp2_rope.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2">deepseek</td>
        <td>deepseekv3, dualpipev, mla-up-proj-tp-overlap, moe-fb-overlap</td>
        <td><a href="pipeline/st/deepseek/deepseek_v3_mcore_tp2_pp2_ep2_dualpipev_fb.sh">deepseek_v3_mcore_tp2_pp2_ep2_dualpipev_fb.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>deepseekv2, moe-grouped-gemm, moe-permutation-async-comm, first-k-dense-replace</td>
        <td><a href="pipeline/st/deepseek/deepseek2_tp1_pp1_mcore_moe.sh">deepseek2_tp1_pp1_mcore_moe.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">glm4</td>
        <td>glm4, overlap_grad_reduce, overlap_param_gather, distributed_optimizer, GQA, GLM-rope</td>
        <td><a href="pipeline/st/glm4/glm4_9b_8k_tp2_pp2_ptd.sh">glm4_9b_8k_tp2_pp2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">gpt4</td>
        <td>gpt4, distributed_optimizer, overlap_grad_reduce, overlap_param_gather</td>
        <td><a href="pipeline/st/gpt4/gpt4_mcore_tp4_cp2_32k_moe_drop.sh">gpt4_mcore_tp4_cp2_32k_moe_drop.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">grok1</td>
        <td>grok1, distributed_optimizer, reuse_fp32_param</td>
        <td><a href="pipeline/st/grok1/grok1_40b_tp4_ep2_ptd.sh">grok1_40b_tp4_ep2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2">high_availability</td>
        <td>high_availability error_dump</td>
        <td><a href="pipeline/st/high_availability/high_availability_error_dump_ptd.sh">high_availability_error_dump_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>high_availability uce_error</td>
        <td><a href="pipeline/st/high_availability/high_availability_uce_error_ptd.sh">high_availability_uce_error_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">hunyuan</td>
        <td>hunyuan, distributed_optimizer, share_kvstates</td>
        <td><a href="pipeline/st/hunyuan/tune_hunyuanLarge_389b_tp1_pp1_ep8_ptd.sh">tune_hunyuanLarge_389b_tp1_pp1_ep8_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">interlm3</td>
        <td>internlm3, distributed_optimizer, fused_ring_attention_update</td>
        <td><a href="pipeline/st/interlm3/internlm3_8b_tp1_pp4_cp2_ptd.sh">internlm3_8b_tp1_pp4_cp2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="9">llama2</td>
        <td>llama2, distributed_optimizer, overlap_grad_reduce, gloo</td>
        <td><a href="pipeline/st/llama2/llama2_tp1_pp8_patch_gloo_ptd.sh">llama2_tp1_pp8_patch_gloo_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>llama2, TP-2D, ring_cp, distributed_optimizer, overlap_grad_reduce</td>
        <td><a href="pipeline/st/llama2/llama2_tp4cp2pp1_tp2d_tpx2tpy2_ringcp.sh">llama2_tp4cp2pp1_tp2d_tpx2tpy2_ringcp.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>llama2, TP-2D, ulysses_cp, distributed_optimizer, overlap_grad_reduce</td>
        <td><a href="pipeline/st/llama2/llama2_tp4cp2pp1_tp2d_tpx2tpy2_ulysses.sh">llama2_tp4cp2pp1_tp2d_tpx2tpy2_ulysses.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>llama2, TP-2D, VPP, distributed_optimizer, overlap_grad_reduce</td>
        <td><a href="pipeline/st/llama2/llama2_tp4pp2vpp2_tp2d_tpx2tpy2.sh">llama2_tp4pp2vpp2_tp2d_tpx2tpy2.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>llama2, distributed_optimizer, overlap_grad_reduce, ascend_coc</td>
        <td><a href="pipeline/st/llama2/llama2_tp8_pp1_coc_ptd.sh">llama2_tp8_pp1_coc_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>llama2, LoRA, lora-fusion, SFT</td>
        <td><a href="pipeline/st/llama2/tune_llama2_tp1_pp1_lora_ptd.sh">tune_llama2_tp1_pp1_lora_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>llama2, LU-LoRA, lora-fusion, SFT</td>
        <td><a href="pipeline/st/llama2/tune_llama2_tp1_pp1_lu_lora_ptd.sh">tune_llama2_tp1_pp1_lu_lora_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>llama2, VPP, distributed_optimizer, overlap_grad_reduce, SFT</td>
        <td><a href="pipeline/st/llama2/tune_llama2_tp2_pp4_vpp2_mcore_full.sh">tune_llama2_tp2_pp4_vpp2_mcore_full.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>llama2, CP, megatron_cp_algo, use-fused-ring-attention-update</td>
        <td><a href="pipeline/st/llama2/llama2_tp2_cp4_general_double_ring.sh">llama2_tp2_cp4_general_double_ring.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">llama3</td>
        <td>llama3, VPP, GQA, recompute, manual_gc</td>
        <td><a href="pipeline/st/llama3/llama3_tp2_pp2_vpp1.sh">llama3_tp2_pp2_vpp1.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">longcat-flash</td>
        <td>longcat-flash, ETP, MLA, distributed_optimizer</td>
        <td><a href="pipeline/st/longcat-flash/longcat_flash_560b_tp2pp1ep2etp1.sh">longcat_flash_560b_tp2pp1ep2etp1.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">mamba2</td>
        <td>mamba2, mamba_cp_algo, distributed_optimizer, reuse_fp32_param, overlap_grad_reduce, overlap_param_gather</td>
        <td><a href="pipeline/st/mamba2/mamba2_2.7b_tp1_pp1.sh">mamba2_2.7b_tp1_pp1.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">phi35-moe</td>
        <td>phi35-moe, distributed_optimizer, overlap_grad_reduce, overlap_param_gather, longrope</td>
        <td><a href="pipeline/st/phi35-moe/phi35_moe_tp1_pp8_mcore.sh">phi35_moe_tp1_pp8_mcore.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2">qwen25</td>
        <td>qwen2-moe, distributed_optimizer, overlap_grad_reduce, overlap_param_gather, profile</td>
        <td><a href="pipeline/st/qwen25/qwen2_moe_tp1_pp2_ep2_cp2_32k.sh">qwen2_moe_tp1_pp2_ep2_cp2_32k.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>qwen25, distributed_optimizer, overlap_grad_reduce, SFT, neat_pack, padded_samples</td>
        <td><a href="pipeline/st/qwen25/tune_qwen25_0point5b_tp1_pp1_pack.sh">tune_qwen25_0point5b_tp1_pp1_pack.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2">qwen3-30b</td>
        <td>qwen3-30b DPO, distributed_optimizer, recompute, DPO</td>
        <td><a href="pipeline/st/qwen3-30b/dpo_qwen3_30b_a3b_16K_A3_ptd_tp2pp4.sh">dpo_qwen3_30b_a3b_16K_A3_ptd_tp2pp4.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
        <tr>
        <td>qwen3-30b layer2, async-save</td>
        <td><a href="pipeline/st/qwen3-30b/qwen3-30b-layer2-dist.sh">qwen3-30b-layer2-dist.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">qwen3-30b</td>
        <td>qwen3-30b w4a16-mxfp4</td>
        <td><a href="pipeline/st/qwen3-30b/tune_qwen3_30b_a3b_4K_full_ptd_tp4_pp2_ep1.sh">tune_qwen3_30b_a3b_4K_full_ptd_tp4_pp2_ep1.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">seed-oss</td>
        <td>seed-oss, distributed_optimizer, calculate-per-token-loss</td>
        <td><a href="pipeline/st/seed-oss/seed_oss_36b_tp2pp2.sh">seed_oss_36b_tp2pp2.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1">posttrain</td>
        <td>layerwise_disaggregated_training</td>
        <td><a href="pipeline/st/layerwise_disaggregated_training/tune_qwen25_7b_tp1pp4_layerwise_disaggregated.sh">tune_qwen25_7b_tp1pp4_layerwise_disaggregated.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="20">UT</td>
        <td rowspan="3">checkpoint</td>
        <td>test_checkpoint_param</td>
        <td><a href="pipeline/ut/checkpoint/test_checkpoint_param.py">test_checkpoint_param.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_checkpoint_v2</td>
        <td><a href="pipeline/ut/checkpoint/test_checkpoint_v2.py">test_checkpoint_v2.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_checkpoint</td>
        <td ><a href="pipeline/ut/checkpoint/test_checkpoint.py">test_checkpoint.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="3">context_parallel</td>
        <td>test_hybrid_context_parallel</td>
        <td><a href="pipeline/ut/context_parallel/test_hybrid_context_parallel.py">test_hybrid_context_parallel.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_ringattn_context_parallel</td>
        <td><a href="pipeline/ut/context_parallel/test_ringattn_context_parallel.py">test_ringattn_context_parallel.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_ulysses_context_parallel</td>
        <td><a href="pipeline/ut/context_parallel/test_ulysses_context_parallel.py">test_ulysses_context_parallel.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="6">elastic_training</td>
        <td>test_elastic_training_common</td>
        <td><a href="pipeline/ut/elastic_training/test_elastic_training_common.py">test_elastic_training_common.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_elastic_training_register</td>
        <td><a href="pipeline/ut/elastic_training/test_elastic_training_register.py">test_elastic_training_register.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_elastic_training_repair</td>
        <td><a href="pipeline/ut/elastic_training/test_elastic_training_repair.py">test_elastic_training_repair.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_elastic_training_rollback</td>
        <td><a href="pipeline/ut/elastic_training/test_elastic_training_rollback.py">test_elastic_training_rollback.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_elastic_training_scale_in_rebuild</td>
        <td><a href="pipeline/ut/elastic_training/test_elastic_training_scale_in_rebuild.py">test_elastic_training_scale_in_rebuild.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_elastic_training_scale_out_rebuild</td>
        <td><a href="pipeline/ut/elastic_training/test_elastic_training_scale_out_rebuild.py">test_elastic_training_scale_out_rebuild.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="1">evaluation</td>
        <td>mmlu, cmmlu, humaneval, ceval, boolq, gsm8k, agieval, bbh, needlebench</td>
        <td><a href="pipeline/ut/evaluation/test_evaluate.py">test_evaluate.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="1">inference</td>
        <td>greedy_search, do_sample_search, beam_search, chat</td>
        <td><a href="pipeline/ut/inference/test_inference.py">test_inference.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="3">model_module</td>
        <td>test_attention, DotProductAttention, FlashSelfAttention, alibi</td>
        <td><a href="pipeline/ut/model_module/test_attention.py">test_attention.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_rotary_pos_embedding</td>
        <td><a href="pipeline/ut/model_module/test_rotary_pos_embedding.py">test_rotary_pos_embedding.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_topk_router, sparsemixer_topk</td>
        <td><a href="pipeline/ut/model_module/test_topk_router.py">test_topk_router.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="3">process_data</td>
        <td>test_process_instruction_data, alpaca, sharegpt, openai, merge_datasets, multi_handler, template</td>
        <td><a href="pipeline/ut/process_data/test_process_instruction_data.py">test_process_instruction_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_process_pairwise_data, alpaca, sharegpt</td>
        <td><a href="pipeline/ut/process_data/test_process_pairwise_data.py">test_process_pairwise_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>test_process_pretrain_data, merge_datasets, GPTSentencePieceTokenizer</td>
        <td><a href="pipeline/ut/process_data/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
</table>

## DT覆盖率看护

### 覆盖率分析脚本

执行`tests/run_coverage.sh`脚本时，可添加UT, ST, PIPELINE, all等运行参数

```bash
cd MindSpeed-LLM
bash tests/run_coverage.sh UT       # 分析UT用例覆盖率
bash tests/run_coverage.sh ST       # 分析ST用例覆盖率
bash tests/run_coverage.sh PIPELINE # 分析PIPELINE用例覆盖率
bash tests/run_coverage.sh all      # 分析UT,ST,PIPELINE所有用例覆盖率
```

设置脚本中的`branch` 的值为 `False`时只分析行覆盖率，将`branch` 的值改为 `True` 则可以测试分支覆盖率。

### 覆盖率报告

在NPU机器运行 `run_coverage.sh` 脚本后，项目目录下将生成 `COVERAGE` 文件夹，其中 `COVERAGE/logs`文件夹保存了详细的用例执行情况，`COVERAGE/report`文件夹保存了仓库用例覆盖率报告。

`COVERAGE/report/htmlcov.tgz`包含了仓库所有文件的详细覆盖率信息，将该文件复制到本地电脑进行解压，然后在浏览器中打开 `htmlcov/index.html` 文件即可进行查看。

## 开发流程

### 1.权重和数据集配置

用例所需使用的权重、Tokenizer、数据集文件，请按照以下要求存放在蓝区服务器的/data目录下，并按照要求在[蓝区资源清单](resource_record.md)中进行登记，否则不予上库！

注意：

- /data/ci目录下只保存用例相关的文件，不要引入其他无关文件
- 为了节省蓝区空间并提高运行效率，请尽量复用原有权重数据，如需上传权重数据，请将权重层数设置为最小
- 模型名称需与huggingface保持一致，严禁省略或自定义

数据路径和命名规则：

- hf权重和词表路径：/data/ci/models/模型名称/hf/权重或词表文件
- mg权重路径：/data/ci/models/模型名称/mg/模型名称_切分方式
- 原数据集：/data/ci/datasets/origin/数据集名称
- 处理后的数据集：/data/ci/datasets/processed/数据集名称
- 评估数据集：/data/ci/datasets/eva_dataset/数据集名称
- 缓存文件夹：/data/ci/cache/缓存文件，用例执行结束前请调用shutil.rmtree(dir_path)删除缓存文件

### 2.本地验证

用例编写后，先确保用例在本地运行无误，然后用蓝区备用服务器生成基线，用例和基线一同上仓

### 3.用例登记

- 测试用例信息登记
为了方便后续对用例进行维护，需要对用例的作者、上仓日期、简要描述以及其他信息进行标注

ST用例需在运行脚本开始时标注以下信息

```bash
#=============================================
# Author: xxx
# Date: xxxx-xx-xx
# Description：Model or feature covered by the testcase
# Remarks: Instructions for the checkpoint, datasets and tokenizer or other more information
#=============================================

```

UT用例在用例执行函数内标注以下信息

```python
def test_featureA():
    '''
    Author: xxx
    Date: xxxx-xx-xx
    Description：Model or feature covered by the testcase
    Remarks: Instructions for the checkpoint, datasets and tokenizer or other more information
    '''
    ......
```

- 用例看护特性列表登记
在`/tests/README.md`文件中登记测试用例所看护的模型，特性信息

- 蓝区资源登记
在`/tests/resource_record.md`文件中登记用例使用的权重、词表、数据集信息

### 4.用例上仓

特性须与看护用例一同上仓，只有业务代码而无看护用例的PR不予合入！

## 开发规则

测试用例全部放置在`tests`目录下，具体层级如下：

- `tests/st/`目录下维护CI门禁会拉起的ST用例
- `tests/ut/`目录下维护CI门禁会拉起的UT用例
- `tests/pipeline/st`目录下维护每日PIPELINE流水线会拉起的ST用例
- `tests/pipeline/ut`目录下维护每日PIPELINE流水线会拉起的UT用例

### ST

① 贡献脚本用例请放置于 `st/shell_scripts` 文件夹下，命名规则为 **{模型名}_{切分策略}** 或者 **{模型名}_{特性名称}**， 如 `llama2_tp2_pp4_vpp2_ptd.sh`，请贡献者严格对齐；

② 注意脚本用例中不需要单独重定向log，日志收集工作已在 `st_run.sh` 中统一管理；

③ 标杆数据请放置于 `st/baseline_results` 文件夹下，**命名保证完全与 shell 脚本对齐**，否则自动化脚本执行将扫描不到；

④ 获取标杆数据：通过门禁任务执行获得首次数据，并将结果保存至本地 log 或者 txt 文件中，后通过本地执行 `st/st_utils/common.py` 中的 `transfer_logs_as_json` 函数进行提取，最后再连同用例脚本上仓即可；

⑤ 在贡献时候需要考虑最终校验的具体指标，精度(Acc.)、性能(Throu.)、显存(Mem.)，在对应指标空白处填上 `Y`，如无校验的保留空白即可。

### UT

① 建议所有 UT 用例通过分布式 `pytest` 来拉起，即继承 tests/common.py 文件下的 `DistributedTest`，指定 `world_size`，具体参照已有用例即可；

② 建议按照功能特性进行文件夹命名区分，至多不超过两层目录，所有用例以 `test` 作为命名前缀；

③ 新增用例可以在原有用例基础上做 `test_xxx` 的补充，尽量保证测试功能的集成性；对于存在 .json 文件的用例，贡献时在 .json 中加入 `test_xxx` 配置，然后在 .py 中通过 `@pytest.mark.parametrize` 传入参数、构造用例，**请注意 .json 中的 key 值命名需与 .py 中的 test_xxx 保持统一**；

④ 在贡献时候需要考虑最终校验的具体指标，精度(Acc.)、性能(Throu.)、显存(Mem.)，在对应指标空白处填上 `Y`，如无校验的保留空白即可。

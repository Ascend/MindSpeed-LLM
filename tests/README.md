## MindSpeed-LLM 测试用例贡献说明

所有测试用例仅支持Megatron-Mcore模型结构。

### 门禁看护列表
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
        <td rowspan="17">ST</td>
        <td rowspan="14">Pretrain</td>
        <td>TP，PP，VPP，distributed_optimizer，o2_gradient，o2_optimizer，重计算，enable_recompute_layers_per_pp_rank，FA_TND，use_fused_rotary_pos_emb</td>
        <td><a href="st/shell_scripts/llama2_tp2_pp4_vpp2_ptd.sh">llama2_tp2_pp4_vpp2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>TP2D, TP, PP, VPP, distributed_optimizer, fused_swiglu</td>
        <td><a href="st/shell_scripts/llama2_tp4pp2vpp2_tp2d_tpx2tpy2.sh">llama2_tp4pp2vpp2_tp2d_tpx2tpy2.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>cp_ring，分布式优化器，reuse_fp32_param，recompute_activation_function，fused_rmsnorm，fused_swiglu，fused_rope，overlap_grad_reduce, overlap_param_gather</td>
        <td><a href="st/shell_scripts/llama2_tp2_cp4_mem_recompute.sh">llama2_tp2_cp4_mem_recompute.sh</a></td>
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
        <td>cp_ring，general_cp，double_ring，分布式优化器，reuse_fp32_param，recompute_activation_function，fused_rmsnorm，fused_swiglu，fused_rope，overlap_grad_reduce, overlap_param_gather</td>
        <td><a href="st/shell_scripts/llama2_tp2_cp4_general_double_ring.sh">llama2_tp2_cp4_general_double_ring.sh</a></td>
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
        <td>pp2vpp，recompute_in_advance，matmul_add</td>
        <td><a href="pipeline/llama3/llama3_tp2_pp2_vpp1.sh">llama3_tp2_pp2_vpp1.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>cp_hybrid，gqa</td>
        <td><a href="st/shell_scripts/chatglm3_gqa_cp8.sh">chatglm3_gqa_cp8.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>swap_attention，recompute_activation_function，enable_recompute_layers_per_pp_rank，reuse_fp32_param</td>
        <td><a href="tests/st/shell_scripts/llama2_tp2_pp4_vpp2_swap.sh">llama2_tp2_pp4_vpp2_swap.sh</a></td>
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
        <td>mla_attention，moe_grouped_gemm，EP，allgather_dispatcher，moe_allgather_overlap_comm，use_fused_rotary_pos_emb，recompute_norm</td>
        <td><a href="st/shell_scripts/deepseek_v2_mcore_tp1_pp1_ep8.sh">deepseek_v2_mcore_tp1_pp1_ep8.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>MOE,PP,EP,Drop,DPP</td>
        <td><a href="st/shell_scripts/mixtral_tp1_pp4_ep2_drop_dpp.sh">mixtral_tp1_pp4_ep2_drop_dpp.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>shared_experts shared_expert_gate</td>
        <td><a href="st/shell_scripts/qwen2_moe_tp1_pp2_ep2_cp2_32k_ptd.sh">qwen2_moe_tp1_pp2_ep2_cp2_32k_ptd.sh</a></td>
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
        <td rowspan="5">UT</td>
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
        <td rowspan="2">Checkpoint</td>
        <td>hf2mcore, mcore2hf, TP, PP, EP, DPP, VPP, moe, noop_layers, lora</td>
        <td rowspan="2"><a href="ut/checkpoint/test_checkpoint.py">test_checkpoint.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>deepseek2, deepseek2_lite, llama2, llama3, qwen2</td>
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

### Pipeline 二级流水看护列表
<table>
    <tr>
        <th>Model</th>
        <th>Module</th>
        <th>Test Case</th>
        <th>Accuracy</th>
        <th>Throughput</th>
        <th>Memory</th>
    </tr>
    <tr>
        <td rowspan="3"><a href="pipeline/context_parallel">CP</td>
        <td>hybrid</td>
        <td><a href="pipeline/context_parallel/test_hybrid_context_parallel.py">test_hybrid_context_parallel.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ring_attn</td>
        <td><a href="pipeline/context_parallel/test_ringattn_context_parallel.py">test_ringattn_context_parallel.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ulysses</td>
        <td><a href="pipeline/context_parallel/test_ulysses_context_parallel.py"> test_ulysses_context_parallel.py </a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="3"><a href="pipeline/model_module">ModelModule</td>
        <td>rope</td>
        <td><a href="pipeline/model_module/test_rotary_pos_embedding.py">test_rotary_pos_embedding.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>moe</td>
        <td><a href="pipeline/model_module/test_topk_router.py">test_topk_router.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>transformer_attention, alibi</td>
        <td><a href="pipeline/model_module/test_attention.py">test_attention.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><a href="pipeline/common">Checkpoint</td>
        <td>hf2mcore, mcore2hf, TP, PP, EP, DPP, VPP, moe, noop_layers, lora</td>
        <td rowspan="2"><a href="pipeline/common/test_checkpoint.py">test_checkpoint.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>mixtral, deepseek2, deepseek2_lite, gemma2, llama3, qwen2, llama2</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><a href="pipeline/common">Inference</td>
        <td>greedy_search, deterministic_computation, chatglm3_inference, baichuan2_inference</td>
        <td><a href="pipeline/common/test_inference.py">test_inference.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td><a href="pipeline/common">Evaluation</td>
        <td>humaneval</td>
        <td><a href="pipeline/common/test_evaluate.py">test_evaluate.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
	<tr>
        <td rowspan="5"><a href="pipeline/common">ProcessData</td>
        <td>instruction_data_alpaca,
        instruction_data_alpaca_history,
        instruction_data_sharegpt,
        instruction_data_openai,</td>
        <td><a href="ut/process_data/test_process_instruction_data_lf.py">test_process_instruction_data_lf.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
	<tr>
        <td>instruction_data_handler</td>
        <td><a href="ut/process_data/test_process_instruction_pack_data.py">test_process_instruction_pack_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>pairwise_data_alpaca, pairwise_data_sharegpt</td>
        <td><a href="ut/process_data/test_process_pairwise_data_lf.py">test_process_pairwise_data_lf.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>GPTSentencePieceTokenizer</td>
        <td><a href="coverage/process_data/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>reasoning_template</td>
        <td><a href="coverage/process_data/test_process_instruction_data.py">test_process_instruction_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><a href="pipeline/baichuan2-13B">Baichuan2-13B</a></td>
        <td>data_process</td>
        <td><a href="pipeline/baichuan2-13B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
        <td>pretrain</td>
        <td><a href="pipeline/baichuan2-13B/baichuan2_13b_tp8_pp1_mcore.sh">baichuan2_13b_tp8_pp1_mcore.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="pipeline/chatglm3-6B">Chatglm3-6B</a></td>
        <td>data_process</td>
        <td><a href="pipeline/chatglm3-6B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>pretrain</td>
        <td><a href="pipeline/chatglm3-6B/chatglm3_tp1_pp2_rope.sh">chatglm3_tp1_pp2_rope.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/bloom-7B">Bloom-7B</a></td>
        <td>data_process</td>
        <td><a href="pipeline/bloom-7B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><a href="pipeline/gemma-7B">Gemma-7B</a></td>
        <td>data_process</td>
        <td><a href="pipeline/gemma-7B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="1">pretrain</td>
        <td><a href="pipeline/gemma-7B/gemma2_tp8_pp1_ptd.sh">gemma2_tp8_pp1_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/glm4">Glm4</a></td>
        <td>no-bias-swiglu-fusion</td>
        <td><a href="pipeline/glm4/glm4_9b_8k_tp2_pp2_ptd.sh">glm4_9b_8k_tp2_pp2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/grok1">Grok1</a></td>
        <td>embedding-multiplier-scale, output-multiplier-scale, input-jitter</td>
        <td><a href="pipeline/grok1/grok1_40b_tp4_ep2_ptd.sh">grok1_40b_tp4_ep2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/hunyuan">Hunyuan</a></td>
        <td>cla-share-factor, cut-max-seqlen, share-kvstates, pad-to-multiple-of, moe-revert-type-after-topk, scale-depth</td>
        <td><a href="pipeline/hunyuan/tune_hunyuanLarge_389b_tp1_pp1_ep8_ptd.sh">tune_hunyuanLarge_389b_tp1_pp1_ep8_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/interlm3">Interlm3</a></td>
        <td>skip-bias-add, dynamic-factor, distributed-timeout-minutes, exit-on-missing-checkpoint</td>
        <td><a href="pipeline/interlm3/internlm3_8b_tp1_pp4_cp2_ptd.sh">internlm3_8b_tp1_pp4_cp2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2"><a href="pipeline/qwen15-7B">Qwen15-7B</a></td>
        <td>data_process</td>
        <td><a href="pipeline/qwen15-7B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>evaluation</td>
        <td><a href="pipeline/qwen15-7B/test_evaluation.py">test_evaluation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/qwen25">Qwen25</a></td>
        <td>sparse-mode, padded-samples</td>
        <td><a href="pipeline/qwen25/tune_qwen25_0point5b_tp1_pp1_pack.sh">tune_qwen25_0point5b_tp1_pp1_pack.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/gpt4">GPT4</a></td>
        <td>pretrain</td>
        <td><a href="pipeline/gpt4/gpt4_mcore_tp4_cp2_32k_moe_drop.sh">gpt4_mcore_tp4_cp2_32k_moe_drop.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/deepseek">Deepseek</a></td>
        <td>pretrain</td>
        <td><a href="pipeline/deepseek/deepseek2_tp1_pp1_mcore_moe.sh">deepseek2_tp1_pp1_mcore_moe.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/deepseek_lite">Deepseek_lite</a></td>
        <td>convert_ckpt</td>
        <td><a href="pipeline/deepseek_lite/test_deepseek_lite_ckpt_optim.py">test_deepseek_lite_ckpt_optim.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><a href="pipeline/mixtral">Mixtral</a></td>
        <td>pretrain</td>
        <td><a href="pipeline/mixtral/mixtral_tp1_pp4_ep2_drop_mcore.sh">mixtral_tp1_pp4_ep2_drop_mcore.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>LoRA</td>
        <td><a href="pipeline/mixtral/tune_mixtral_tp2_pp2_lora_ptd.sh">tune_mixtral_tp2_pp2_lora_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="7"><a href="pipeline/llama2">Llama2</a></td>
        <td>LoRA</td>
        <td><a href="pipeline/llama2/tune_llama2_tp1_pp1_lora_ptd.sh">tune_llama2_tp1_pp1_lora_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>FullSFT</td>
        <td><a href="pipeline/llama2/tune_llama2_tp2_pp4_vpp2_mcore_full.sh">tune_llama2_tp2_pp4_vpp2_mcore_full.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="4">pretrain</td>
        <td><a href="pipeline/llama2/llama2_tp8_pp1_coc_ptd.sh">llama2_tp8_pp1_coc_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td><a href="pipeline/llama2/llama2_tp4cp2pp1_tp2d_tpx2tpy2_ulysses.sh">llama2_tp4cp2pp1_tp2d_tpx2tpy2_ulysses.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td><a href="pipeline/llama2_tp4cp2pp1_tp2d_tpx2tpy2_ringcp.sh">llama2_tp4cp2pp1_tp2d_tpx2tpy2_ringcp.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td><a href="pipeline/llama2/llama2_tp1_pp8_patch_gloo_ptd.sh">llama2_tp1_pp8_patch_gloo_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>convert_ckpt</td>
        <td><a href="pipeline/llama2/test_llama2_7b_ckpt_optim.py">test_llama2_7b_ckpt_optim.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="1"><a href="pipeline/minicpm">Minicpm</a></td>
        <td>dim-model-base, scale-emb</td>
        <td><a href="pipeline/minicpm/minicpm_2b_tp1_pp1.sh">minicpm_2b_tp1_pp1.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="5"><a href="pipeline/phi35-moe">Phi-3.5-MoE-instruct</a></td>
        <td>pretrain</td>
        <td><a href="pipeline/phi35-moe/phi35_moe_tp1_pp8_mcore.sh">phi35_moe_tp1_pp8_mcore.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
	<tr>
        <td>data_process</td>
        <td><a href="pipeline/phi35-moe/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ckpt_hf2mg</td>
        <td><a href="pipeline/phi35-moe/test_ckpt_hf2mg.py">test_ckpt_hf2mg.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>inference</td>
        <td><a href="pipeline/phi35-moe/test_generation.py">test_generation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>evaluation</td>
        <td><a href="pipeline/phi35-moe/test_evaluation.py">test_evaluation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2"><a href="pipeline/high_availability">HighAvailability</a></td>
        <td>UCE ERROR</td>
        <td><a href="pipeline/high_availability/high_availability_uce_error_ptd.sh">high_availability_uce_error_ptd.sh</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ERROR DUMP</td>
        <td><a href="pipeline/high_availability/high_availability_error_dump_ptd.sh">high_availability_error_dump_ptd.sh</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
</table>

### DT覆盖率看护
在NPU机器运行 `run_coverage.sh` 脚本，运行目录将生成 `htmlcov` 文件夹，将该文件夹复制到本地电脑，在浏览器中打开 `htmlcov/index.html` 文件，可以看到覆盖率信息。

脚本中 `branch` 的值改为 `True` ，可以测试分支覆盖率。


### 开发规则

#### ST

① 贡献脚本用例请放置于 `st/shell_scripts` 文件夹下，命名规则为 **{模型名}_{切分策略}** 或者 **{模型名}_{特性名称}**， 如 `llama2_tp2_pp4_vpp2_ptd.sh`，请贡献者严格对齐；

② 注意脚本用例中不需要单独重定向log，日志收集工作已在 `st_run.sh` 中统一管理；

③ 标杆数据请放置于 `st/baseline_results` 文件夹下，**命名保证完全与 shell 脚本对齐**，否则自动化脚本执行将扫描不到；

④ 获取标杆数据：通过门禁任务执行获得首次数据，并将结果保存至本地 log 或者 txt 文件中，后通过本地执行 `st/st_utils/common.py` 中的 `transfer_logs_as_json` 函数进行提取，最后再连同用例脚本上仓即可；

⑤ 在贡献时候需要考虑最终校验的具体指标，精度(Acc.)、性能(Throu.)、显存(Mem.)，在对应指标空白处填上 `Y`，如无校验的保留空白即可。


#### UT

① 建议所有 UT 用例通过分布式 `pytest` 来拉起，即继承 tests/common.py 文件下的 `DistributedTest`，指定 `world_size`，具体参照已有用例即可；

② 建议按照功能特性进行文件夹命名区分，至多不超过两层目录，所有用例以 `test` 作为命名前缀；

③ 新增用例可以在原有用例基础上做 `test_xxx` 的补充，尽量保证测试功能的集成性；对于存在 .json 文件的用例，贡献时在 .json 中加入 `test_xxx` 配置，然后在 .py 中通过 `@pytest.mark.parametrize` 传入参数、构造用例，**请注意 .json 中的 key 值命名需与 .py 中的 test_xxx 保持统一**；

④ 在贡献时候需要考虑最终校验的具体指标，精度(Acc.)、性能(Throu.)、显存(Mem.)，在对应指标空白处填上 `Y`，如无校验的保留空白即可。



#### Pipeline

①贡献脚本用例放置于`pipeline/`的对应模型文件夹下，如`baichuan2-13B`,文件命名规则为 {模型名}_{切分策略} 或者 {模型名}_{特性名称}， 如 `baichuan2_13B_tp8_pp1_ptd.sh`，请贡献者严格对齐；

② 注意脚本用例中不需要单独重定向log，日志收集工作已在 `pipe_run.sh` 中进行统一管理；

③ 标杆数据请放置于 `pipeline/baseline` 文件夹下，**命名保证完全与 shell 脚本对齐**，否则自动化脚本执行将扫描不到；

④ 获取标杆数据：通过门禁任务执行获得首次数据，并将结果保存至本地 log 或者 txt 文件中，后通过本地执行 `tests/st/st_utils/common.py` 中的 `transfer_logs_as_json` 函数进行提取，最后再连同用例脚本上仓即可；

⑤ 在贡献时候需要考虑最终校验的具体指标，精度(Acc.)、性能(Throu.)、显存(Mem.)，在对应指标空白处填上 `Y`，如无校验的保留空白即可。

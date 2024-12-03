## MindSpeed-LLM 测试用例贡献说明

### 门禁看护列表
<table>
    <tr>
        <th>Tests</th>
        <th>Module</th>
        <th>Structure</th>
        <th>Features</th>
        <th>Scripts</th>
        <th>Acc.</th>
        <th>Throu.</th>
        <th>Mem.</th>
    </tr>
    <tr>
        <td rowspan="16">ST</td>
        <td rowspan="12">Pretrain</td>
        <td>Mcore</td>
        <td>TP，PP，VPP，重计算，enable_recompute_layers_per_pp_rank</td>
        <td><a href="st/shell_scripts/llama2_tp2_pp4_vpp2_ptd.sh">llama2_tp2_pp4_vpp2.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>cp_ring，分布式优化器，reuse_fp32_param，recompute_activation_function，fused_rmsnorm，fused_swiglu，fused_rope，overlap_grad_reduce, overlap_param_gather</td>
        <td><a href="st/shell_scripts/llama2_tp2_cp4_mem_recompute.sh">llama2_tp2_cp4_mem_recompute.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>cp_hybrid，gqa</td>
        <td><a href="st/shell_scripts/chatglm3_gqa_cp8.sh">chatglm3_gqa_cp8.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>swap_attention，recompute_activation_function，enable_recompute_layers_per_pp_rank，reuse_fp32_param</td>
        <td><a href="tests/st/shell_scripts/llama2_tp2_pp4_vpp2_swap.sh">llama2_tp2_pp4_vpp2_swap.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>glm_rope, rotary_percent</td>
        <td><a href="st/shell_scripts/chatglm3_tp1_pp2_rope.sh">chatglm3_tp1_pp2_rope.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>EP，CP，num_experts，moe_router_topk，aux_loss，moe_allgather，group_query_attention，rotary_base</td>
        <td><a href="st/shell_scripts/mixtral_mcore_tp4_cp2_ep2_ptd.sh">mixtral_mcore_tp4_cp2_ep2_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>moe_expert_capacity_factor，moe_alltoall，pad_to_capacity, topk_softmax_with_capacity</td>
        <td><a href="st/shell_scripts/gpt4_mcore_tp4_cp2_32k_moe_drop.sh">gpt4_mcore_tp4_cp2_32k_moe_drop.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>enable_high_availability</td>
        <td><a href="st/shell_scripts/llama2_tp2_pp1_ha_save_ptd.sh">llama2_tp2_pp1_ha_save_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>mla_attention，moe_grouped_gemm，EP，allgather_dispatcher</td>
        <td><a href="st/shell_scripts/deepseek_v2_mcore_tp1_pp1_ep8.sh">deepseek_v2_mcore_tp1_pp1_ep8.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>post_norm, query_pre_attn_scalar, interleave_sliding_window, add_rmsnorm_offset, input_embeds_norm</td>
        <td><a href="st/shell_scripts/gemma2_tp8_pp1_ptd.sh">gemma2_tp8_pp1_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>MOE，PP，EP，Drop，DPP</td>
        <td><a href="st/shell_scripts/mixtral_tp1_pp4_ep2_drop_dpp.sh">mixtral_tp1_pp4_ep2_drop_dpp.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Legacy</td>
        <td>TP，PP，VPP，SP，全重计算，fused_rmsnorm，fused_swiglu，fused_rope，overlap_grad_reduce</td>
        <td><a href="st/shell_scripts/llama2_tp2_pp4_vpp2_legacy.sh">llama2_tp2_pp4_vpp2_legacy.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2">LoRA</td>
        <td rowspan="2">Legacy</td>
        <td>CCLoRA, TP, PP, 全重计算</td>
        <td><a href="st/shell_scripts/tune_llama2_tp2_pp4_lora_ptd.sh">tune_llama2_tp2_pp4_lora_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>CCLoRA单卡</td>
        <td><a href="st/shell_scripts/tune_llama2_tp1_pp1_lora_ptd.sh">tune_llama2_tp1_pp1_lora_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="2">FullSFT</td>
        <td>Legacy</td>
        <td>prompt_type, variable_seq_lengths</td>
        <td><a href="st/shell_scripts/tune_qwen7b_tp8_pp1_full_ptd.sh">tune_qwen7b_tp8_pp1_full_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>Mcore</td>
        <td>prompt_type, variable_seq_lengths, VPP</td>
        <td><a href="st/shell_scripts/tune_llama2_tp2_pp4_vpp2_mcore_full.sh">tune_llama2_tp2_pp4_vpp2_mcore_full.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td rowspan="15">UT</td>
        <td>Inference</td>
        <td>Legacy</td>
        <td>greedy_search, lora_inference, deterministic_computation, chatglm3_inference</td>
        <td><a href="ut/inference/test_inference.py">test_inference.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>Evaluation</td>
        <td>Legacy</td>
        <td>mmlu, prompt_mmlu,      
        prompt_boolq, prompt_ceval, qwen2_mmlu, lora_mmlu, agieval, humaneval, bbh</td>
        <td><a href="ut/evaluation/test_evaluate.py">test_evaluate.py</td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="3">CP</td>
        <td rowspan="3">Mcore</td>
        <td>hybrid</td>
        <td><a href="ut/dist_algo/context_parallel/test_hybrid_context_parallel.py">test_hybrid_context_parallel.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ring_attn</td>
        <td><a href="ut/dist_algo/context_parallel/test_ringattn_context_parallel.py">test_ringattn_context_parallel.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>ulysses</td>
        <td><a href="ut/dist_algo/context_parallel/test_ulysses_context_parallel.py"> test_ulysses_context_parallel.py </a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">ModelModule</td>
        <td rowspan="2">Mcore</td>
        <td>rope</td>
        <td><a href="ut/model_module/embeddings/test_rotary_pos_embedding.py">test_rotary_pos_embedding.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>transformer_attention, alibi</td>
        <td><a href="ut/model_module/transformer/test_attention.py">test_attention.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4">Checkpoint</td>
        <td rowspan="2"> Mcore </td>
        <td> hf2mcore, tp, pp, ep, dpp, vpp, deepseek2; hf2mcore, tp, deepseek2</td>
        <td><a href="ut/checkpoint/test_checkpoint.py">test_checkpoint.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>hf2mcore, tp, pp, dpp, vpp, chatglm3, qwen2</td>
        <td><a href="ut/checkpoint/test_convert_ckpt_from_huggingface.py">test_hf2mcore.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="2">Legacy</td>
        <td> legacy2mcore, lora</td>
        <td><a href="ut/checkpoint/test_convert_ckpt_from_huggingface.py">test_legacy2hf.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>legacy2legacy, lora</td>
        <td><a href="ut/checkpoint/test_convert_ckpt_from_megatron.py">test_legacy2legacy.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
	<tr>
        <td rowspan="3">ProcessData</td>
        <td rowspan="3">Mcore</td>
        <td>pretrain_data_alpaca, pretrain_merge_datasets, instruction_data_alpaca, instruction_merge_datasets</td>
        <td><a href="ut/process_data/test_preprocess_data.py">test_preprocess_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
	<tr>
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

</table>

### Pipline 二级流水看护列表
<table>
    <tr>
        <th>Model</th>
        <th>Structure</th>
        <th>Module</th>
        <th>Test Case</th>
        <th>Accuracy</th>
        <th>Throughput</th>
        <th>Memory</th>
    </tr>
    <tr>
        <td rowspan="4"><a href="pipeline/baichuan2-13B">Baichuan2-13B</a></td>
        <td rowspan="4">Legacy</td>
        <td>pretrain</td>
        <td><a href="pipeline/baichuan2-13B/baichuan2_13B_tp8_pp1_ptd.sh">baichuan2_13B_tp8_pp1_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>data_process</td>
        <td><a href="pipeline/baichuan2-13B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>inference</td>
        <td><a href="pipeline/baichuan2-13B/test_generation.py">test_generation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>evaluation</td>
        <td><a href="pipeline/baichuan2-13B/test_evaluation.py">test_evaluation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4"><a href="pipeline/chatglm3-6B">Chatglm3-6B</a></td>
        <td rowspan="4">Legacy</td>
        <td>pretrain</td>
        <td><a href="pipeline/chatglm3-6B/chatglm3_tp1_pp2_legacy.sh">chatglm3_tp1_pp2_legacy.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>data_process</td>
        <td><a href="pipeline/chatglm3-6B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>inference</td>
        <td><a href="pipeline/chatglm3-6B/test_generation.py">test_generation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>evaluation</td>
        <td><a href="pipeline/chatglm3-6B/test_evaluation.py">test_evaluation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4"><a href="pipeline/bloom-7B">Bloom-7B</a></td>
        <td rowspan="4">Legacy</td>
        <td>pretrain</td>
        <td><a href="pipeline/bloom-7B/bloom_7B_legacy_tp8_pp1_ptd.sh">bloom_7B_legacy_tp8_pp1_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>data_process</td>
        <td><a href="pipeline/bloom-7B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>inference</td>
        <td><a href="pipeline/bloom-7B/test_generation.py">test_generation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>evaluation</td>
        <td><a href="pipeline/bloom-7B/test_evaluation.py">test_evaluation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4"><a href="pipeline/gemma-7B">Gemma-7B</a></td>
        <td rowspan="4">Legacy</td>
        <td>pretrain</td>
        <td><a href="pipeline/gemma-7B/gemma_7B_legacy_tp8_pp1_ptd.sh">gemma_7B_legacy_tp8_pp1_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>data_process</td>
        <td><a href="pipeline/gemma-7B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>inference</td>
        <td><a href="pipeline/gemma-7B/test_generation.py">test_generation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>evaluation</td>
        <td><a href="pipeline/gemma-7B/test_evaluation.py">test_evaluation.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td rowspan="4"><a href="pipeline/qwen15-7B">Qwen15-7B</a></td>
        <td rowspan="4">Legacy</td>
        <td>pretrain</td>
        <td><a href="pipeline/qwen15-7B/qwen15_7b_legacy_tp8_pp1_ptd.sh">qwen15_7b_legacy_tp8_pp1_ptd.sh</a></td>
        <td>Y</td>
        <td>Y</td>
        <td>Y</td>
    </tr>
    <tr>
        <td>data_process</td>
        <td><a href="pipeline/qwen15-7B/test_process_pretrain_data.py">test_process_pretrain_data.py</a></td>
        <td>Y</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>inference</td>
        <td><a href="pipeline/qwen15-7B/test_generation.py">test_generation.py</a></td>
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
</table>




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

③ 新增用例可以在原有用例基础上做 `test_xxx` 的补充，尽量保证测试功能的集成性；对于存在 .json 文件的用例，贡献时在 .json 中加入 `test_xxx` 配置，然后在 .py 中通过 `@pytest.mark.parameterize` 传入参数、构造用例，**请注意 .json 中的 key 值命名需与 .py 中的 test_xxx 保持统一**；

④ 在贡献时候需要考虑最终校验的具体指标，精度(Acc.)、性能(Throu.)、显存(Mem.)，在对应指标空白处填上 `Y`，如无校验的保留空白即可。



#### Pipeline

①贡献脚本用例放置于`pipline/`的对应模型文件夹下，如`baichuan2-13B`,文件命名规则为 {模型名}_{切分策略} 或者 {模型名}_{特性名称}， 如 `baichuan2_13B_tp8_pp1_ptd.sh`，请贡献者严格对齐；

② 注意脚本用例中不需要单独重定向log，日志收集工作已在 `pipe_run.sh` 中进行统一管理；

③ 标杆数据请放置于 `pipline/baseline` 文件夹下，**命名保证完全与 shell 脚本对齐**，否则自动化脚本执行将扫描不到；

④ 获取标杆数据：通过门禁任务执行获得首次数据，并将结果保存至本地 log 或者 txt 文件中，后通过本地执行 `tests/st/st_utils/common.py` 中的 `transfer_logs_as_json` 函数进行提取，最后再连同用例脚本上仓即可；

⑤ 在贡献时候需要考虑最终校验的具体指标，精度(Acc.)、性能(Throu.)、显存(Mem.)，在对应指标空白处填上 `Y`，如无校验的保留空白即可。

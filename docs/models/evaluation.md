MindSpeed-LLM 支持大模型在公开基准数据集上进行准确率评估，当前支持的 Benchmark 如下：

| Benchmark | 下载链接                                                     | 验证集 | MindSpeed-LLM                                                | OpenCompass                                                  |
| --------- | ------------------------------------------------------------ | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MMLU      | [GitHub](https://people.eecs.berkeley.edu/~hendrycks/data.tar) | test   | [45.73%](../../examples/mcore/llama2/evaluate_llama2_7b_mmlu_ptd.sh) | [45.3%](https://hub.opencompass.org.cn/dataset-detail/MMLU)  |
| CEval     | [HuggingFace](https://huggingface.co/datasets/ceval/ceval-exam/tree/main) | val    | [33.87%](../../examples/mcore/llama2/evaluate_llama2_7b_ceval_ptd.sh) | [32.5%](https://hub.opencompass.org.cn/dataset-detail/C-Eval) |
| BoolQ     | [Juhe](https://www.juhe.cn/market/product/id/10243)          | dev    | [75.44%](../../examples/mcore/llama2/evaluate_llama2_7b_boolq_ptd.sh) | [74.9%](https://hub.opencompass.org.cn/dataset-detail/BoolQ) |
| BBH       | [GitHub](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh) | test   | [34.4%](../../examples/mcore/llama2/evaluate_llama2_7b_bbh_ptd.sh) | [32.5%](https://hub.opencompass.org.cn/dataset-detail/BBH)   |
| AGIEval   | [GitHub](https://github.com/ruixiangcui/AGIEval/tree/main)   | test   | [20.6%](../../examples/mcore/llama2/evaluate_llama2_7b_agieval_ptd.sh) | [20.6%](https://hub.opencompass.org.cn/dataset-detail/AGIEval) |
| HumanEval | [GitHub](https://github.com/openai/human-eval/tree/master/data) | test   | [12.8%](../../examples/mcore/llama2/evaluate_llama2_7b_humaneval_ptd.sh) | [12.2%](https://hub.opencompass.org.cn/dataset-detail/HumanEval) |


MindSpeed-LLM 已支持的大模型评估数据统计如下：


| 模型                   | 任务     | MindSpeed-LLM | 社区                                                                    | 模型               | 任务     | MindSpeed-LLM | 社区                                                                                 |
|----------------------|--------|-----------|-----------------------------------------------------------------------|------------------|--------|-----------|------------------------------------------------------------------------------------|
| Aquila-7B            | BoolQ  | 77.3%     | --                                                                    | Aquila2-7B       | BoolQ  | 77.8%     | --                                                                                 |
| Aquila2-34B          | BoolQ  | 88.0%     | --                                                                    | Baichuan-7B      | BoolQ  | 69.0%     | [67.0%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)                       |
| Baichuan-13B         | BoolQ  | 74.7%     | [73.6%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)          | Baichuan2-7B     | BoolQ  | 70.0%     | [63.2%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)                       |
| Baichuan2-13B        | BoolQ  | 78.0%     | [67.0%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)          | Bloom-7B         | MMLU   | 25.1%     | --                                                                                 |
| Bloom-176B           | BoolQ  | 64.5%     | --                                                                    | ChatGLM3-6B      | MMLU   | 61.5%     | --                                                                                 |
| GLM4-9B              | MMLU   | 74.5%     | [74.7%](https://huggingface.co/THUDM/glm-4-9b)                        | CodeQwen1.5-7B   | Human. | 54.8%     | [51.8%](https://qwenlm.github.io/zh/blog/codeqwen1.5/)                             |
| CodeLLaMA-34B        | Human. | 48.8%     | [48.8%](https://paperswithcode.com/sota/code-generation-on-humaneval) | Gemma-2B         | MMLU   | 39.6%     | --                                                                                 |
| Gemma-7B             | MMLU   | 52.2%     | --                                                                    | InternLM-7B      | MMLU   | 48.7%     | [51.0%](https://huggingface.co/internlm/internlm-7b)                               |
| Gemma2-9B            | MMLU   | 70.7%     | [71.3%](https://huggingface.co/google/gemma-2-9b)                     | Gemma2-27B       | MMLU   | 75.5%     | [75.2%](https://huggingface.co/google/gemma-2-27b)                                 |
| LLaMA-7B             | BoolQ  | 74.6%     | [75.4%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)          | LLaMA-13B        | BoolQ  | 79.6%     | [78.7%](https://hub.opencompass.org.cn/dataset-detail/BoolQ)                       |
| LLaMA-33B            | BoolQ  | 83.2%     | [83.1%](https://paperswithcode.com/sota/question-answering-on-boolq)  | LLaMA-65B        | BoolQ  | 85.7%     | [86.6%](https://paperswithcode.com/sota/question-answering-on-boolq)               |
| LLaMA2-7B            | MMLU   | 45.7%     | --                                                                    | LLaMA2-13B       | BoolQ  | 82.2%     | [81.7%](https://paperswithcode.com/sota/question-answering-on-boolq)               |
| LLaMA2-34B           | BoolQ  | 82.0%     | --                                                                    | LLaMA2-70B       | BoolQ  | 86.4%     | --                                                                                 |
| LLaMA3-8B            | MMLU   | 65.2%     | --                                                                    | LLaMA3-70B       | BoolQ  | 78.4%     | --                                                                                 |
| LLaMA3.1-8B          | MMLU   | 65.3%     | --                                                                    | LLaMA3.1-70B     | MMLU   | 81.8%     | --                                                                                 |
| LLaMA3.2-1B          | MMLU   | 31.8%     | [32.2%](https://modelscope.cn/models/LLM-Research/Llama-3.2-1B)       | LLaMA3.2-3B      | MMLU   | 56.3%     | [58.0%](https://modelscope.cn/models/LLM-Research/Llama-3.2-3B)                    |
| Mistral-7B           | MMLU   | 56.3%     | --                                                                    | Mixtral-8x7B     | MMLU   | 70.6%     | [70.6%](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu) |
| Mistral-8x22B        | MMLU   | 77%       | [77.8%](https://mistral.ai/news/mixtral-8x22b/)                       | MiniCPM-MoE-8x2B | BoolQ  | 83.9%     | --                                                                                 |
| QWen-7B              | MMLU   | 58.1%     | [58.2%](https://huggingface.co/Qwen/Qwen-7B)                          | Qwen-14B         | MMLU   | 65.3%     | [66.3%](https://huggingface.co/Qwen/Qwen-14B)                                      |
| QWen-72B             | MMLU   | 74.6%     | [77.4%](https://huggingface.co/Qwen/Qwen-72B)                         | QWen1.5-0.5B     | MMLU   | 39.1%     | --                                                                                 |
| QWen1.5-1.8b         | MMLU   | 46.2%     | [46.8%](https://qwenlm.github.io/zh/blog/qwen1.5/)                    | QWen1.5-4B       | MMLU   | 59.0%     | [56.1%](https://qwenlm.github.io/zh/blog/qwen1.5)                                  |
| QWen1.5-7B           | MMLU   | 60.3%     | [61.0%](https://qwenlm.github.io/zh/blog/qwen1.5/)                    | QWen1.5-14B      | MMLU   | 67.3%     | [67.6%](https://qwenlm.github.io/zh/blog/qwen1.5)                                  |
| QWen1.5-32B          | MMLU   | 72.5%     | [73.4%](https://huggingface.co/Qwen/Qwen-72B)                         | QWen1.5-72B      | MMLU   | 76.4%     | [77.5%](https://qwenlm.github.io/zh/blog/qwen1.5)                                  |
| Qwen1.5-110B         | MMLU   | 80.4%     | [80.4%](https://qwenlm.github.io/zh/blog/qwen1.5-110b/)               | Yi-34B           | MMLU   | 76.3%     | [75.8%](https://hub.opencompass.org.cn/dataset-detail/MMLU)                        |
| QWen2-0.5B           | MMLU   | 44.6%     | [45.4%](https://qwenlm.github.io/zh/blog/qwen2/)                      | QWen2-1.5B       | MMLU   | 54.7%     | [56.5%](https://qwenlm.github.io/zh/blog/qwen2/)                                   |
| QWen2-7B             | MMLU   | 70.3%     | [70.3%](https://qwenlm.github.io/zh/blog/qwen2/)                      | QWen2-57B-A14B   | MMLU   | 75.6%     | [76.5%](https://qwenlm.github.io/zh/blog/qwen2/)                                   |
| QWen2-72B            | MMLU   | 83.6%     | [84.2%](https://qwenlm.github.io/zh/blog/qwen2/)                      | MiniCPM-2B       | MMLU   | 51.6%     | [53.4%](https://github.com/OpenBMB/MiniCPM?tab=readme-ov-file#3)                   |
| DeepSeek-V2-Lite-16B | MMLU   | 58.1%     | [58.3%](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)          | QWen2.5-0.5B     | MMLU   | 47.67%    | [47.5%](https://qwenlm.github.io/blog/qwen2.5-llm/)                                |
| QWen2.5-1.5B         | MMLU   | 59.4%     | [60.9%](https://qwenlm.github.io/blog/qwen2.5-llm/)                   | QWen2.5-3B       | MMLU   | 65.6%     | [65.6%](https://qwenlm.github.io/blog/qwen2.5-llm/)                                |
| QWen2.5-7B           | MMLU   | 73.8%     | [74.2%](https://qwenlm.github.io/blog/qwen2.5-llm/)                   | QWen2.5-14B      | MMLU   | 79.4%     | [79.7%](https://qwenlm.github.io/blog/qwen2.5-llm/)                                |
| QWen2.5-32B          | MMLU   | 83.3%     | [83.3%](https://qwenlm.github.io/blog/qwen2.5-llm/)                   | QWen2.5-72B      | MMLU   | 85.59%    | [86.1%](https://qwenlm.github.io/blog/qwen2.5-llm/)                                |
| InternLM2.5-1.8b     | MMLU   | 51.3%     | [53.5%](https://huggingface.co/internlm/internlm2_5-1_8b)             | InternLM2.5-7B   | MMLU   | 71.6%     | [71.6%](https://huggingface.co/internlm/internlm2_5-7b)                            |
| InternLM2.5-20b      | MMLU   | 73.3%     | [74.2%](https://huggingface.co/internlm/internlm2_5-20b)              | InternLM3-8b     | MMLU   | 76.6%     | [76.6%](https://huggingface.co/internlm/internlm3-8b-instruct)                     |
| Yi1.5-6B             | MMLU   | 63.2%     | [63.5%](https://huggingface.co/01-ai/Yi-1.5-6B/tree/main)             | Yi1.5-9B         | MMLU   | 69.2%     | [69.5%](https://huggingface.co/01-ai/Yi-1.5-9B/tree/main)                          |
| Yi1.5-34B            | MMLU   | 76.9%     | [77.1%](https://huggingface.co/01-ai/Yi-1.5-34B/tree/main)            | CodeQWen2.5-7B   | Human. | 66.5%     | [61.6%](https://modelscope.cn/models/Qwen/Qwen2.5-Coder-7B)                        |
| Qwen2.5-Math-7B      |MMLU-STEM| 67.8%    | [67.8%](https://github.com/QwenLM/Qwen2.5-Math/tree/main/)            | Qwen2.5-Math-72B |MMLU-STEM| 83.7%    | [82.8%](https://github.com/QwenLM/Qwen2.5-Math/tree/main/)                         |
| MiniCPM3-4B          | MMLU   | 63.7%     | 64.6%                                                                 | Phi-3.5-mini-instruct | MMLU   | 64.39%    | 64.34%                                                                        |
| Phi-3.5-MoE-instruct | MMLU   | 78.5%     | [78.9%](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)        | DeepSeek-Math-7B  |MMLU-STEM| 56.5%   | [56.5%](https://github.com/deepseek-ai/DeepSeek-Math)                              |
| DeepSeek-V2.5        | MMLU   | 79.3%     | [80.6%](https://github.com/deepseek-ai/DeepSeek-V3)                   | DeepSeek-V2-236B | MMLU   | 78.1%     | [78.5%](https://huggingface.co/deepseek-ai/DeepSeek-V2)                            |
| LLaMA3.3-70B-Instruct | MMLU   | 82.7%     | --                                                                    |

## 以上模型脚本环境变量声明：
ASCEND_LAUNCH_BLOCKING：将Host日志输出到串口,0-关闭/1-开启<br>
ASCEND_SLOG_PRINT_TO_STDOUT：设置默认日志级别,0-debug/1-info/2-warning/3-error<br>
HCCL_WHITELIST_DISABLE：HCCL白名单开关,1-关闭/0-开启<br>
HCCL_CONNECT_TIMEOUT：设置HCCL超时时间，默认值为120<br>
CUDA_DEVICE_MAX_CONNECTIONS：定义了任务流能够利用或映射到的硬件队列的数量<br>
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化<br>
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景<br>
PYTORCH_NPU_ALLOC_CONF：内存碎片优化开关，默认是expandable_segments:False，使能时expandable_segments:True<br>
ASCEND_RT_VISIBLE_DEVICES：指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。通过该环境变量，可实现不修改应用程序即可调整所用Device的功能。<br>
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量<br>
TOKENIZERS_PARALLELISM： 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为<br>

## 大模型分布式评估使用介绍

####  1. 基准评估
MindSpeed-LLM 基准评估脚本命名风格及启动方法为：
```shell
# Legacy
# 命名及启动：examples/legacy/model_name/evaluate_xxx.sh
bash examples/legacy/llama2/evaluate_llama2_7b_ptd.sh

# Mcore
# 命名及启动：examples/mcore/model_name/evaluate_xxx.sh
bash examples/mcore/llama2/evaluate_llama2_7b_mmlu_ptd.sh

# 使用lora权重的评估脚本命名风格及启动方法为（以 legacy 为例）：
bash examples/legacy/llama2/evaluate_llama2_7B_lora_ptd.sh
```

```shell
# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/llama-2-hf/"  #词表路径
CHECKPOINT="./model_weights/llama-2-7b-legacy"  #权重路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"  # 支持 mmlu、ceval、agieval、bbh、boolq、human_eval

# 启动评估脚本（以 legacy 为例）
bash examples/legacy/llama2/evaluate_llama2_7B_mmlu_ptd.sh
```

【--max-new-tokens】

表示模型输出的生成长度，多项选择问题的输出长度会比编码任务的输出长度小，该参数很大程度上影响了模型的评估性能


【--evaluation-batch-size】

可以设置多batch推理，提升模型评估性能

【--broadcast】

在开启Expert Parallel时，需启用该参数进行评估


####  2. 指令微调评估

使用指令微调后权重的评估脚本命名风格及启动方法为（以 legacy 为例）：

```shell
bash examples/legacy/llama2/evaluate_llama2_7B_full_mmlu_ptd.sh
```

【--prompt-type】

模型对话模板，选择模型对应的对话模板进行评估

【--hf-chat-template】

如果模型的tokenizer已经具备`chat_template`属性，则可以选择通过添加`--hf-chat-template`来使用模型内置的对话模板进行评估

【--eval-language】

根据评估数据集语言来确定，默认为`en`，如果评估数据集为中文数据集，则应设置为`zh`

####  3. LoRA权重评估

使用lora权重的评估脚本命名风格及启动方法为（以 legacy 为例）：

```shell
bash examples/legacy/llama2/evaluate_llama2_7B_lora_mmlu_ptd.sh
```

## 评估介绍
[mmlu评估介绍](../../docs/features/mmlu-evalution.md)
[cmmlu评估介绍](../../docs/features/cmmlu-evalution.md)

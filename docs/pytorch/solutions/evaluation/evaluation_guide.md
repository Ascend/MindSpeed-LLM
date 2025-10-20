MindSpeed-LLM 支持大模型在公开基准数据集上进行准确率评估，当前支持的 Benchmark 如下：
详细统计信息见[evaluation.md](../../models/models_evaluation.md)

## 以上模型脚本环境变量声明：
脚本中的环境变量配置见[环境变量说明](../../features/environment_variable.md)

## 大模型分布式评估使用介绍

####  1. 基准评估
MindSpeed-LLM 基准评估脚本命名风格及启动方法为：
```shell
# 命名及启动：examples/mcore/model_name/evaluate_xxx.sh
bash examples/mcore/llama2/evaluate_llama2_7b_mmlu_ptd.sh
```

```shell
# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf/"  #词表路径
CHECKPOINT="./model_weights/llama-2-7b-mcore"  #权重路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"  # 支持 mmlu、ceval、agieval、bbh、boolq、human_eval

# 启动评估脚本
bash examples/mcore/llama2/evaluate_llama2_7b_mmlu_ptd.sh
```

【--max-new-tokens】

表示模型输出的生成长度，多项选择问题的输出长度会比编码任务的输出长度小，该参数很大程度上影响了模型的评估性能


【--evaluation-batch-size】

可以设置多batch推理，提升模型评估性能

【--broadcast】

在开启Expert Parallel时，需启用该参数进行评估


####  2. 指令微调评估

使用指令微调后权重的评估脚本命名风格及启动方法为：

```shell
bash examples/mcore/llama2/evaluate_llama2_7b_full_mmlu_ptd.sh
```

【--prompt-type】

模型对话模板，选择模型对应的对话模板进行评估

【--hf-chat-template】

如果模型的tokenizer已经具备`chat_template`属性，则可以选择通过添加`--hf-chat-template`来使用模型内置的对话模板进行评估

【--eval-language】

根据评估数据集语言来确定，默认为`en`，如果评估数据集为中文数据集，则应设置为`zh`

####  3. LoRA权重评估

使用lora权重的评估脚本命名风格及启动方法为：

```shell
# 需要加载lora权重启动评估脚本，命名风格及启动方法为：
bash examples/mcore/codellama/evaluate_codellama_34b_lora_ptd.sh
```

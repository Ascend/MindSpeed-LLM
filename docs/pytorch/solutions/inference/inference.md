# MindSpeed-LLM 流式推理

## 流式推理效果

【Chatglm3-6b-Base流式推理效果】

```shell
Instruction: "how are you?"
MindSpeed-LLM:   "I'm just a computer program, so I don't have feelings or physical sensations, \
              but I'm here to help you with any questions you might have. \
              Is there something specific you would like to know?"
HuggingFace: "I'm just a computer program, so I don't have feelings or physical sensations, \
              but I'm here to help you with any questions you might have. \
              Is there something specific you would like to know?"
```

【Lama3.1-8b-Instruct流式推理效果】

```shell
Instruction: "how are you?"
MindSpeed-LLM:   "I hope you are doing well. I am writing to ask for your help with a project I am working on. \
              I am a student at [University Name] and I am doing a research project on [Topic]."
HuggingFace: "I hope you are doing well. I am writing to ask for your help with a project I am working on.\ 
              I am a student at [University Name] and I am doing a research project on [Topic]."
```

## 流式推理示例

### 初始化环境变量 

`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

`source /usr/local/Ascend/nnal/atb/set_env.sh`

### 启动脚本

使用LLaMA2-7B模型目录下的[流式推理脚本](../../../../examples/mcore/llama2/generate_llama2_7b_ptd.sh)。

#### 填写相关路径

`CHECKPOINT`：指向权重转换后保存的路径。

`TOKENIZER_PATH`：指定模型的分词器所在文件夹路径。

`TOKENIZER_MODEL`：指定模型的分词器文件路径（例如`tokenizer.model`）。


因此，根据之前的示例，路径应填写如下：
```shell
CHECKPOINT="./model_weights/llama-2-7b-mcore/"
TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf/"
TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf/tokenizer.model"
```
【注意】除了路径的配置，推理脚本中的其他参数应与训练时的参数保持一致（例如TP/PP/EP/VPP等并行切分，--noop-layers/--num-layer-list等自定义模型结构参数），否则会出现模型权重加载失败的情况。
#### 运行脚本

```shell
bash examples/mcore/llama2/generate_llama2_7b_ptd.sh
```


#### 流式推理脚本相关参数

【--task】

用于指定推理过程中所采用的生成策略，支持 greedy_search, greedy_search_with_sampling, beam_search, beam_search_with_sampling及chat策略。

【--stream】

使能此参数后可实现流式推理输出。

【--max-new-tokens】

生成的token的最大数量，忽略输入提示中的token数量。

【--max-length】

生成的token的最大长度。对应于输入prompt的长度 + max_new_tokens。如果设置了max_new_tokens，该参数的效果将被覆盖。

【--add-eos-token】

设置结束序列生成的token标识符，支持设置多个结束符。

【--top-k】

默认为0，若设置大于1的整数，则保留前k个概率最高的词汇token，用于top-k采样。

【--top-p】

默认为1.0，若设置为小于1的浮点数，则仅保留概率总和为top-p或更高的最小一组最有可能的token进行采样。

【--temperature】

默认为1.0，采样策略中的温度系数。

【--num-beams】

使用束搜索（Beam Search）策略时，设置束的大小。

【--use-kv-cache】

在线推理过程中设置使用kv-cache来加速生成。

【--use-flash-attn】

在线推理过程中设置使用prompt_flash_attention和incre_flash_attention来加速推理生成，需要在kv-cache开启的前提下使用，目前支持MHA，GQA和Alibi。

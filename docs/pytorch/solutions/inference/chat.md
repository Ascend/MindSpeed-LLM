# MindSpeed-LLM Chat对话

## Chat对话示例

### 使用说明

可以使用模型的[chat脚本](../../../../examples/mcore/llama2/chat_llama2_7b_ptd.sh)并加载预置模型权重，与模型进行多轮对话。其中预置模型权重可以是大语言模型的chat权重或是[指令微调](../finetune/instruction_finetune.md)后的权重。

### 初始化环境变量 

`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

`source /usr/local/Ascend/nnal/atb/set_env.sh`

### 启动脚本
使用LLaMA2-7B模型目录下的[chat脚本](../../../../examples/mcore/llama2/chat_llama2_7b_ptd.sh)。

#### 填写相关路径

`CKPT_LOAD_DIR`：指向权重转换后保存的路径。

`TOKENIZER_PATH`：指定模型的分词器所在文件夹路径。

`TOKENIZER_MODEL`：指定模型的分词器文件路径（例如`tokenizer.model`）。


因此，根据之前的示例，路径应填写如下：
```shell
CHECKPOINT="./model_weights/llama-2-7b-mcore/"
TOKENIZER_PATH="./model_from_hf/llama-2-7b-hf-chat/"
TOKENIZER_MODEL="./model_from_hf/llama-2-7b-hf-chat/tokenizer.model"
```

【--tokenizer-type】 

参数值为PretrainedFromHF时， 词表路径仅需要填到模型文件夹即可，不需要到tokenizer.model文件

**示例：**
```shell 
    TOKENIZER_PATH="./model_from_hf/llama-2-hf/"
    --tokenizer-name-or-path ${TOKENIZER_PATH}
```
参数值不为PretrainedFromHF时，例如Llama2Tokenizer，需要指定到tokenizer.model文件

**示例：**
```shell 
    TOKENIZER_MODEL="./model_from_hf/llama-2-hf/tokenizer.model"
    --tokenizer-model ${TOKENIZER_MODEL} \
```


#### 运行脚本

```shell
bash examples/mcore/llama2/chat_llama2_7b_ptd.sh
```

#### Chat对话脚本相关参数

多轮对话脚本与流式推理相关的参数设置可以在[`流式推理`](inference.md)文件内查看。

【--task】

默认参数为"chat", 用于指定进行多轮对话任务。

【--history-turns】

在多轮对话中，可以指定参数`--history-turns`来改变历史对话记录轮数，默认为记录`3`轮

【--hf-chat-template】

如果模型的tokenizer已经具备`chat_template`属性，则可以选择通过添加`--hf-chat-template`来使用模型内置的对话模板

【--prompt-type】

模型对话模板，作用与`--hf-chat-template`一致，但不需要模型的tokenizer已经具备`chat_template`属性，微调后推理对话时应选择模型对应的对话模板， `prompt-type`的可选项可以在[`templates`](../../../../configs/finetune/templates.json)文件内查看。


#### Chat对话程序相关命令

在Chat对话程序出现`You >> `的提示后，可以输入文本与模型进行多轮对话。

【清空对话记录】

如果需要清空之前的对话记录新建一个Chat对话程序，可以对话框中输入  [clear, new] 其中的一个命令

【退出对话程序】

如果需要退出Chat对话程序，可以在对话框中输入 [q, quit, exit] 其中的一个命令。


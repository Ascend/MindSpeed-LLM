import os
import time
import signal
import platform
import argparse

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, logging
from modeling_visualglm_ascend import ChatGLMForConditionalGenerationWithImage
logging.set_verbosity_error()

from utils.image_encoder import IMAGE_ENCODER_OM

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
BATCH = [1]
max_seq_len = 2048


# 修改transformers的TopPLogitsWarper
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    cumulative_probs = sorted_logits.softmax(
        dim=-1).cpu().float().cumsum(dim=-1).to(sorted_logits.device).to(sorted_logits.dtype)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
    if self.min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores


transformers.generation.TopPLogitsWarper.__call__ = __call__


def _init_torch_npu():
    # 使用二进制优化，消除动态shape的编译问题
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "Tril"
    torch.npu.set_option(option)


def set_device(device_id):
    torch.npu.set_device(torch.device(f"npu:{device_id}"))


def load_model(model_path, device_id):
    # from utils.image_encoder import IMAGE_ENCODER_OM
    image_encoder_om = IMAGE_ENCODER_OM(model_path, int(device_id))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    soc_version = torch_npu._C._npu_get_soc_version()
    model = ChatGLMForConditionalGenerationWithImage.from_pretrained(model_path, trust_remote_code=True).half().npu().eval()
    if soc_version not in [104, 220, 221, 222, 223, 224]:
        # if on Atlas or Atalas推理系列产品 chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    # adapt lm_head padding weight shape
                    hs = config.hidden_size
                    lmhead_weight_offset = torch.zeros(14, hs, device=module.weight.data.device, dtype=module.weight.data.dtype)
                    # eliminate TransData op before lm_head calculation
                    module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                    module.weight.data = torch.cat((module.weight.data, lmhead_weight_offset), dim=0)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
        print("soc version: ", soc_version, " is not Atlas 900 A2 PODc, support NZ")
    return model, tokenizer


def build_prompt(history, prefix):
    prompt = prefix
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nVisualGLM-6B：{response}"
    return prompt


def signal_handler(signal_in, frame):
    global stop_stream
    stop_stream = True


def main(model, tokennizer):
    global stop_stream
    while True:
        history = []
        prefix = "欢迎使用 VisualGLM-6B 模型，输入图片路径和内容即可进行对话，clear 清空对话历史，stop 终止程序"
        print(prefix)
        image_path = input("\n请输入图片路径：")
        if image_path == "stop":
            break
        prefix = prefix + "\n" + image_path
        query = "描述这张图片。"
        while True:
            count = 0
            torch.npu.synchronize()
            with torch.no_grad():
                for response, history in model.stream_chat(tokenizer, image_path, query, history=history):
                    if stop_stream:
                        stop_stream = False
                        break
                    else:
                        count += 1
                        if count % 8 == 0:
                            os.system(clear_command)
                            print(build_prompt(history, prefix), flush=True)
                            signal.signal(signal.SIGINT, signal_handler)
            os.system(clear_command)
            print(build_prompt(history, prefix), flush=True)
            query = input("\n用户：")
            if query.strip() == "clear":
                break
            if query.strip() == "stop":
                stop_stream = True
                exit(0)


def warm_up(model, batch):
    past_key_values = None
    input_ids = torch.randint(130000, (batch, 4)).npu()
    input_ids[:, -2] = 130001
    input_ids[:, -1] = 130004
    position_ids = torch.randint(2048, (1, 2, 4)).npu()
    position_ids[0][0][0] = 2047
    # attention_mask = (torch.randint(4, (1, 1, 4, 4)) == torch.randint(1, (1, 1, 4, 4))).npu()
    attention_mask = None
    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }
    with torch.no_grad():
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    for _ in range(5):
        past_key_values = outputs.past_key_values
        input_ids = torch.randint(130000, (batch, 1)).npu()
        position_ids = torch.randint(2048, (1, 2, 1)).npu()
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids
        }
        with torch.no_grad():
            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )


#全量+增量
def full_and_incremental_test(seq_len, batch, test_cycle, model):
    print("start run.")
    warm_up(model, batch)
    torch.npu.empty_cache()
    past_key_values = None
    input_ids = torch.randint(130000, (batch, seq_len)).npu()
    input_ids[:, -2] = 130001
    input_ids[:, -1] = 130004
    position_ids = torch.randint(2048, (1, 2, seq_len)).npu()
    position_ids[0][0][0] = 2047
    # attention_mask = (torch.randint(4, (1, 1, seq_len, seq_len)) == torch.randint(1, (1, 1, seq_len, seq_len))).npu()
    attention_mask = None
    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }
    torch.npu.synchronize()
    start = time.time()
    with torch.no_grad():
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
    torch.npu.synchronize()
    end = time.time()
    first_time = (end - start) * 1000
    print(f"first token: {first_time}ms")
    sum_time = 0
    test_cycle -= 1
    for i in range(test_cycle):
        torch.npu.empty_cache()
        past_key_values = outputs.past_key_values
        input_ids = torch.randint(130000, (batch, 1)).npu()
        position_ids = torch.randint(2048, (1, 2, 1)).npu()
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids
        }
        torch.npu.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        torch.npu.synchronize()
        end = time.time()
        cur_time = (end - start) * 1000
        sum_time += cur_time
    avg_time = sum_time / test_cycle
    print(f"average token: {sum_time / test_cycle}ms")
    print(f"response time: {first_time + sum_time}ms")
    return first_time, avg_time


def performance_test(model):
    for batch_level in BATCH:
        file = open(f"{batch_level}_batch_performance_visualglm.csv", 'w')
        file.write(f"Batch,MaxSeqLen,InputSeqLen(Encoding),OutputSeqLen(Decoding),ResponseTime(ms),FirstTokenTime(ms),TimePerTokens(ms),TokensPerSecond(ms)\n")
        for seq_len_level in range(5, 11):
            for test_cycle_level in range(5, 11):
                seq_len = 2 ** seq_len_level
                test_cycle = 2 ** test_cycle_level
                input_param = {
                    "seq_len": seq_len,
                    "batch": batch_level,
                    "test_cycle": test_cycle,
                    "model": model
                    }
                print(f"batch: {batch_level}, seq_len: {seq_len}, test_cycle: {test_cycle}")
                first_time, avg_token = full_and_incremental_test(**input_param)
                file.write(f"{batch_level},{max_seq_len},{seq_len},{test_cycle},{round(first_time+avg_token*test_cycle, 2)},{round(first_time, 2)},{round(avg_token, 2)},{round(1000/avg_token,2)}\n")
        file.close()


def run_example(args, model):
    time1 = time.time()
    image_path = os.path.join(args.model_path, "./examples/1.jpeg")
    response, history = model.chat(tokenizer, image_path, "描述一下这个场景,这个场景是一部电影中的经典桥段", history=[])
    print('首次执行时间', time.time() - time1)
    print(response, '\n')

    time3 = time.time()
    image_path_2 = os.path.join(args.model_path, "./examples/2.jpeg")
    response, history = model.chat(tokenizer, image_path_2, "这是什么东西", history=[])
    print('非首次执行时间', time.time() - time3)
    print(response, '\n')

    time5 = time.time()
    image_path_3 = os.path.join(args.model_path, "./examples/3.jpeg")
    response, history = model.chat(tokenizer, image_path_3, "这张图片描述了什么", history=[])
    print('非首次执行时间', time.time() - time5)
    print(response, '\n')


if __name__ == "__main__":
    _init_torch_npu()
    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/testdata/weights/visualglm6b",
        help="Location of Model weights, which contains model folders",)
    parser.add_argument(
        "--device_id",
        type=str,
        default=0,
        help="Choose device id",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="performance",
        help="mode",
    )
    args = parser.parse_args()
    set_device(args.device_id)
    print("args.model_path=", args.model_path)
    
    main_model, main_tokenizer = load_model(args.model_path, args.device_id)
    if args.mode == "performance":
        performance_test(main_model)
    elif args.mode == "precision":
        os.environ["PRECISION_TEST"] = "1"
        run_example(args, main_model)
    elif args.mode == "predict":
        run_example(args, main_model)
    elif args.mode == "run":
        main(main_model, main_tokenizer)
    else:
        print("mode not implemented!")
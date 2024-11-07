import csv
import os.path as osp
import time

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from initialize import initialize, initialize_model_and_tokenizer


soc_version_map = {-1: "unknown soc version",
                   100: "Ascend100", 101: "Ascend101", 102: "Ascend102", 103: "Ascend103", 104: "Ascend104",
                   200: "Ascend200", 201: "Ascend201", 202: "Ascend202", 203: "Ascend203",
                   220: "Ascend220", 221: "Ascend221", 222: "Ascend222", 223: "Ascend223",
                   240: "Ascend240", 241: "Ascend241", 242: "Ascend242",
                   250: "Ascend250", 251: "Ascend251", 252: "Ascend252", 253: "Ascend253"
                   }


def add_performance_specific_args(parser):
    """Arguments for performance test"""
    group = parser.add_argument_group(
        "performance", "Evaluation configurations")
    group.add_argument(
        "--test_batchs", 
        type=lambda a: list(map(int, a.split(","))), 
        default="1,4,8",
        help="All batchsizes to test performance"
    )
    group.add_argument(
        "--test_seqlen_cases", 
        type=lambda a: [(int(b.split(',')[0]), int(b.split(',')[1])) 
                        for b in a.split(";")], 
        default="256,64;512,128;1024,256;1536,512",
        help="All (seqlen, test_cycle) to test performance"
    )
    group.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="a directory to save csv files"
    )


def warm_up(model, batch, seq_len=1024):
    input_ids = torch.randint(
        150000, (batch, seq_len), device=torch.npu.current_device(), 
        dtype=torch.int64
    )
    input_ids[:, -2] = 150001
    input_ids[:, -1] = 150004
    position_ids = torch.arange(
        seq_len, device=torch.npu.current_device(), 
        dtype=torch.int64).view(1, -1)
    attention_mask = torch.randn(
        1, 1, seq_len, seq_len, device=torch.npu.current_device()) < 0.5
    with torch.no_grad():
        logits, *_ = model(input_ids, position_ids, attention_mask)

    for i in range(5):
        input_ids = torch.randint(
            150000, (batch, 1), device=torch.npu.current_device(), 
            dtype=torch.int64
        )
        position_ids = torch.arange(
            1, device=torch.npu.current_device(), dtype=torch.int64).view(1, -1)
        attention_mask = torch.randn(
            1, 1, 1, seq_len + i + 1, device=torch.npu.current_device()) < 0.5
        logits, *_ = model(input_ids, position_ids, attention_mask)


def test_performance_for_one_case(model, seq_len, batch, test_cycle):
    warm_up(model, batch, seq_len=512 if seq_len < 512 else seq_len)
    input_ids = torch.randint(150000, (batch, seq_len),
                              device=torch.npu.current_device())
    input_ids[:, -2] = 150001
    input_ids[:, -1] = 150004
    position_ids = torch.arange(
        seq_len, device=torch.npu.current_device(), dtype=torch.int64
        ).view(1, -1)
    attention_mask = torch.randn(
        1, 1, seq_len, seq_len, device=torch.npu.current_device()) < 0.5
    
    model_time = []
    for i in range(test_cycle):
        torch.npu.synchronize()
        model_start = time.time()
        logits, *_ = model(input_ids, position_ids, attention_mask)
        # synchronize to make sure the model time is correct.
        torch.npu.synchronize()
        model_time.append(time.time() - model_start)
        input_ids = torch.randint(
            150000, (batch, 1), device=torch.npu.current_device())
        position_ids = torch.arange(
            1, device=torch.npu.current_device(), dtype=torch.int64).view(1, -1)
        attention_mask = torch.randn(
            1, 1, 1, seq_len + i + 1, device=torch.npu.current_device()) < 0.5

    if torch.distributed.get_rank() == 0:
        print('Batch size = ', batch)
        print('Input seqlen = ', seq_len)
        print('Output seqlen = {}, takes {} ms.'.format(
            test_cycle, round(sum(model_time) * 1000, 4)))
        print('E2E performance is {} token/second.'.format(
            round((test_cycle) / (sum(model_time)), 4)))
        print('First token\'s model latency is {} ms.'.format(
            round(model_time[0] * 1000, 2)))
        print('Model latency is {} ms.'.format(round(
            (sum(model_time) - model_time[0]) * 1000 / (test_cycle - 1), 2)))
        print('Model latency list is:{}'.format(
            [round(item * 1000, 2) for item in model_time]))
    
    return (
        round(model_time[0] * 1000, 2), 
        round((sum(model_time) - model_time[0]) * 1000 / (test_cycle - 1), 2)
    )


def test_performance(
        model, 
        test_batchs=(1, 4, 8), 
        test_seqlen_cases=((256, 64), (512, 128), (1024, 256), (1536, 512)),
        output_dir="./"):
    rank = torch.distributed.get_rank()
    device_version = soc_version_map.get(torch_npu._C._npu_get_soc_version())

    csv_path = osp.join(
        output_dir, f"zhiputest_{device_version}_rank{rank}.csv")
    file = open(csv_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(file)
    table_header = (
        "Batch", "MaxSeqLen", "InputSeqLen(Encoding)",
        "OutputSeqLen(Decoding)", "TokensPerSecond", "ResponseTime(ms)",
        "FirstTokenTime(ms)", "TimePerTokens(ms)"
    )
    csv_writer.writerow(table_header)
    
    for batch_level in test_batchs:
        for seq_len, test_cycle in test_seqlen_cases:
            input_param = {
                "seq_len": seq_len,
                "batch": batch_level,
                "test_cycle": test_cycle,
                "model": model
            }
            first_time, avg_token = test_performance_for_one_case(**input_param)
            csv_writer.writerow([
                batch_level, 2048, seq_len, test_cycle, 1000 / avg_token,
                first_time + avg_token * test_cycle, first_time, avg_token
            ])

    file.close()

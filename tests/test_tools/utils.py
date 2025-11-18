"""
We can't use assert in our code for codecheck, so create this auxiliary function to wrap
the assert case in ut for ci.
"""
import os
import hashlib
import logging
import re
import json
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
import subprocess
import torch
import torch_npu
import xxhash

import pytest

import megatron.core.parallel_state as mpu
from megatron.core.parallel_state import initialize_model_parallel
from mindspeed.core.parallel_state import initialize_model_parallel_wrapper
from mindspeed.core.context_parallel.model_parallel_utils import initialize_model_parallel_cp_wrapper
from mindspeed_llm.core.parallel_state import initialize_model_parallel_decorator


def judge_expression(expression):
    if not expression:
        raise AssertionError


def hash_tensor_in_chunks(tensor, chunk_size=1024 * 1024):
    """分块计算张量的哈希值，适用于超大张量，支持 BFloat16 等类型。"""
    hasher = xxhash.xxh3_64()
    numel = tensor.numel()
    tensor_flat = tensor.view(-1)  # 展平张量为一维

    if tensor.dtype == torch.bfloat16:
        tensor_flat = tensor_flat.to(torch.float32)

    for i in range(0, numel, chunk_size):
        chunk = tensor_flat[i:i + chunk_size]
        if not chunk.is_contiguous():
            chunk = chunk.contiguous()
        hasher.update(chunk.cpu().numpy().tobytes())

    return hasher.digest()


def calculate_hash_for_model(data, chunk_size=1024 * 1024):
    final_hasher = xxhash.xxh3_64()

    tensor_data = {k: v for k, v in data.items() if torch.is_tensor(v)}
    non_tensor_data = {k: v for k, v in data.items() if not torch.is_tensor(v)}

    if tensor_data:
        tensor_hashes = [
            hash_tensor_in_chunks(value, chunk_size)
            for key, value in sorted(tensor_data.items())
        ]
        for key, tensor_hash in zip(sorted(tensor_data.keys()), tensor_hashes):
            final_hasher.update(key.encode('utf-8'))
            final_hasher.update(tensor_hash)

    for key in sorted(non_tensor_data.keys()):
        final_hasher.update(key.encode('utf-8'))
        value = non_tensor_data[key]
        if isinstance(value, (int, float)):
            final_hasher.update(str(value).encode('utf-8'))
        elif isinstance(value, str):
            final_hasher.update(value.encode('utf-8'))
        else:
            final_hasher.update(repr(value).encode('utf-8'))

    return final_hasher.hexdigest()


def compare_state_dicts(state_dict1, state_dict2):
    # Add rerun_state_machine for the Megatron-0.8.0 checkpoint
    if 'rerun_state_machine' not in state_dict1.keys() and 'rerun_state_machine' in state_dict2.keys():
        state_dict1['rerun_state_machine'] = state_dict2['rerun_state_machine']
    if state_dict1.keys() != state_dict2.keys():
        print(f"base:{state_dict1.keys()} != save:{state_dict2.keys()}")
        return False

    for key in state_dict1.keys():
        value1 = state_dict1[key]
        value2 = state_dict2[key]

        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            if not torch.equal(value1, value2):
                print(f"Difference found in key: {key}")
                return False
        elif isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_state_dicts(value1, value2):
                return False
        else:
            pass

    return True


def process_file(file_path):
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    layer_ckpt = {}
    # 兼容带vpp的权重
    for key in data.keys():
        if key.startswith('model'):
            layer_ckpt.update(data[key])
    data = layer_ckpt
    return data


def compare_with_base_hash(file_path, base_hash, file_type='pt'):
    if not os.path.exists(file_path):
        return f"Error: File {file_path} does not exist"

    if file_type == 'pt':
        try:
            data = process_file(file_path)
            if isinstance(data, str):
                return data
            current_hash = calculate_hash_for_model(data)
        except Exception as e:
            raise ValueError(f"Error: Failed to process file {file_path} - {str(e)}") from e
    elif file_type == 'safetensors' or 'bin':
        current_hash = get_md5sum(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return current_hash == base_hash


def weight_compare_hash(model_dir, base_hash, suffix="pt"):
    models_path = glob.glob(os.path.join(model_dir, '**', f'*.{suffix}'), recursive=True)
    models_path.sort()
    if not models_path:
        raise ValueError(f"Error: No .{suffix} files found in current directory")

    if len(models_path) != len(base_hash):
        raise ValueError(f"Error: Number of files don't match ({len(models_path)} vs {len(base_hash)})")

    cpu_count = os.cpu_count() or 1
    max_workers = min(cpu_count, len(models_path))
    logging.info(f"Using {max_workers} workers based on CPU count: {cpu_count}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            executor.submit(compare_with_base_hash, models_path[i], base_hash[i], suffix)
            for i in range(len(models_path))
        ]

        for _, future in enumerate(tasks):
            result = future.result()
            if not result:
                return False

    return True


def weight_compare(dir_1, dir_2, suffix="pt", use_md5=False):
    models_path = glob.glob(os.path.join(dir_1, '**', f'*.{suffix}'), recursive=True)
    if not models_path:
        print(f"Can't find any weight files in {dir_1}.")
        return False
    for path_1 in models_path:
        path_1 = os.path.normpath(path_1)
        path_2 = path_1.replace(os.path.normpath(dir_1), os.path.normpath(dir_2))
        if use_md5:
            are_equal = (get_md5sum(path_1) == get_md5sum(path_2))
        else:
            state_dict1 = torch.load(path_1, weights_only=False)
            state_dict2 = torch.load(path_2, weights_only=False)
            are_equal = compare_state_dicts(state_dict1, state_dict2)
        if not are_equal:
            return False

    return True


def weight_compare_optim(dir_1, dir_2, suffix="pt", use_md5=False):
    models_path = glob.glob(os.path.join(dir_1, '**', f'*.{suffix}'), recursive=True)
    
    if not models_path:
        raise FileNotFoundError(f"{dir_1} is not a file or not exists !")
    
    for path_1 in models_path:
        path_1 = os.path.normpath(path_1)
        path_2 = path_1.replace(os.path.normpath(dir_1), os.path.normpath(dir_2))

        file_name = os.path.basename(path_1)
        if file_name == 'distrib_optim.pt':
            use_md5 = True  
        elif file_name == 'model_optim_rng.pt':
            use_md5 = False  

        if use_md5:
            are_equal = (get_md5sum(path_1) == get_md5sum(path_2))
        else:
            state_dict1 = torch.load(path_1, weights_only=False)
            state_dict2 = torch.load(path_2, weights_only=False)
            are_equal = compare_state_dicts(state_dict1, state_dict2)
        
        if not are_equal:
            return False

    return True


def compare_file_md5_same(file1, file2):
    return get_md5sum(file1) == get_md5sum(file2)


def get_md5sum(fpath):
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"{fpath} is not a file or not exists !")
    md5sum = hashlib.md5()
    with open(fpath, 'rb') as f:
        md5sum.update(f.read())
        return md5sum.hexdigest()


def delete_distrib_optim_files(folder_path):

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "distrib_optim.pt":
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted: {file_path}")
                except Exception as e:
                    logging.exception(f"Failed to delete {file_path}: {e}")  
                    raise 

                
@pytest.fixture
def build_args(request, monkeypatch):
    params = request.getfixturevalue("params")
    argv = [sys.argv[0]]
    for k, v in params.items():
        if v is None:
            argv.append(f"--{k}")
        elif isinstance(v, list):
            argv.extend([f"--{k}"] + [str(value) for value in v])
        else:
            argv.extend([f"--{k}", str(v)])
    monkeypatch.setattr(sys, "argv", argv)


def create_testconfig(path: str, cmd: bool = False):
    with open(path) as f:
        raw_data = json.load(f)
    
    res = {k: [tuple(s.values()) if len(s) > 1 else tuple(s.values())[0] for s in v] for k, v in raw_data.items()}

    if not cmd:
        return res

    def __dict2cmdlist(param_value):
        cmdlsts = []
        cmdlst = []
        for target in param_value:
            if not isinstance(target, dict):
                continue
            for k, v in target.items():
                cmdlst.append(f"--{k}")
                if v is not None:
                    if isinstance(v, str):
                        cmdlst.extend(v.split())
                    else:
                        cmdlst.append(v)
        cmdlsts.extend(cmdlst)
        return cmdlsts

    res_cmd = {key: __dict2cmdlist(value) for key, value in res.items()}
    return res_cmd


class ListHandler(logging.Handler):
    # Extract inference log, the regular expression is universal.
    # Just pass the pattern you want.
    def __init__(self, pattern):
        super().__init__()
        self.log_capture = []
        self.pattern = pattern
    
    def emit(self, record):
        log_entry = self.format(record)
        if re.search(self.pattern, log_entry, re.DOTALL):
            self.log_capture.append(log_entry)


def setup_logger(pattern):
    # Set the logger and the handler.
    # Different tasks will not form interference, feel relieved to use. 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = ListHandler(pattern)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return handler, handler.log_capture


def run_cmd(cmd_strlist):
    return subprocess.run(cmd_strlist).returncode

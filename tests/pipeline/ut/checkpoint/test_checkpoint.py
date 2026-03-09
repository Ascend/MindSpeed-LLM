# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests of Checkpoint"""

import sys
import os
import shutil
from pathlib import Path
import logging
import re
import math
import pytest
from mindspeed_llm import megatron_adaptor
from tests.test_tools.utils import create_testconfig, weight_compare, run_cmd, weight_compare_hash


BASE_DIR = Path(__file__).absolute().parents[4]
CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")


class TestCheckpoint(object):
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    test_config_cmd = create_testconfig(Path(__file__).with_suffix(".json"), cmd=True)

    def test_deepseek2_hf2mcore_tp1pp4ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_hf2mcore_tp1pp4ep8'])
        assert exit_code == 0
        base_hash = self.test_config['test_deepseek2_hf2mcore_tp1pp4ep8'][1]
        save_dir = self.test_config['test_deepseek2_hf2mcore_tp1pp4ep8'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")


    def test_deepseek2_mcore2hf_tp1pp4ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_mcore2hf_tp1pp4ep8'])
        assert exit_code == 0
        base_hash = self.test_config['test_deepseek2_mcore2hf_tp1pp4ep8'][1]
        load_dir = self.test_config['test_deepseek2_mcore2hf_tp1pp4ep8'][0]['load-dir']
        save_dir = os.path.join(self.test_config['test_deepseek2_mcore2hf_tp1pp4ep8'][0]['save-dir'], 'mg2hf')
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(load_dir)
        shutil.rmtree(save_dir)

    def test_qwen25_hf2mcore_tp4pp2dpp(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen25_hf2mcore_tp4pp2dpp'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen25_hf2mcore_tp4pp2dpp'][1]
        save_dir = self.test_config['test_qwen25_hf2mcore_tp4pp2dpp'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")

    def test_qwen25_mcore2hf_tp4pp2dpp(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen25_mcore2hf_tp4pp2dpp'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen25_mcore2hf_tp4pp2dpp'][1]
        load_dir = self.test_config['test_qwen25_mcore2hf_tp4pp2dpp'][0]['load-dir']
        save_dir = os.path.join(self.test_config['test_qwen25_mcore2hf_tp4pp2dpp'][0]['save-dir'], 'mg2hf')
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(load_dir)
        shutil.rmtree(save_dir)

    def test_llama3_noop_layer_hf2mg(self):
        """
        Test case for nooplayer.
        """
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_llama3_noop_layer_hf2mg'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama3_noop_layer_hf2mg'][1]
        save_dir = self.test_config['test_llama3_noop_layer_hf2mg'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)

    def test_llama2_merge_lora2mg(self):
        """
        Test case for merge lora and base.
        """
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_llama2_merge_lora2mg'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama2_merge_lora2mg'][1]
        save_dir = self.test_config['test_llama2_merge_lora2mg'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)

    def test_deepseek2_lite_hf2mcore_tp1pp1ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_lite_hf2mcore_tp1pp1ep8'])
        assert exit_code == 0
        base_hash = self.test_config['test_deepseek2_lite_hf2mcore_tp1pp1ep8'][1]
        save_dir = self.test_config['test_deepseek2_lite_hf2mcore_tp1pp1ep8'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_deepseek2_lite_mcore2hf_tp1pp1ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_deepseek2_lite_mcore2hf_tp1pp1ep8'])
        assert exit_code == 0
        base_hash = self.test_config['test_deepseek2_lite_mcore2hf_tp1pp1ep8'][1]
        save_dir = os.path.join(self.test_config['test_deepseek2_lite_mcore2hf_tp1pp1ep8'][0]['save-dir'], 'mg2hf')
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)


    def test_llama2_lora2mcore_tp1pp1(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_llama2_lora2mcore_tp1pp1'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama2_lora2mcore_tp1pp1'][1]
        save_dir = self.test_config['test_llama2_lora2mcore_tp1pp1'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)


    def test_qwen2_moe_hf2mcore_tp1pp2ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_hf2mcore_tp1pp2ep2'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen2_moe_hf2mcore_tp1pp2ep2'][1]
        save_dir = self.test_config['test_qwen2_moe_hf2mcore_tp1pp2ep2'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")


    def test_qwen2_moe_mcore2hf_tp1pp2ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPT_PYPATH] + self.test_config_cmd['test_qwen2_moe_mcore2hf_tp1pp2ep2'])
        assert exit_code == 0
        base_hash = self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][1]
        load_dir = self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][0]['load-dir']
        save_dir = os.path.join(self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][0]['save-dir'], 'mg2hf')
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(load_dir)
        shutil.rmtree(save_dir)
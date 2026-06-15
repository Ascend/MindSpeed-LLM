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

import os
import shutil
from pathlib import Path
from tests.test_tools.utils import create_testconfig, run_cmd, weight_compare_hash, compare_safetensors_weights


BASE_DIR = Path(__file__).absolute().parents[4]
CKPT_PYPATH = os.path.join(BASE_DIR, "convert_ckpt.py")
CKPTV2_PYPATH = os.path.join(BASE_DIR, "convert_ckpt_v2.py")


class TestCheckpoint:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    test_config_cmd = create_testconfig(Path(__file__).with_suffix(".json"), cmd=True)

    @staticmethod
    def _dict_to_cmdlist(param_dict):
        cmdlst = []
        for k, v in param_dict.items():
            if k.startswith('_'):
                continue
            cmdlst.append(f"--{k}")
            if v is not None:
                if isinstance(v, str):
                    cmdlst.extend(v.split())
                else:
                    cmdlst.append(v)
        return cmdlst

    @staticmethod
    def _build_roundtrip_cmd(mcore2hf_param, mg_dir, roundtrip_dir):
        """Reuse mcore2hf param dict, overriding save-dir and load-dir."""
        param = dict(mcore2hf_param)
        param['save-dir'] = roundtrip_dir
        param['load-dir'] = mg_dir
        return TestCheckpoint._dict_to_cmdlist(param)

    def test_deepseek2_hf2mcore_tp1pp4ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        hf_dir = self.test_config['test_deepseek2_hf2mcore_tp1pp4ep8'][0]['load-dir']
        mg_dir = self.test_config['test_deepseek2_hf2mcore_tp1pp4ep8'][0]['save-dir']

        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_deepseek2_hf2mcore_tp1pp4ep8'])
        assert exit_code == 0

        roundtrip_dir = mg_dir.rstrip('/') + '-roundtrip'
        mcore2hf_param = self.test_config['test_deepseek2_mcore2hf_tp1pp4ep8'][0]
        exit_code = run_cmd(
            ["python3", CKPTV2_PYPATH] + self._build_roundtrip_cmd(mcore2hf_param, mg_dir, roundtrip_dir)
        )
        assert exit_code == 0

        assert compare_safetensors_weights(roundtrip_dir, hf_dir)

        if os.path.exists(roundtrip_dir):
            shutil.rmtree(roundtrip_dir)

    def test_deepseek2_mcore2hf_tp1pp4ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        param = self.test_config['test_deepseek2_mcore2hf_tp1pp4ep8'][0]
        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_deepseek2_mcore2hf_tp1pp4ep8'])
        assert exit_code == 0
        save_dir = param['save-dir']
        _reference_dir = param.get('_reference_dir')
        if _reference_dir:
            assert compare_safetensors_weights(save_dir, _reference_dir)
        else:
            base_hash = self.test_config['test_deepseek2_mcore2hf_tp1pp4ep8'][1]
            assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)

    def test_qwen25_hf2mcore_tp4pp2dpp(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        hf_dir = self.test_config['test_qwen25_hf2mcore_tp4pp2dpp'][0]['load-dir']
        mg_dir = self.test_config['test_qwen25_hf2mcore_tp4pp2dpp'][0]['save-dir']

        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_qwen25_hf2mcore_tp4pp2dpp'])
        assert exit_code == 0

        roundtrip_dir = mg_dir.rstrip('/') + '-roundtrip'
        mcore2hf_param = self.test_config['test_qwen25_mcore2hf_tp4pp2dpp'][0]
        exit_code = run_cmd(
            ["python3", CKPTV2_PYPATH] + self._build_roundtrip_cmd(mcore2hf_param, mg_dir, roundtrip_dir)
        )
        assert exit_code == 0

        assert compare_safetensors_weights(roundtrip_dir, hf_dir)

        if os.path.exists(roundtrip_dir):
            shutil.rmtree(roundtrip_dir)

    def test_qwen25_mcore2hf_tp4pp2dpp(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        param = self.test_config['test_qwen25_mcore2hf_tp4pp2dpp'][0]
        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_qwen25_mcore2hf_tp4pp2dpp'])
        assert exit_code == 0
        save_dir = param['save-dir']
        _reference_dir = param.get('_reference_dir')
        if _reference_dir:
            assert compare_safetensors_weights(save_dir, _reference_dir)
        else:
            base_hash = self.test_config['test_qwen25_mcore2hf_tp4pp2dpp'][1]
            assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)

    def test_llama3_noop_layer_hf2mg(self):
        """
        Test case for nooplayer.
        noop-layers are discarded in hf2mg and cannot be recovered in mg2hf,
        so round-trip comparison is not applicable. Use md5 validation instead.
        """
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_llama3_noop_layer_hf2mg'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama3_noop_layer_hf2mg'][1]
        save_dir = self.test_config['test_llama3_noop_layer_hf2mg'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "pt")
        shutil.rmtree(save_dir)

    def test_llama2_merge_lora2hf_tp2pp4(self):
        """
        Test case for merge lora and base, output in HF format.
        """
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_llama2_merge_lora2hf_tp2pp4'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama2_merge_lora2hf_tp2pp4'][1]
        save_dir = self.test_config['test_llama2_merge_lora2hf_tp2pp4'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)

    def test_deepseek2_lite_hf2mcore_tp1pp1ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        hf_dir = self.test_config['test_deepseek2_lite_hf2mcore_tp1pp1ep8'][0]['load-dir']
        mg_dir = self.test_config['test_deepseek2_lite_hf2mcore_tp1pp1ep8'][0]['save-dir']

        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_deepseek2_lite_hf2mcore_tp1pp1ep8'])
        assert exit_code == 0

        roundtrip_dir = mg_dir.rstrip('/') + '-roundtrip'
        mcore2hf_param = self.test_config['test_deepseek2_lite_mcore2hf_tp1pp1ep8'][0]
        exit_code = run_cmd(
            ["python3", CKPTV2_PYPATH] + self._build_roundtrip_cmd(mcore2hf_param, mg_dir, roundtrip_dir)
        )
        assert exit_code == 0

        assert compare_safetensors_weights(roundtrip_dir, hf_dir)

        if os.path.exists(roundtrip_dir):
            shutil.rmtree(roundtrip_dir)

    def test_deepseek2_lite_mcore2hf_tp1pp1ep8(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        param = self.test_config['test_deepseek2_lite_mcore2hf_tp1pp1ep8'][0]
        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_deepseek2_lite_mcore2hf_tp1pp1ep8'])
        assert exit_code == 0
        save_dir = param['save-dir']
        _reference_dir = param.get('_reference_dir')
        if _reference_dir:
            assert compare_safetensors_weights(save_dir, _reference_dir)
        else:
            base_hash = self.test_config['test_deepseek2_lite_mcore2hf_tp1pp1ep8'][1]
            assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)

    def test_llama2_lora2hf_tp1pp1(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_llama2_lora2hf_tp1pp1'])
        assert exit_code == 0
        base_hash = self.test_config['test_llama2_lora2hf_tp1pp1'][1]
        save_dir = self.test_config['test_llama2_lora2hf_tp1pp1'][0]['save-dir']
        assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)

    def test_qwen2_moe_hf2mcore_tp1pp2ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        hf_dir = self.test_config['test_qwen2_moe_hf2mcore_tp1pp2ep2'][0]['load-dir']
        mg_dir = self.test_config['test_qwen2_moe_hf2mcore_tp1pp2ep2'][0]['save-dir']

        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_qwen2_moe_hf2mcore_tp1pp2ep2'])
        assert exit_code == 0

        roundtrip_dir = mg_dir.rstrip('/') + '-roundtrip'
        mcore2hf_param = self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][0]
        exit_code = run_cmd(
            ["python3", CKPTV2_PYPATH] + self._build_roundtrip_cmd(mcore2hf_param, mg_dir, roundtrip_dir)
        )
        assert exit_code == 0

        assert compare_safetensors_weights(roundtrip_dir, hf_dir)

        if os.path.exists(roundtrip_dir):
            shutil.rmtree(roundtrip_dir)

    def test_qwen2_moe_mcore2hf_tp1pp2ep2(self):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        param = self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][0]
        exit_code = run_cmd(["python3", CKPTV2_PYPATH] + self.test_config_cmd['test_qwen2_moe_mcore2hf_tp1pp2ep2'])
        assert exit_code == 0
        save_dir = param['save-dir']
        _reference_dir = param.get('_reference_dir')
        if _reference_dir:
            assert compare_safetensors_weights(save_dir, _reference_dir)
        else:
            base_hash = self.test_config['test_qwen2_moe_mcore2hf_tp1pp2ep2'][1]
            assert weight_compare_hash(save_dir, base_hash, "safetensors")
        shutil.rmtree(save_dir)

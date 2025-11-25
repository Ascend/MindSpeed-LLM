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
"""Coverage of Inference"""

import sys
import os
from io import StringIO
from pathlib import Path
import re
import logging
from torch import distributed as dist
import pytest
from inference import main
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import build_args, create_testconfig, setup_logger
from tests.test_tools import acquire_json

PATTERN = r"MindSpeed-LLM:\n(.*)"


def acquire_context(log_capture):
    # Acquire the final score for evaluation tasks, still universal.
    context_str = log_capture[0]
    context_pattern = r"MindSpeed-LLM:\s*([\s\S]*)"
    match = re.search(context_pattern, context_str)
    if match:
        context = match.group(1)
    else:
        raise ValueError("No matching context found in the provided log.")
    return context


class TestInference(DistributedTest):
    world_size = 8
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("params", test_config["test_llama2_mcore_prompt_greedy_search"])
    def test_llama2_mcore_greedy_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=============== llama2 mcore prompt greedy search =============")
            print(log_capture)
            context = acquire_context(log_capture)
            assert [context] == [
                "I'm doing well, thanks.\nI'm doing well, thanks. I'm doing well, thanks. I'm doing"
            ], "forward pass has been changed, check it!"
    
    @pytest.mark.parametrize("params", test_config["test_llama2_mcore_do_sample_search"])
    def test_llama2_mcore_do_sample_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        if dist.get_rank() == 0:
            print("=============== llama2 mcore do_sample search =============")
            print(log_capture)
            context = acquire_context(log_capture)
            assert [context] == [
                "I have been reading this thread for a while and decided to finally join and say hello. I am a 23 year old female from the"
            ], "forward pass has been changed, check it!"
            
    @pytest.mark.parametrize("params", test_config["test_llama2_mcore_do_beam_search"])
    def test_llama2_mcore_do_beam_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        if dist.get_rank() == 0:
            print("=============== llama2 mcore beam search =============")
            print(log_capture)
            context = acquire_context(log_capture)
            # 减层
            assert [context] == ['[INST] how are you? [/INST]\n[INST] how are you? [/INST] [INST] how are you'
                                 ], f"forward pass has been changed to {[context]}, check it!"
    
    @pytest.mark.parametrize("params", test_config["test_llama3_mcore_greedy_search_with_tp2pp4sp"])
    def test_llama3_mcore_greedy_search_with_tp2pp4sp(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["HCCL_DETERMINISTIC"] = "True"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=============== llama3 mcore greedy search tp2pp4sp =============")
            print(log_capture)
            context = acquire_context(log_capture)
            assert [context] == [
                'I hope you are well. I am fine. I am writing to you because I have a problem. I am a student in the university. I am studying in the university of the Philippines. I am studying in the university of the Philippines. I am'
            ], f"forward pass has been changed to {[context]}, check it!"
    
    @pytest.mark.parametrize("params", test_config["test_llama3_mcore_beam_search_with_sampling_tp2pp4sp"])
    def test_llama3_mcore_beam_search_with_sampling_tp2pp4sp(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["HCCL_DETERMINISTIC"] = "True"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        if dist.get_rank() == 0:
            print("=============== llama3 mcore beam search with sampling =============")
            print(log_capture)
            context = acquire_context(log_capture)
            assert [context] == [
                'I am well.\nI have been working on a new project for the past few months. It is a book about the history of the world. I have been working on it for the past few months. I have been working on it for the past few'
            ], f"forward pass has been changed to {[context]}, check it!"
        
    @pytest.mark.parametrize("params", test_config["test_llama3_chat"])
    def test_llama3_chat(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["HCCL_DETERMINISTIC"] = "True"
        
        sys.stdin = StringIO("how are you?\nexit\n")
        if dist.get_rank() == 0:
            print("=============== llama3 mcore chat =============")
            
        main()

    @pytest.mark.parametrize("params", test_config["test_lingmini2_mcore_greedy_search"])
    def test_lingmini2_mcore_greedy_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=============== lingmini2 mcore greedy search only for coverage =============")
            print(log_capture)
            context = acquire_context(log_capture)


    @pytest.mark.parametrize("params", test_config["test_minicpm3_mcore_greedy_search"])
    def test_minicpm3_mcore_greedy_search(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        if dist.get_rank() == 0:
            print("=============== minicpm3 mcore greedy search only for coverage =============")
            print(log_capture)
            context = acquire_context(log_capture)

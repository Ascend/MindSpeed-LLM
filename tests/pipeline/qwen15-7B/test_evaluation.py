import sys
import os
from pathlib import Path
import logging
import re
import pytest
import math
import torch.distributed as dist
from evaluation import main
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import build_args, create_testconfig, setup_logger
from tests.ut.evaluation.test_evaluate import acquire_score


PATTERN = r"acc = (.*)"


class TestEvaluate(DistributedTest):
    world_size = 8
    cur_dir = Path(__file__).parent
    json_file = next(cur_dir.glob("*.json"), None)
    test_config = create_testconfig(json_file)

    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_mmlu_evaluate"])
    def test_qwen_mmlu_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=================== qwen15_7B MMLU score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.6374, abs_tol=1e-2), expected_score


    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_cmmlu_evaluate"])
    def test_qwen_mmlu_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=================== qwen15_7B CMMLU score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.656716, abs_tol=1e-2), expected_score


    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_boolq_evaluate"])
    def test_qwen_boolq_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=================== qwen15_7B BOOLQ score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.8, abs_tol=1e-2), expected_score


    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_ceval_evaluate"])
    def test_qwen_ceval_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=================== qwen15_7B CEVAL score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.6474, abs_tol=1e-2), expected_score


    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_bbh_evaluate"])
    def test_qwen_bbh_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=================== qwen15_7B BBH score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.6667, abs_tol=1e-2), expected_score


    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_gsm8k_evaluate"])
    def test_qwen_gsm8k_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=================== qwen15_7B GSM8K score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.7179487, abs_tol=1e-2), expected_score


    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_agi_evaluate"])
    def test_qwen_agi_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=================== qwen15_7B AGI score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.75, abs_tol=1e-2), expected_score


    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_mmlu_ppl_evaluate"])
    def test_qwen_mmlu_ppl_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()

        if dist.get_rank() == 0:
            print("=================== qwen15_7B MMLU_PPL score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.25146198, abs_tol=1e-2), expected_score
    
    
    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_hellaswag_evaluate"])
    def test_qwen_hellaswag_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)
        
        main()
        
        if dist.get_rank() == 0:
            print("=================== qwen15_7B HELLASWAG score ===============")
            print(log_capture)
            
            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 1.0, abs_tol=1e-2), expected_score


    @pytest.mark.parametrize("params", test_config["test_qwen15_7B_human_eval_evaluate"])
    def test_qwen_human_eval_evaluate(self, build_args, params):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if dist.get_rank() == 0:
            handler, log_capture = setup_logger(PATTERN)

        main()
        
        if dist.get_rank() == 0:
            print("=================== qwen15_7B HUMAN_EVAL score ===============")
            print(log_capture)

            expected_score = acquire_score(log_capture)
            assert math.isclose(expected_score, 0.0, abs_tol=1e-2), expected_score

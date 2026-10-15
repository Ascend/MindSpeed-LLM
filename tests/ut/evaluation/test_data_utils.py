# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
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
"""Unit tests for evaluation data distribution utilities."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pytest

from mindspeed_llm.tasks.evaluation import utils as evaluation_utils
from tests.test_tools.utils import create_testconfig


def _assert_args_include(args, expected):
    assert {name: getattr(args, name) for name in expected} == expected


class TestEvaluationDataUtils:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def test_add_text_generate_args_registers_defaults(self):
        parser = ArgumentParser()

        configured_parser = evaluation_utils.add_text_generate_args(parser)
        args = parser.parse_args([])

        assert configured_parser is parser
        _assert_args_include(
            args,
            {
                "task_data_path": [],
                "temperature": 0.5,
                "evaluation_batch_size": 1,
                "greedy": False,
                "top_p": 0.9,
                "top_k": 0,
                "max_new_tokens": 128,
                "task": [],
                "instruction_template": "",
                "no_chat_template": False,
                "use_kv_cache": False,
                "hf_chat_template": False,
                "eval_language": "en",
                "max_eval_samples": None,
                "broadcast": False,
                "alternative_prompt": False,
                "origin_postprocess": False,
                "chain_of_thought": False,
            },
        )

    def test_add_text_generate_args_parses_explicit_values(self):
        parser = evaluation_utils.add_text_generate_args(ArgumentParser())

        args = parser.parse_args(
            [
                "--task-data-path",
                "dataset-a.json",
                "dataset-b.json",
                "--temperature",
                "0.7",
                "--evaluation-batch-size",
                "4",
                "--greedy",
                "--top-p",
                "0.8",
                "--top-k",
                "10",
                "--max-new-tokens",
                "64",
                "--task",
                "mmlu",
                "gsm8k",
                "--instruction-template",
                "question-answer",
                "--no-chat-template",
                "--use-kv-cache",
                "--hf-chat-template",
                "--eval-language",
                "zh",
                "--max-eval-samples",
                "32",
                "--broadcast",
                "--alternative-prompt",
                "--origin-postprocess",
                "--chain-of-thought",
            ]
        )

        _assert_args_include(
            args,
            {
                "task_data_path": ["dataset-a.json", "dataset-b.json"],
                "temperature": 0.7,
                "evaluation_batch_size": 4,
                "greedy": True,
                "top_p": 0.8,
                "top_k": 10,
                "max_new_tokens": 64,
                "task": ["mmlu", "gsm8k"],
                "instruction_template": "question-answer",
                "no_chat_template": True,
                "use_kv_cache": True,
                "hf_chat_template": True,
                "eval_language": "zh",
                "max_eval_samples": 32,
                "broadcast": True,
                "alternative_prompt": True,
                "origin_postprocess": True,
                "chain_of_thought": True,
            },
        )

    def test_compute_and_conjugate_data_parallel_domains(self):
        domains = evaluation_utils.compute_all_dp_domains(
            world_size=8,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )

        assert domains == [[0, 2], [1, 3], [4, 6], [5, 7]]
        assert evaluation_utils.conjugate_data_parallel_domain(domains) == [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
        ]

    @pytest.mark.parametrize(("rank", "expected_index"), test_config["test_get_index_by_rank"])
    def test_get_index_by_rank(self, rank, expected_index):
        groups = [[0, 1, 4, 5], [2, 3, 6, 7]]

        assert evaluation_utils.get_index_by_rank(rank, groups) == expected_index

    def test_get_index_by_rank_rejects_unknown_rank(self):
        with pytest.raises(ValueError, match="Invalid rank 8"):
            evaluation_utils.get_index_by_rank(8, [[0, 1, 4, 5], [2, 3, 6, 7]])

    def test_split_and_align_dataframe_for_uneven_workload(self):
        data = pd.DataFrame({"sample": list("abcde")})

        parts = evaluation_utils.split_dataframe(data, 3)

        assert [len(part) for part in parts] == [2, 2, 1]
        pd.testing.assert_frame_equal(pd.concat(parts), data)

        aligned_parts, align_start_rank = evaluation_utils.alignment_data_length(parts)

        assert [len(part) for part in aligned_parts] == [2, 2, 2]
        assert aligned_parts[-1]["sample"].tolist() == ["e", "a"]
        assert align_start_rank == 2

    def test_dataframe_alignment_leaves_even_workload_unchanged(self):
        parts = evaluation_utils.split_dataframe(
            pd.DataFrame({"sample": list("abcdef")}),
            3,
        )

        aligned_parts, align_start_rank = evaluation_utils.alignment_data_length(parts)

        assert [part["sample"].tolist() for part in aligned_parts] == [
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
        ]
        assert align_start_rank == -1

    def test_split_and_align_list_for_uneven_workload(self):
        parts = evaluation_utils.split_list(list("abcde"), 3)

        assert parts == [["a", "b"], ["c", "d"], ["e"]]

        aligned_parts, align_start_rank = evaluation_utils.alignment_list_data_length(parts)

        assert aligned_parts == [["a", "b"], ["c", "d"], ["e", "a"]]
        assert align_start_rank == 2

    def test_list_alignment_leaves_even_workload_unchanged(self):
        parts = evaluation_utils.split_list(list("abcdef"), 3)

        aligned_parts, align_start_rank = evaluation_utils.alignment_list_data_length(parts)

        assert aligned_parts == [["a", "b"], ["c", "d"], ["e", "f"]]
        assert align_start_rank == -1

    @pytest.mark.parametrize(
        ("rank", "samples", "expected_samples", "expected_align_start"),
        test_config["test_get_final_dataset_selects_dataframe_for_rank"],
    )
    def test_get_final_dataset_selects_dataframe_for_rank(
        self, monkeypatch, rank, samples, expected_samples, expected_align_start
    ):
        monkeypatch.setattr(evaluation_utils.dist, "get_rank", lambda: rank)
        data = pd.DataFrame({"sample": samples})

        rank_data, domains, align_start_rank = evaluation_utils.get_final_dataset(
            data,
            world_size=8,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )

        assert rank_data["sample"].tolist() == expected_samples
        assert domains == [[0, 2], [1, 3], [4, 6], [5, 7]]
        assert align_start_rank == expected_align_start

    def test_get_final_list_dataset_selects_and_aligns_data_for_rank(self, monkeypatch):
        monkeypatch.setattr(evaluation_utils.dist, "get_rank", lambda: 3)

        rank_data, domains, align_start_rank = evaluation_utils.get_final_list_dataset(
            list("abcde"),
            world_size=8,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
        )

        assert rank_data == ["d", "e", "a"]
        assert domains == [[0, 2], [1, 3], [4, 6], [5, 7]]
        assert align_start_rank == 1

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
"""Unit tests for HumanEval result processing and persistence helpers."""

import multiprocessing
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from mindspeed_llm.tasks.evaluation.eval_utils import human_utils
from tests.test_tools.utils import create_testconfig


class TestHumanEvalUtils:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("method_name", test_config["test_write_only_string_io_rejects_reads"])
    def test_write_only_string_io_rejects_reads(self, method_name):
        stream = human_utils.WriteOnlyStringIO("hidden")

        with pytest.raises(IOError):
            getattr(stream, method_name)()

        assert not stream.readable()

    @pytest.mark.parametrize(
        ("text", "expected"),
        test_config["test_humaneval_postprocess_extracts_first_fenced_block"],
    )
    def test_humaneval_postprocess_extracts_first_fenced_block(self, text, expected):
        assert human_utils.humaneval_postprocess(text) == expected

    @pytest.mark.parametrize(
        ("completion", "expected_passed", "expected_result"),
        test_config["test_check_correctness_isolates_pass_fail_and_timeout"],
    )
    def test_check_correctness_isolates_pass_fail_and_timeout(
        self, completion, expected_passed, expected_result, monkeypatch
    ):
        problem = {
            "task_id": "HumanEval/0",
            "prompt": "def add(left, right):\n",
            "test": "def check(candidate):\n    assert candidate(20, 22) == 42",
            "entry_point": "add",
        }

        monkeypatch.setattr(human_utils, "multiprocessing", multiprocessing.get_context("fork"))

        result = human_utils.check_correctness(problem, completion, timeout=1.0, completion_id=7)

        assert result["task_id"] == "HumanEval/0"
        assert result["completion_id"] == 7
        assert result["passed"] is expected_passed
        assert expected_result in result["result"]

    def test_time_limit_clears_alarm_after_timeout(self, monkeypatch):
        timer_calls = []
        signal_handlers = {}

        def record_signal_handler(signum, handler):
            signal_handlers[signum] = handler

        fake_signal = types.SimpleNamespace(
            ITIMER_REAL=0,
            SIGALRM=14,
            setitimer=lambda timer, seconds: timer_calls.append((timer, seconds)),
            signal=record_signal_handler,
        )
        monkeypatch.setattr(human_utils, "signal", fake_signal)

        with pytest.raises(human_utils.TimeoutException, match="Timed out"):
            with human_utils.time_limit(0.25):
                signal_handlers[fake_signal.SIGALRM](fake_signal.SIGALRM, None)

        assert timer_calls == [(fake_signal.ITIMER_REAL, 0.25), (fake_signal.ITIMER_REAL, 0)]

    @pytest.mark.parametrize(
        "maximum_memory_bytes",
        test_config["test_reliability_guard_disables_unsafe_operations"],
    )
    def test_reliability_guard_disables_unsafe_operations(self, maximum_memory_bytes, monkeypatch):
        limit_calls = []
        fake_resource = types.SimpleNamespace(
            RLIMIT_AS="address_space",
            RLIMIT_DATA="data",
            RLIMIT_STACK="stack",
            setrlimit=lambda limit, value: limit_calls.append((limit, value)),
        )
        fake_platform = types.SimpleNamespace(uname=lambda: types.SimpleNamespace(system="Linux"))
        disable_calls = []
        fake_faulthandler = types.SimpleNamespace(disable=lambda: disable_calls.append(True))
        fake_builtins = types.SimpleNamespace(exit=object(), quit=object())

        unsafe_os_methods = (
            "kill",
            "system",
            "putenv",
            "remove",
            "removedirs",
            "rmdir",
            "fchdir",
            "setuid",
            "fork",
            "forkpty",
            "killpg",
            "rename",
            "renames",
            "truncate",
            "replace",
            "unlink",
            "fchmod",
            "fchown",
            "chmod",
            "chown",
            "chroot",
            "lchflags",
            "lchmod",
            "lchown",
            "getcwd",
            "chdir",
        )
        fake_os = types.SimpleNamespace(environ={})
        for method_name in unsafe_os_methods:
            setattr(fake_os, method_name, object())

        unsafe_shutil_methods = ("rmtree", "move", "chown")
        fake_shutil = types.SimpleNamespace(**{method_name: object() for method_name in unsafe_shutil_methods})
        fake_subprocess = types.SimpleNamespace(Popen=object())
        blocked_modules = ("ipdb", "joblib", "resource", "psutil", "tkinter")
        fake_sys = types.SimpleNamespace(modules={module_name: object() for module_name in blocked_modules})
        fake_dunder_builtins = {"help": object()}

        monkeypatch.setattr(human_utils, "resource", fake_resource)
        monkeypatch.setattr(human_utils, "platform", fake_platform)
        monkeypatch.setattr(human_utils, "faulthandler", fake_faulthandler)
        monkeypatch.setattr(human_utils, "builtins", fake_builtins)
        monkeypatch.setattr(human_utils, "os", fake_os)
        monkeypatch.setattr(human_utils, "shutil", fake_shutil)
        monkeypatch.setattr(human_utils, "subprocess", fake_subprocess)
        monkeypatch.setattr(human_utils, "sys", fake_sys)
        monkeypatch.setattr(human_utils, "__builtins__", fake_dunder_builtins)

        human_utils.reliability_guard(maximum_memory_bytes)

        expected_limit = (maximum_memory_bytes, maximum_memory_bytes)
        assert limit_calls == [
            (fake_resource.RLIMIT_AS, expected_limit),
            (fake_resource.RLIMIT_DATA, expected_limit),
            (fake_resource.RLIMIT_STACK, expected_limit),
        ]
        assert disable_calls == [True]
        assert fake_builtins.exit is None and fake_builtins.quit is None
        assert fake_os.environ["OMP_NUM_THREADS"] == "1"
        assert all(getattr(fake_os, method_name) is None for method_name in unsafe_os_methods)
        assert all(getattr(fake_shutil, method_name) is None for method_name in unsafe_shutil_methods)
        assert fake_subprocess.Popen is None
        assert fake_dunder_builtins["help"] is None
        assert all(fake_sys.modules[module_name] is None for module_name in blocked_modules)

    def test_swallow_io_hides_stdout_stderr_and_stdin(self):
        with human_utils.swallow_io():
            print("not visible")
            sys.stderr.write("also hidden")
            with pytest.raises(IOError):
                input()

    def test_chdir_changes_directory_and_restores_it(self, tmp_path):
        original_dir = os.getcwd()

        with human_utils.chdir(tmp_path):
            assert Path.cwd() == tmp_path

        assert os.getcwd() == original_dir

    def test_chdir_restores_directory_after_exception(self, tmp_path):
        original_dir = os.getcwd()

        with pytest.raises(RuntimeError, match="stop"):
            with human_utils.chdir(tmp_path):
                raise RuntimeError("stop")

        assert os.getcwd() == original_dir

    def test_chdir_dot_leaves_directory_unchanged(self):
        original_dir = os.getcwd()

        with human_utils.chdir("."):
            assert os.getcwd() == original_dir

        assert os.getcwd() == original_dir

    @pytest.mark.parametrize("suffix", test_config["test_write_and_stream_jsonl_support_plain_and_gzip"])
    def test_write_and_stream_jsonl_support_plain_and_gzip(self, tmp_path, suffix):
        output_file = tmp_path / f"samples{suffix}"
        first_batch = [{"task_id": "one", "value": 1}, {"task_id": "two", "value": 2}]
        second_batch = [{"task_id": "three", "value": 3}]

        human_utils.write_jsonl(str(output_file), first_batch)
        human_utils.write_jsonl(str(output_file), second_batch, append=True)

        assert list(human_utils.stream_jsonl(str(output_file))) == first_batch + second_batch

    def test_stream_jsonl_ignores_blank_lines(self, tmp_path):
        sample_file = tmp_path / "samples.jsonl"
        sample_file.write_text('\n{"task_id": "one"}\n   \n', encoding="utf-8")

        assert list(human_utils.stream_jsonl(str(sample_file))) == [{"task_id": "one"}]

    def test_estimate_pass_at_k_accepts_scalar_sample_count(self):
        result = human_utils.estimate_pass_at_k(4, np.array([0, 1, 4]), 1)

        np.testing.assert_allclose(result, np.array([0.0, 0.25, 1.0]))

    def test_estimate_pass_at_k_accepts_per_problem_counts(self):
        result = human_utils.estimate_pass_at_k(np.array([2, 3]), np.array([1, 1]), 2)

        np.testing.assert_allclose(result, np.array([1.0, 2.0 / 3.0]))

    def test_estimate_pass_at_k_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match=r"len\(num_samples\) must equal to len\(num_correct\)"):
            human_utils.estimate_pass_at_k([2], [1, 0], 1)

    def test_evaluate_functional_correctness_aggregates_and_persists_results(self, tmp_path, monkeypatch):
        sample_file = tmp_path / "samples.jsonl"
        samples = [
            {"task_id": "HumanEval/0", "completion": "correct"},
            {"task_id": "HumanEval/0", "completion": "incorrect"},
        ]
        human_utils.write_jsonl(str(sample_file), samples)

        def fake_check_correctness(problem, completion, timeout, completion_id):
            del problem, timeout
            passed = completion == "correct"
            return {
                "task_id": "HumanEval/0",
                "passed": passed,
                "result": "passed" if passed else "failed: assertion",
                "completion_id": completion_id,
            }

        monkeypatch.setattr(human_utils, "check_correctness", fake_check_correctness)

        scores = human_utils.evaluate_functional_correctness(
            str(sample_file),
            {"HumanEval/0": {"task_id": "HumanEval/0"}},
            k=[1, 2],
            n_workers=2,
            timeout=0.1,
        )

        assert scores == pytest.approx({"pass@1": 0.5, "pass@2": 1.0})
        saved_results = list(human_utils.stream_jsonl(f"{sample_file}_results.jsonl"))
        assert [item["passed"] for item in saved_results] == [True, False]
        assert [item["result"] for item in saved_results] == ["passed", "failed: assertion"]

    def test_get_score_formats_metrics_and_details(self, monkeypatch):
        predictions = ["completion one", "completion two"]
        references = ["HumanEval/0", "HumanEval/1"]
        test_set = [{"prompt": "prompt zero"}, {"prompt": "prompt one"}]
        problems = {
            "HumanEval/0": {"task_id": "HumanEval/0"},
            "HumanEval/1": {"task_id": "HumanEval/1"},
        }

        def fake_evaluate(sample_file, problem_file):
            assert problem_file is problems
            samples = list(human_utils.stream_jsonl(sample_file))
            results = [
                {**sample, "passed": index == 0, "result": "passed" if index == 0 else "failed"}
                for index, sample in enumerate(samples)
            ]
            human_utils.write_jsonl(f"{sample_file}_results.jsonl", results)
            return {"pass@1": 0.5}

        monkeypatch.setattr(human_utils, "evaluate_functional_correctness", fake_evaluate)

        score = human_utils.get_score(predictions, references, test_set, problems)

        assert score["humaneval_pass@1"] == 50.0
        assert score["details"]["0"]["prompt"] == "prompt zero"
        assert score["details"]["0"]["is_correct"] is True
        assert score["details"]["1"]["prompt"] == "prompt one"
        assert score["details"]["1"]["is_correct"] is False

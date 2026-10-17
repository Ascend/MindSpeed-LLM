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
"""Test of the RerunStateMachine core in megatron_data/megatron_dataset_iter.py."""
# pylint: disable=redefined-outer-name  # pytest fixtures shadow module-level helper names

from collections import defaultdict
from pathlib import Path

import pytest
import torch

import mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_iter as miter
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_iter import (
    EXIT_CODE_FAILED_ON_RESULT_VALIDATION,
    EXIT_CODE_RESUME_TO_DISAMBIGUATE,
    Call,
    Caller,
    RerunDataIterator,
    RerunMode,
    RerunState,
    RerunStateMachine,
    destroy_rerun_state_machine,
    get_rerun_state_machine,
    initialize_rerun_state_machine,
)

from tests.test_tools.utils import create_testconfig


@pytest.fixture(autouse=True)
def clean_rerun_state_machine():
    destroy_rerun_state_machine()
    yield
    destroy_rerun_state_machine()


@pytest.fixture
def cpu_cuda(monkeypatch):
    """Stub CUDA/distributed/uname calls so the state machine runs on CPU."""
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: b"cuda-rng-state")
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(miter.os, "uname", lambda: ("Linux", "node0", "", "", "x86_64"), raising=False)
    # single-rank all_reduce leaves the flag tensor unchanged
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor: None, raising=False)
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)  # force CPU: no 'cuda' device in UT
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)


def _validate(machine, result, **kwargs):
    """Single call site so repeated invocations share one validation Caller."""
    machine.validate_result(result=result, **kwargs)


def _make_machine(mode=RerunMode.VALIDATE_RESULTS, **kwargs):
    return RerunStateMachine(mode=mode, **kwargs)


class TestModeAccessors:
    """Rerun mode accessors and implicit initialization."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, operation, mode, expected_mode", test_config["test_mode_accessors"])
    def test_mode_accessors(self, label, operation, mode, expected_mode):
        expected = getattr(RerunMode, expected_mode)
        if operation == "default":
            assert RerunStateMachine().get_mode() == expected, label
        elif operation == "set":
            machine = _make_machine()
            machine.set_mode(getattr(RerunMode, mode))
            assert machine.get_mode() == expected, label
        else:
            machine = get_rerun_state_machine()
            assert isinstance(machine, RerunStateMachine), label
            assert machine.get_mode() == expected, label

    def test_result_rejected_tracker_file_must_be_writable(self, tmp_path):
        _make_machine(result_rejected_tracker_filename=str(tmp_path / "tracker.txt"))
        assert (tmp_path / "tracker.txt").exists()


class TestSanitizeDataIterators:
    """_sanitize_data_iterators input normalization."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, mode, iterators, expected_len, expect_raise",
        test_config["test_sanitize_data_iterators"],
    )
    def test_sanitize_data_iterators(self, label, mode, iterators, expected_len, expect_raise):
        initialize_rerun_state_machine()
        machine = _make_machine(mode=getattr(RerunMode, mode))
        iterator = RerunDataIterator(iter([1]))
        payload = {
            "single": iterator,
            "list_with_none": [None, iterator],
            "none": None,
            "unwrapped": iter([1]),
        }[iterators]

        if expect_raise:
            with pytest.raises(AssertionError, match="not wrapped"):
                machine._sanitize_data_iterators(payload)
            return

        sanitized = machine._sanitize_data_iterators(payload)

        assert len(sanitized) == expected_len, label


class TestShouldRunForwardBackwardDisabled:
    def test_single_pass_then_done(self):
        machine = _make_machine(mode=RerunMode.DISABLED)
        assert machine.should_run_forward_backward(None) is True
        assert machine.state == RerunState.INITIAL_RUN
        assert machine.current_iteration == 0
        assert machine.should_run_forward_backward(None) is False
        assert machine.state == RerunState.NOT_RUNNING_YET

    def test_iteration_counter_advances_per_step(self):
        machine = _make_machine(mode=RerunMode.DISABLED)
        machine.should_run_forward_backward(None)
        machine.should_run_forward_backward(None)
        machine.should_run_forward_backward(None)
        assert machine.current_iteration == 1


class TestShouldRunForwardBackwardValidate:
    def test_initial_run_saves_state_and_advances_iterators(self, cpu_cuda):
        initialize_rerun_state_machine()
        saved = []
        machine = _make_machine(state_save_func=lambda: saved.append({"x": 1}) or {"x": 1})
        iterator = RerunDataIterator(iter([1, 2, 3]))
        next(iterator)

        assert machine.should_run_forward_backward(iterator) is True

        assert machine.state == RerunState.INITIAL_RUN
        assert saved == [{"x": 1}]
        assert machine.saved_state["other_state"] == {"x": 1}
        assert iterator.saved_microbatches == []
        assert machine.rerun_requested is False

    def test_initial_run_without_rejection_finishes_step(self, cpu_cuda):
        machine = _make_machine()
        machine.should_run_forward_backward(None)
        assert machine.should_run_forward_backward(None) is False
        assert machine.state == RerunState.NOT_RUNNING_YET

    def test_rerun_requested_rewinds_iterators_and_restores_state(self, cpu_cuda):
        initialize_rerun_state_machine()
        restored = []
        machine = _make_machine(state_save_func=lambda: {"x": 1}, state_restore_func=restored.append)
        iterator = RerunDataIterator(iter([1, 2]))
        next(iterator)
        machine.should_run_forward_backward(iterator)
        machine.rerun_requested = True

        assert machine.should_run_forward_backward(iterator) is True

        assert machine.state == RerunState.RERUNNING_IN_PLACE
        assert restored == [{"x": 1}]
        assert iterator.replaying is True

    def test_first_rerun_done_without_checkpoint_stays_in_place(self, cpu_cuda):
        machine = _make_machine()
        machine.should_run_forward_backward(None)
        machine.rerun_requested = True
        machine.should_run_forward_backward(None)

        assert machine.should_run_forward_backward(None) is False

        # no checkpoint requested: state machine remains in RERUNNING_IN_PLACE
        assert machine.state == RerunState.RERUNNING_IN_PLACE

    def test_first_rerun_done_with_checkpoint_moves_to_will_rerun(self, cpu_cuda):
        machine = _make_machine()
        machine.should_run_forward_backward(None)
        machine.rerun_requested = True
        machine.should_run_forward_backward(None)
        machine.checkpoint_requested = True

        assert machine.should_run_forward_backward(None) is False

        assert machine.state == RerunState.WILL_RERUN_FROM_CHECKPOINT

    def test_will_rerun_from_checkpoint_enters_second_rerun(self, cpu_cuda):
        machine = _make_machine()
        machine.state = RerunState.WILL_RERUN_FROM_CHECKPOINT
        assert machine.should_run_forward_backward(None) is True
        assert machine.state == RerunState.RERUNNING_FROM_CHECKPOINT

    def test_second_rerun_done_with_restart_again(self, cpu_cuda):
        machine = _make_machine()
        machine.state = RerunState.RERUNNING_FROM_CHECKPOINT
        machine.restart_again_requested = True
        assert machine.should_run_forward_backward(None) is False
        assert machine.state == RerunState.RERUNNING_AGAIN_FROM_CHECKPOINT

    def test_second_rerun_done_with_continue_returns_to_normal(self, cpu_cuda):
        machine = _make_machine()
        machine.state = RerunState.RERUNNING_FROM_CHECKPOINT
        machine.continue_requested = True
        assert machine.should_run_forward_backward(None) is False
        assert machine.state == RerunState.NOT_RUNNING_YET

    def test_second_rerun_done_without_decision_stays(self, cpu_cuda):
        machine = _make_machine()
        machine.state = RerunState.RERUNNING_FROM_CHECKPOINT
        assert machine.should_run_forward_backward(None) is False
        assert machine.state == RerunState.RERUNNING_FROM_CHECKPOINT

    def test_checkpointed_data_iterators_are_restored_on_start(self, cpu_cuda, monkeypatch):
        initialize_rerun_state_machine()
        machine = _make_machine()
        iterator = RerunDataIterator(iter([1, 2, 3]))
        next(iterator)
        checkpoint = iterator.state_dict()
        fresh = RerunDataIterator(iter([1, 2, 3]))
        loaded = []
        monkeypatch.setattr(fresh, "load_state_dict", loaded.append)
        machine.data_iterator_checkpoints = [checkpoint]

        machine.should_run_forward_backward(fresh)

        assert machine.data_iterator_checkpoints is None
        assert loaded == [checkpoint]

    def test_unknown_state_raises(self, cpu_cuda):
        machine = _make_machine()
        machine.state = None
        with pytest.raises(RuntimeError, match="Should not be here"):
            machine.should_run_forward_backward(None)


class TestShouldRunForwardBackwardStatsMode:
    def test_first_rerun_done_reports_stats_and_resets(self, cpu_cuda):
        machine = _make_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
        machine.should_run_forward_backward(None)
        machine.rerun_requested = True
        machine.should_run_forward_backward(None)
        machine.current_iteration = 1  # odd: no reporting this step

        assert machine.should_run_forward_backward(None) is False

        assert machine.state == RerunState.NOT_RUNNING_YET
        assert machine.saved_results == {}

    def test_reporting_interval_logs_stats_without_distributed(self, cpu_cuda):
        machine = _make_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
        machine.state = RerunState.RERUNNING_IN_PLACE
        machine.current_iteration = 2  # REPORTING_INTERVAL_ITERATIONS == 2
        assert machine.should_run_forward_backward(None) is False
        assert machine.state == RerunState.NOT_RUNNING_YET


class TestShouldCheckpointAndExit:
    def test_disabled_mode_never_exits(self):
        machine = _make_machine(mode=RerunMode.DISABLED)
        machine.state = RerunState.RERUNNING_IN_PLACE
        assert machine.should_checkpoint_and_exit() == (False, False, 0)

    def test_stats_mode_never_exits(self):
        machine = _make_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
        machine.state = RerunState.RERUNNING_IN_PLACE
        assert machine.should_checkpoint_and_exit() == (False, False, 0)

    def test_normal_state_continues(self):
        machine = _make_machine()
        assert machine.should_checkpoint_and_exit() == (False, False, 0)

    def test_rerunning_in_place_checkpoints_and_fails(self):
        machine = _make_machine()
        machine.state = RerunState.RERUNNING_IN_PLACE
        assert machine.should_checkpoint_and_exit() == (True, True, EXIT_CODE_FAILED_ON_RESULT_VALIDATION)

    def test_will_rerun_from_checkpoint_checkpoints_and_resumes(self):
        machine = _make_machine()
        machine.state = RerunState.WILL_RERUN_FROM_CHECKPOINT
        assert machine.should_checkpoint_and_exit() == (True, True, EXIT_CODE_RESUME_TO_DISAMBIGUATE)

    def test_rerunning_from_checkpoint_fails_without_new_checkpoint(self):
        machine = _make_machine()
        machine.state = RerunState.RERUNNING_FROM_CHECKPOINT
        assert machine.should_checkpoint_and_exit() == (False, True, EXIT_CODE_FAILED_ON_RESULT_VALIDATION)

    def test_rerunning_again_resumes_without_new_checkpoint(self):
        machine = _make_machine()
        machine.state = RerunState.RERUNNING_AGAIN_FROM_CHECKPOINT
        assert machine.should_checkpoint_and_exit() == (False, True, EXIT_CODE_RESUME_TO_DISAMBIGUATE)


class TestValidateResult:
    def test_disabled_mode_rejection_raises(self, cpu_cuda, tmp_path):
        tracker = tmp_path / "tracker.txt"
        machine = _make_machine(mode=RerunMode.DISABLED, result_rejected_tracker_filename=str(tracker))
        with pytest.raises(RuntimeError, match="Unexpected result"):
            _validate(machine, torch.tensor(1.0), rejection_func=lambda r: True, message="spiky loss")
        assert "rerun_disabled" in tracker.read_text().lower()

    def test_disabled_mode_accepts_good_result(self, cpu_cuda):
        machine = _make_machine(mode=RerunMode.DISABLED)
        assert _validate(machine, torch.tensor(1.0), rejection_func=lambda r: False) is None

    def test_first_iteration_is_skipped(self, cpu_cuda):
        machine = _make_machine()
        machine.state = RerunState.INITIAL_RUN
        machine.current_iteration = 0
        _validate(machine, torch.tensor(1.0), rejection_func=lambda r: True)
        assert machine.rerun_requested is False

    def test_validate_outside_forward_backward_raises(self, cpu_cuda):
        machine = _make_machine()
        machine.current_iteration = 1
        with pytest.raises(AssertionError, match="forward-backward"):
            _validate(machine, torch.tensor(1.0), rejection_func=lambda r: False)

    def _rejected_machine(self, tmp_path, initial=1.0):
        """Machine right after an initial-run rejection."""
        tracker = tmp_path / "tracker.txt"
        machine = _make_machine(result_rejected_tracker_filename=str(tracker))
        machine.current_iteration = 1
        machine.state = RerunState.INITIAL_RUN
        _validate(machine, torch.tensor(initial), rejection_func=lambda r: True, message="spiky loss")
        assert machine.rerun_requested is True
        # emulate the per-pass reset done by should_run_forward_backward
        machine.validation_counts = defaultdict(int)
        return machine, tracker

    def test_initial_run_rejection_requests_rerun(self, cpu_cuda, tmp_path):
        machine, tracker = self._rejected_machine(tmp_path)
        assert machine.failed_validation_call is not None
        assert machine.initial_result.item() == 1.0
        assert "initial_run" in tracker.read_text().lower()

    def test_initial_run_accepts_good_result(self, cpu_cuda):
        machine = _make_machine()
        machine.current_iteration = 1
        machine.state = RerunState.INITIAL_RUN
        _validate(machine, torch.tensor(1.0), rejection_func=lambda r: False)
        assert machine.rerun_requested is False
        assert machine.failed_validation_call is None

    def test_first_rerun_mismatch_flags_transient_error(self, cpu_cuda, tmp_path):
        machine, tracker = self._rejected_machine(tmp_path)
        machine.state = RerunState.RERUNNING_IN_PLACE

        _validate(machine, torch.tensor(2.0), rejection_func=lambda r: False, message="spiky loss")

        assert machine.checkpoint_requested is False
        assert "first_rerun_not_reproducible" in tracker.read_text().lower()

    def test_first_rerun_match_requests_checkpoint(self, cpu_cuda, tmp_path):
        machine, tracker = self._rejected_machine(tmp_path)
        machine.state = RerunState.RERUNNING_IN_PLACE

        _validate(machine, torch.tensor(1.0), rejection_func=lambda r: False, message="spiky loss")

        assert machine.checkpoint_requested is True
        assert machine.suspicious_node == "node0"
        assert machine.suspicious_device == 0
        assert "first_rerun_reproducible" in tracker.read_text().lower()

    def test_first_rerun_mismatch_within_tolerance_requests_checkpoint(self, cpu_cuda, tmp_path):
        machine, _ = self._rejected_machine(tmp_path)
        machine.state = RerunState.RERUNNING_IN_PLACE
        # |2 * (1.0 - 1.1) / (1.0 + 1.1)| ~= 0.095 < 0.1
        _validate(machine, torch.tensor(1.1), rejection_func=lambda r: False, tolerance=0.1)
        assert machine.checkpoint_requested is True

    def test_second_rerun_on_same_gpu_requests_restart_again(self, cpu_cuda, tmp_path):
        machine, _ = self._rejected_machine(tmp_path)
        machine.state = RerunState.RERUNNING_FROM_CHECKPOINT
        machine.suspicious_node = "node0"
        machine.suspicious_device = 0

        _validate(machine, torch.tensor(1.0), rejection_func=lambda r: False)

        assert machine.restart_again_requested is True

    def test_second_rerun_mismatch_flags_persistent_error(self, cpu_cuda, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)  # different GPU
        machine, tracker = self._rejected_machine(tmp_path)
        machine.state = RerunState.RERUNNING_FROM_CHECKPOINT
        machine.suspicious_node = "node0"
        machine.suspicious_device = 0

        _validate(machine, torch.tensor(2.0), rejection_func=lambda r: False, message="spiky loss")

        assert machine.restart_again_requested is False
        assert machine.continue_requested is False
        assert "second_rerun_not_reproducible" in tracker.read_text().lower()

    def test_second_rerun_match_non_fatal_requests_continue(self, cpu_cuda, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        machine, tracker = self._rejected_machine(tmp_path)
        machine.state = RerunState.RERUNNING_FROM_CHECKPOINT
        machine.suspicious_node = "node0"
        machine.suspicious_device = 0

        _validate(machine, torch.tensor(1.0), rejection_func=lambda r: False, fatal=False)

        assert machine.continue_requested is True
        assert "second_rerun_reproducible" in tracker.read_text().lower()

    def test_second_rerun_match_fatal_does_not_continue(self, cpu_cuda, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        machine, _ = self._rejected_machine(tmp_path)
        machine.state = RerunState.RERUNNING_FROM_CHECKPOINT
        machine.suspicious_node = "node0"
        machine.suspicious_device = 0

        _validate(machine, torch.tensor(1.0), rejection_func=lambda r: False, fatal=True)

        assert machine.continue_requested is False

    def test_unmatched_validation_call_is_ignored_during_rerun(self, cpu_cuda, tmp_path):
        machine, _ = self._rejected_machine(tmp_path)
        machine.state = RerunState.RERUNNING_IN_PLACE
        machine.failed_validation_call = Call(caller=Caller(filename="other.py", lineno=1, rank=0), sequence=1)
        _validate(machine, torch.tensor(2.0), rejection_func=lambda r: False)
        assert machine.checkpoint_requested is False


class TestValidateResultStatsMode:
    def test_stats_mode_records_relative_differences(self, cpu_cuda):
        machine = _make_machine(mode=RerunMode.REPORT_DETERMINISM_STATS)
        machine.current_iteration = 1
        machine.state = RerunState.INITIAL_RUN
        _validate(machine, torch.tensor(1.0), rejection_func=lambda r: False)
        assert machine.rerun_requested is True
        assert len(machine.saved_results) == 1

        machine.state = RerunState.RERUNNING_IN_PLACE
        machine.validation_counts = defaultdict(int)
        _validate(machine, torch.tensor(2.0), rejection_func=lambda r: False)

        assert len(machine.stats) == 1
        stats = next(iter(machine.stats.values()))
        assert stats.samples == [pytest.approx(2 / 3)]


class TestIsUnexpectedlyLarge:
    def test_sampling_phase_never_triggers(self):
        machine = _make_machine()
        for value in (1.0, 2.0, 3.0):
            assert machine.is_unexpectedly_large(torch.tensor(value), 10, "loss", num_samples=3) is False
        assert machine.max_values["loss"] == 3.0

    def test_triggers_above_threshold_times_max(self):
        machine = _make_machine()
        machine.is_unexpectedly_large(torch.tensor(3.0), 10, "loss", num_samples=1)
        assert machine.is_unexpectedly_large(torch.tensor(29.0), 10, "loss", num_samples=1) is False
        assert machine.is_unexpectedly_large(torch.tensor(31.0), 10, "loss", num_samples=1) is True

    def test_nan_and_inf_are_ignored(self):
        machine = _make_machine()
        machine.is_unexpectedly_large(torch.tensor(1.0), 10, "loss", num_samples=1)
        assert machine.is_unexpectedly_large(torch.tensor(float("nan")), 10, "loss") is False
        assert machine.is_unexpectedly_large(torch.tensor(float("inf")), 10, "loss") is False

    def test_resample_restarts_sampling(self):
        machine = _make_machine()
        machine.is_unexpectedly_large(torch.tensor(100.0), 10, "loss", num_samples=1)
        assert machine.is_unexpectedly_large(torch.tensor(1.0), 10, "loss", num_samples=1, resample=True) is False
        # max is resampled from scratch, so 1.0 * 10 is the new threshold
        assert machine.is_unexpectedly_large(torch.tensor(5.0), 10, "loss", num_samples=1) is False

    def test_contexts_are_tracked_independently(self):
        machine = _make_machine()
        machine.is_unexpectedly_large(torch.tensor(1.0), 10, "loss", num_samples=1)
        assert machine.is_unexpectedly_large(torch.tensor(50.0), 10, "grad", num_samples=1) is False


class TestStateDict:
    def _machine_with_state(self, tmp_path):
        initialize_rerun_state_machine()
        machine = _make_machine()
        machine.current_iteration = 5
        machine.state = RerunState.WILL_RERUN_FROM_CHECKPOINT
        machine.rerun_requested = True
        machine.suspicious_node = "node0"
        machine.suspicious_device = 3
        machine.max_values["loss"] = 2.5
        machine.large_value_counts["loss"] = 7
        return machine

    def test_state_dict_captures_mode_and_sharded_state(self, tmp_path):
        machine = self._machine_with_state(tmp_path)
        iterator = RerunDataIterator(iter([1, 2, 3]))
        next(iterator)

        state = machine.state_dict(iterator, "torch_dist")

        assert state["mode"] == RerunMode.VALIDATE_RESULTS
        sharded = state["sharded"]
        assert sharded["state"] == RerunState.WILL_RERUN_FROM_CHECKPOINT
        assert sharded["current_iteration"] == 5
        assert sharded["rerun_requested"] is True
        assert sharded["suspicious_node"] == "node0"
        assert sharded["suspicious_device"] == 3
        assert sharded["max_values"] == {"loss": 2.5}
        assert sharded["large_value_counts"] == {"loss": 7}
        assert len(sharded["data_iterator_checkpoints"]) == 1
        assert "error_injector_checkpoint" in sharded

    def test_state_dict_without_iterator(self, tmp_path):
        machine = self._machine_with_state(tmp_path)
        state = machine.state_dict(None, "torch_dist")
        assert state["sharded"]["data_iterator_checkpoints"] is None

    def test_load_state_dict_round_trip(self, tmp_path):
        machine = self._machine_with_state(tmp_path)
        state = machine.state_dict(None, "torch_dist")

        clone = _make_machine()
        clone.load_state_dict(state)

        assert clone.state == RerunState.WILL_RERUN_FROM_CHECKPOINT
        assert clone.current_iteration == 5
        assert clone.rerun_requested is True
        assert clone.suspicious_node == "node0"
        assert clone.suspicious_device == 3
        assert clone.max_values == {"loss": 2.5}

    def test_load_ignored_when_machine_disabled(self, tmp_path):
        machine = self._machine_with_state(tmp_path)
        state = machine.state_dict(None, "torch_dist")

        disabled = _make_machine(mode=RerunMode.DISABLED)
        disabled.load_state_dict(state)

        assert disabled.current_iteration == -1
        assert disabled.state == RerunState.NOT_RUNNING_YET

    def test_load_ignored_when_checkpoint_disabled(self, tmp_path):
        disabled = _make_machine(mode=RerunMode.DISABLED)
        disabled.current_iteration = 9
        state = disabled.state_dict(None, "torch_dist")

        machine = _make_machine()
        machine.load_state_dict(state)

        assert machine.current_iteration == -1


class TestGetSkippedIterationsFromTrackerFile:
    def _write_tracker(self, path, lines):
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    def test_second_rerun_reproducible_iterations_are_skipped(self, tmp_path):
        tracker = self._write_tracker(
            tmp_path / "tracker.txt",
            [
                "ts=2026-09-22 10:00:00 node=node0 device=0 jobID=N/A rank=0 "
                "iteration=7 status=second_rerun_reproducible result=1.0",
                "ts=2026-09-22 10:01:00 node=node0 device=0 jobID=N/A rank=1 "
                "iteration=3 status=second_rerun_reproducible result=2.0",
            ],
        )
        assert RerunStateMachine.get_skipped_iterations_from_tracker_file(tracker) == [3, 7]

    def test_malformed_lines_are_ignored(self, tmp_path):
        tracker = self._write_tracker(
            tmp_path / "tracker.txt",
            [
                "not a tracker line",
                "ts=2026-09-22 10:00:00 node=node0 device=0 jobID=N/A rank=0 "
                "iteration=7 status=second_rerun_reproducible result=1.0",
            ],
        )
        assert RerunStateMachine.get_skipped_iterations_from_tracker_file(tracker) == [7]

    def test_missing_file_returns_empty_list(self, tmp_path):
        assert RerunStateMachine.get_skipped_iterations_from_tracker_file(str(tmp_path / "gone.txt")) == []

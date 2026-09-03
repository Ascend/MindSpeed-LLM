# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
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
"""Tests of FSDP2 TrainMonitor resume logging (#1780)."""


def _avg_interval_loss(cumulative_loss, last_logged_loss, current_step, last_logged_step):
    step_diff = current_step - last_logged_step
    if step_diff <= 0:
        return 0.0
    return (cumulative_loss - last_logged_loss) / step_diff


def _restore_logging_cursors(extra_state, global_step):
    """Mirror Trainer.resume cursor + total restore."""
    last_logged_loss = extra_state.get("_last_logged_loss_scalar", 0.0)
    last_logged_lm_loss = extra_state.get("_last_logged_lm_loss_scalar", 0.0)
    last_logged_mtp_loss = extra_state.get("_last_logged_mtp_loss_scalar", 0.0)
    last_logged_aux_loss = extra_state.get("_last_logged_aux_loss_scalar", 0.0)
    return {
        "_global_step_last_logged": extra_state.get("_global_step_last_logged", global_step),
        "_last_logged_step": extra_state.get("_last_logged_step", global_step),
        "_last_logged_loss_scalar": last_logged_loss,
        "_last_logged_lm_loss_scalar": last_logged_lm_loss,
        "_last_logged_mtp_loss_scalar": last_logged_mtp_loss,
        "_last_logged_aux_loss_scalar": last_logged_aux_loss,
        "_total_loss_scalar": extra_state.get("_total_loss_scalar", last_logged_loss),
        "_total_lm_loss_scalar": extra_state.get("_total_lm_loss_scalar", last_logged_lm_loss),
        "_total_mtp_loss_scalar": extra_state.get("_total_mtp_loss_scalar", last_logged_mtp_loss),
        "_total_aux_loss_scalar": extra_state.get("_total_aux_loss_scalar", last_logged_aux_loss),
    }


class TestTrainMonitorResume:
    """Resume logging cursor / total restore regressions for Issue #1780."""

    def test_buggy_resume_divides_by_global_step(self):
        step_loss = 7.74
        avg = _avg_interval_loss(step_loss, 0.0, current_step=31, last_logged_step=0)
        assert abs(avg - step_loss / 31) < 1e-6

    def test_fixed_resume_averages_one_step(self):
        step_loss = 7.74
        avg = _avg_interval_loss(step_loss, 0.0, current_step=31, last_logged_step=30)
        assert abs(avg - step_loss) < 1e-6

    def test_restore_logging_cursors_prefers_saved_last_logged_step(self):
        extra = {
            "global_step": 30,
            "_global_step_last_logged": 0,
            "_last_logged_step": 30,
            "_last_logged_lm_loss_scalar": 12.5,
            "_total_lm_loss_scalar": 12.5,
        }
        cursors = _restore_logging_cursors(extra, global_step=30)
        assert cursors["_last_logged_step"] == 30
        assert cursors["_global_step_last_logged"] == 0
        assert cursors["_last_logged_lm_loss_scalar"] == 12.5
        assert cursors["_total_lm_loss_scalar"] == 12.5

    def test_restore_logging_cursors_defaults_old_checkpoint_to_global_step(self):
        extra = {"global_step": 30, "_global_step_last_logged": 0}
        cursors = _restore_logging_cursors(extra, global_step=30)
        assert cursors["_last_logged_step"] == 30
        assert cursors["_total_lm_loss_scalar"] == 0.0
        assert cursors["_last_logged_lm_loss_scalar"] == 0.0

    def test_new_checkpoint_resume_keeps_total_and_last_logged_aligned(self):
        extra = {
            "global_step": 30,
            "_last_logged_step": 30,
            "_last_logged_lm_loss_scalar": 232.2,
            "_total_lm_loss_scalar": 232.2,
        }
        cursors = _restore_logging_cursors(extra, global_step=30)
        post_resume_total = cursors["_total_lm_loss_scalar"] + 7.74
        avg = _avg_interval_loss(
            post_resume_total,
            cursors["_last_logged_lm_loss_scalar"],
            current_step=31,
            last_logged_step=cursors["_last_logged_step"],
        )
        assert abs(avg - 7.74) < 1e-6

    def test_last_logged_without_total_seeds_total_from_last_logged(self):
        extra = {
            "global_step": 30,
            "_last_logged_step": 30,
            "_last_logged_lm_loss_scalar": 232.2,
        }
        cursors = _restore_logging_cursors(extra, global_step=30)
        assert cursors["_total_lm_loss_scalar"] == 232.2
        buggy = _avg_interval_loss(7.74, 232.2, current_step=31, last_logged_step=30)
        assert buggy < -200
        fixed = _avg_interval_loss(
            cursors["_total_lm_loss_scalar"] + 7.74,
            cursors["_last_logged_lm_loss_scalar"],
            current_step=31,
            last_logged_step=30,
        )
        assert abs(fixed - 7.74) < 1e-6

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
"""Test of pure helpers in core/transformer/multi_token_prediction.py."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer.multi_token_prediction import (
    generate_mtp_batch_list_on_this_tp_rank,
    get_mtp_layer_input,
    get_mtp_num_layers_to_build,
    mtp_block_build_layers_wrapper,
    mtp_block_forward,
    mtp_layer_forward,
    mtp_layer_init_wrapper,
    mtp_reduce_loss_in_tracker,
    track_mtp_metrics,
)
from megatron.core import mpu, parallel_state
from tests.test_tools.utils import create_testconfig

_MTP = "mindspeed_llm.core.transformer.multi_token_prediction"


def _config(mtp_num_layers=2):
    return SimpleNamespace(mtp_num_layers=mtp_num_layers)


class TestGetMtpNumLayersToBuild:
    """get_mtp_num_layers_to_build: MTP layer count per pipeline stage."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, is_last_stage, is_first_stage, schedules_method, dualpipev_first_chunk, mtp_num_layers, expected",
        test_config["test_get_mtp_num_layers_to_build"],
    )
    def test_get_mtp_num_layers_to_build(
        self, label, is_last_stage, is_first_stage, schedules_method, dualpipev_first_chunk, mtp_num_layers, expected
    ):
        with (
            mock.patch.object(mpu, "is_pipeline_last_stage", return_value=is_last_stage),
            mock.patch.object(mpu, "is_pipeline_first_stage", return_value=is_first_stage),
            mock.patch(f"{_MTP}.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(
                schedules_method=schedules_method, dualpipev_first_chunk=dualpipev_first_chunk
            )
            assert get_mtp_num_layers_to_build(_config(mtp_num_layers)) == expected, label


class TestGetMtpLayerInput:
    """get_mtp_layer_input: assemble (and optionally shift) MTP layer inputs."""

    def test_uses_batch_list_without_shift(self):
        tokens = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[5, 6, 7, 8]])
        loss_mask = torch.tensor([[1, 1, 1, 1]])
        batch_list = [
            {
                "tokens": tokens,
                "position_ids": torch.tensor([[0, 1, 2, 3]]),
                "labels": labels,
                "loss_mask": loss_mask,
                "attention_mask": None,
            }
        ]
        out_ids, _, out_labels, out_mask, _ = get_mtp_layer_input(None, batch_list, 0)
        assert torch.equal(out_ids, tokens)
        assert torch.equal(out_labels, labels)
        assert torch.equal(out_mask, loss_mask)

    def test_rolls_when_no_batch_list(self):
        input_data = (
            torch.tensor([[1, 2, 3, 4]]),
            None,
            torch.tensor([[10, 20, 30, 40]]),
            torch.tensor([[1, 1, 1, 0]]),
            None,
        )
        out_ids, _, out_labels, _, _ = get_mtp_layer_input(input_data, None, 0)
        assert out_ids[0, 0].item() == 2
        assert out_labels[0, 0].item() == 20

    def test_defaults_loss_mask_to_ones(self):
        input_data = (torch.tensor([[1, 2, 3, 4]]), None, torch.tensor([[10, 20, 30, 40]]), None, None)
        _, _, _, out_mask, _ = get_mtp_layer_input(input_data, None, 0)
        assert out_mask is not None
        assert tuple(out_mask.shape) == (1, 4)

    def test_raises_without_labels(self):
        input_data = (torch.tensor([[1, 2, 3]]), None, None, torch.tensor([[1, 1, 1]]), None)
        with pytest.raises(AssertionError):
            get_mtp_layer_input(input_data, None, 0)


class TestMtpLayerForward:
    """mtp_layer_forward: cross-attention guard + norm/proj/transformer body."""

    def test_context_not_none_raises(self):
        with mock.patch(
            "mindspeed_llm.core.transformer.multi_token_prediction.get_args", return_value=SimpleNamespace()
        ):
            with pytest.raises(NotImplementedError):
                mtp_layer_forward(
                    SimpleNamespace(),
                    decoder_input=None,
                    hidden_states=None,
                    attention_mask=None,
                    context=object(),
                )

    @staticmethod
    def _self():
        # eh_proj halves the concatenated (2H) back to H; norms are identity;
        # transformer_layer / hc_head pass hidden_states through.
        return SimpleNamespace(
            config=SimpleNamespace(sequence_parallel=False, fp8=False),
            sequence_parallel=False,
            enorm=lambda x: x,
            hnorm=lambda x: x,
            eh_proj=lambda x: (x[..., : x.shape[-1] // 2], None),
            transformer_layer=lambda **kw: (kw["hidden_states"], None),
            hc_head=lambda h, mhc_stage: h,
        )

    def _run(self, pre_process, post_process):
        h = 8
        decoder_input = torch.randn(4, 2, h)
        hidden_states = torch.randn(4, 2, h)
        with (
            mock.patch(
                "mindspeed_llm.core.transformer.multi_token_prediction.get_args",
                return_value=SimpleNamespace(enable_mhc=False, hc_mult=1),
            ),
            mock.patch(
                "mindspeed_llm.core.transformer.multi_token_prediction.make_viewless_tensor",
                side_effect=lambda inp, **k: inp,
            ),
            mock.patch(
                "mindspeed_llm.core.transformer.multi_token_prediction.all_gather_last_dim_from_tensor_parallel_region",
                side_effect=lambda x: x,
            ),
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.hc_repeat", side_effect=lambda x, *a: x),
        ):
            return mtp_layer_forward(
                self._self(),
                decoder_input=decoder_input,
                hidden_states=hidden_states,
                attention_mask=None,
                pre_process=pre_process,
                post_process=post_process,
            )

    def test_body_baseline(self):
        out = self._run(pre_process=False, post_process=False)
        assert tuple(out.shape) == (4, 2, 8)

    def test_body_pre_and_post_process(self):
        out = self._run(pre_process=True, post_process=True)
        assert tuple(out.shape) == (4, 2, 8)


class TestMtpReduceLossInTracker:
    """mtp_reduce_loss_in_tracker: reduce logged MTP losses (dist-free path)."""

    def test_empty_tracker_is_noop(self):
        with mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.MTPLossLoggingHelper") as mock_helper:
            mock_helper.tracker = {}
            mtp_reduce_loss_in_tracker()  # must not raise

    def test_no_groups_leaves_values_unchanged(self):
        with mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.MTPLossLoggingHelper") as mock_helper:
            values = torch.tensor([1.0, 2.0])
            mock_helper.tracker = {"values": values, "reduce_group": None, "avg_group": None}
            mtp_reduce_loss_in_tracker()
            assert torch.equal(mock_helper.tracker['values'], values)


class TestTrackMtpMetrics:
    """track_mtp_metrics: write per-layer MTP losses to logging sinks."""

    def test_empty_tracker_is_noop(self):
        with mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.MTPLossLoggingHelper") as mock_helper:
            mock_helper.tracker = {}
            total = {}
            track_mtp_metrics(1.0, 0, writer=None, total_loss_dict=total)
            assert not total

    def test_populates_loss_dict_and_writer(self):
        with mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.MTPLossLoggingHelper") as mock_helper:
            mock_helper.tracker = {"values": torch.tensor([1.0, 2.0]), "reduce_group": None, "avg_group": None}
            writer = mock.MagicMock()
            total = {}
            track_mtp_metrics(1.0, 5, writer=writer, total_loss_dict=total)
            assert 'mtp_1 loss' in total
            assert 'mtp_2 loss' in total
            assert writer.add_scalar.call_count == 2

    def test_wandb_writer_logs_per_layer(self):
        with mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.MTPLossLoggingHelper") as mock_helper:
            mock_helper.tracker = {"values": torch.tensor([1.0, 2.0]), "reduce_group": None, "avg_group": None}
            wandb_writer = mock.MagicMock()
            track_mtp_metrics(1.0, 5, writer=None, wandb_writer=wandb_writer, total_loss_dict={})
            assert wandb_writer.log.call_count == 2

    def test_barrier_called_when_group_present(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.MTPLossLoggingHelper") as mock_helper,
            mock.patch("torch.distributed.barrier") as mock_barrier,
        ):
            group = object()
            mock_helper.tracker = {"values": torch.tensor([1.0]), "reduce_group": group, "avg_group": None}
            track_mtp_metrics(1.0, 5, writer=None, total_loss_dict={})
            mock_barrier.assert_called_once()


class TestMtpBlockBuildLayersWrapper:
    """mtp_block_build_layers_wrapper: attach final_layernorms after the wrapped init."""

    def test_empty_layer_specs_gives_empty_norm_list(self):
        calls = []
        wrapped = mtp_block_build_layers_wrapper(lambda self: calls.append(True))
        self_obj = SimpleNamespace(
            submodules=SimpleNamespace(layer_specs=[]),
            config=SimpleNamespace(hidden_size=8, layernorm_epsilon=1e-5),
        )
        wrapped(self_obj)
        assert calls
        assert len(self_obj.final_layernorms) == 0


class TestGenerateMtpBatchList:
    """generate_mtp_batch_list_on_this_tp_rank: reset-mask seq math + return guard."""

    def test_reset_attention_mask_recomputes_seq_len(self):
        with (
            mock.patch.object(mpu, "is_pipeline_last_stage", return_value=False),
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.set_actual_seq_len") as mock_set_asl,
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.get_actual_seq_len") as mock_get_asl,
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(
                reset_attention_mask=True,
                mtp_num_layers=1,
                context_parallel_size=1,
            )
            mock_get_asl.return_value = torch.tensor([4, 6])  # seq_len=4: 4 kept, 6 -> 6-1=5
            batch = {"position_ids": torch.zeros(1, 4)}
            result = generate_mtp_batch_list_on_this_tp_rank(batch)
            assert result is None  # not last-stage/CP<=1 -> guard returns None
            # set_actual_seq_len called with rows [[4,6],[4,5]]
            arg = mock_set_asl.call_args[0][0]
            assert arg.tolist() == [[4, 6], [4, 5]]

    def test_returns_none_when_not_applicable(self):
        with (
            mock.patch.object(mpu, "is_pipeline_last_stage", return_value=False),
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(
                reset_attention_mask=False,
                mtp_num_layers=0,
                context_parallel_size=1,
            )
            assert generate_mtp_batch_list_on_this_tp_rank({'tokens': None}) is None


class TestMtpLayerInitWrapper:
    """mtp_layer_init_wrapper: build transformer layer + hc head, set mtp_idx."""

    def test_sets_mtp_idx_and_builds_head(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.get_mhc_spec", return_value="hc_spec"),
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.build_module") as mock_build,
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(enable_mhc=False)
            # transformer_layer built first, then hc_head; give the layer a self_attention
            built_layer = SimpleNamespace(self_attention=SimpleNamespace(mtp_idx=None))
            mock_build.side_effect = [built_layer, "hc_head"]

            def fake_fn(self, config, submodules, layer_number):
                self.config = config
                self.layer_number = layer_number

            wrapped = mtp_layer_init_wrapper(fake_fn)
            self_obj = SimpleNamespace()
            subm = SimpleNamespace(transformer_layer="tl_spec")
            wrapped(self_obj, config="cfg", submodules=subm, layer_number=3)

            assert self_obj.transformer_layer is built_layer
            assert self_obj.transformer_layer.mtp_idx == 3
            assert built_layer.self_attention.mtp_idx == 3
            assert self_obj.final_layernorm is None
            assert self_obj.hc_head == 'hc_head'


class TestMtpBlockForward:
    """mtp_block_forward: loop over MTP layers, per-layer loss, scaled main hidden."""

    @staticmethod
    def _self(training=False):
        return SimpleNamespace(
            layers=[lambda **kw: kw["hidden_states"]],
            final_layernorms=[lambda x: x],
            training=training,
            mtp_loss_scaling_factor=1.0,
            config=SimpleNamespace(mtp_num_layers=1, calculate_per_token_loss=True),
        )

    def test_block_forward_returns_main_hidden(self):
        s, b, h = 4, 2, 8
        hidden = torch.randn(s, b, h)

        class _Emb:
            def __init__(self):
                self.word_embeddings = SimpleNamespace(weight=torch.randn(10, h))

            def __call__(self, input_ids, position_ids):
                return torch.randn(s, b, h)

        labels = torch.zeros(s, b, dtype=torch.long)
        loss_mask = torch.ones(s, b)
        with (
            mock.patch(
                "mindspeed_llm.core.transformer.multi_token_prediction.get_args",
                return_value=SimpleNamespace(is_instruction_dataset=False, context_parallel_size=1),
            ),
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.get_mtp_batch_list", return_value=None),
            mock.patch(
                "mindspeed_llm.core.transformer.multi_token_prediction.make_viewless_tensor",
                side_effect=lambda inp, **k: inp,
            ),
            mock.patch.object(parallel_state, "get_context_parallel_world_size", return_value=1),
            mock.patch("mindspeed_llm.core.transformer.multi_token_prediction.MTPLossAutoScaler") as mock_scaler,
        ):
            mock_scaler.apply = lambda main, scaled: main
            out = mtp_block_forward(
                self._self(),
                input_ids=torch.zeros(s, b, dtype=torch.long),
                position_ids=torch.zeros(s, b, dtype=torch.long),
                hidden_states=hidden,
                attention_mask=None,
                labels=labels,
                loss_mask=loss_mask,
                embedding=_Emb(),
                output_layer=lambda x, weight, runtime_gather_output: (torch.randn(s, b, 10), None),
                compute_language_model_loss=lambda lbl, lg: torch.ones(s, b),
            )
        assert tuple(out.shape) == (s, b, h)

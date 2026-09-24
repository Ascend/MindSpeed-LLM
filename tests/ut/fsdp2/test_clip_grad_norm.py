# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mindspeed_llm.fsdp2.optim.clip_grad_norm as clip_grad_norm_module
from mindspeed_llm.fsdp2.optim.clip_grad_norm import fsdp2_clip_grad_norm
from tests.test_tools.utils import create_testconfig


class FakeParallelState:
    def get_fsdp_group(self):
        return None

    def is_ep_enable(self):
        return True

    def get_ep_group(self):
        return None

    def get_efsdp_device_mesh(self):
        return None


class TestFSDP2ClipGradNorm:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("params", test_config["test_clip_grad_norm"])
    def test_clip_grad_norm(self, params):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate FSDP2 gradient-norm calculation and clipping.
        Remarks: This test uses CPU tensors and does not require /data/ci resources.
        """
        parameters = []
        for gradient in params["gradients"]:
            parameter = torch.nn.Parameter(torch.zeros(len(gradient)))
            parameter.grad = torch.tensor(gradient)
            parameters.append(parameter)

        total_norm = fsdp2_clip_grad_norm(
            parameters,
            max_norm=params["max_norm"],
            norm_type=float("inf") if params.get("norm_type") == "inf" else 2.0,
        )

        torch.testing.assert_close(total_norm, torch.tensor(params["expected_total_norm"]))
        for parameter, expected_gradient in zip(parameters, params["expected_gradients"]):
            torch.testing.assert_close(parameter.grad, torch.tensor(expected_gradient))

    def test_nonfinite_gradient_can_be_rejected(self):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate non-finite gradient handling.
        Remarks: This test uses CPU tensors and does not require /data/ci resources.
        """
        parameter = torch.nn.Parameter(torch.zeros(1))
        parameter.grad = torch.tensor([float("nan")])

        with pytest.raises(RuntimeError, match="non-finite"):
            fsdp2_clip_grad_norm([parameter], max_norm=1.0, error_if_nonfinite=True)

    @pytest.mark.parametrize("params", test_config["test_clip_grad_norm_wrapper"])
    def test_clip_grad_norm_wrapper(self, params):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate ordinary-model wrapper behavior.
        Remarks: This test uses CPU tensors and does not require /data/ci resources.
        """
        parameter = torch.nn.Parameter(torch.zeros(len(params["gradient"])))
        parameter.grad = torch.tensor(params["gradient"])
        model = SimpleNamespace(parameters=lambda: [parameter])

        total_norm = clip_grad_norm_module.clip_grad_norm(model, max_norm=params["max_norm"])

        torch.testing.assert_close(total_norm, torch.tensor(params["expected_total_norm"]))
        torch.testing.assert_close(parameter.grad, torch.tensor(params["expected_gradient"]))

    @pytest.mark.parametrize("params", test_config["test_ep_fsdp2_clip_grad_norm"])
    def test_ep_fsdp2_clip_grad_norm(self, params, monkeypatch):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate EP-aware clipping with a shared global coefficient.
        Remarks: This test uses CPU tensors and does not require /data/ci resources.
        """
        monkeypatch.setattr(clip_grad_norm_module, "ParallelState", FakeParallelState)
        ep_parameter = torch.nn.Parameter(torch.zeros(len(params["ep_gradient"])))
        ep_parameter.grad = torch.tensor(params["ep_gradient"])
        non_ep_parameter = torch.nn.Parameter(torch.zeros(len(params["non_ep_gradient"])))
        non_ep_parameter.grad = torch.tensor(params["non_ep_gradient"])
        model = SimpleNamespace(_ep_param_groups={"ep": [ep_parameter], "non_ep": [non_ep_parameter]})

        total_norm = clip_grad_norm_module.clip_grad_norm(
            model,
            max_norm=params["max_norm"],
            norm_type=float("inf") if params.get("norm_type") == "inf" else 2.0,
        )

        torch.testing.assert_close(total_norm, torch.tensor(params["expected_total_norm"]))
        torch.testing.assert_close(ep_parameter.grad, torch.tensor(params["expected_ep_gradient"]))
        torch.testing.assert_close(non_ep_parameter.grad, torch.tensor(params["expected_non_ep_gradient"]))

    def test_empty_gradients_return_zero_norm(self):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate empty gradient handling.
        Remarks: This test uses CPU tensors and does not require /data/ci resources.
        """
        parameter = torch.nn.Parameter(torch.zeros(1))

        total_norm = fsdp2_clip_grad_norm([parameter], max_norm=1.0)

        torch.testing.assert_close(total_norm, torch.tensor(0.0))

    def test_reduce_group_uses_expected_collective(self, monkeypatch):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate finite and infinity norm reduction collectives.
        Remarks: This test mocks collectives and does not require /data/ci resources.
        """
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3.0, 4.0])
        collectives = []

        def record_collective(value, op, group):
            collectives.append((value.item(), op, group))

        group = object()
        monkeypatch.setattr(clip_grad_norm_module.dist, "all_reduce", record_collective)

        finite_sum = clip_grad_norm_module._fsdp2_reduce_group([parameter], 2.0, [("fsdp", group)])
        inf_max = clip_grad_norm_module._fsdp2_reduce_group([parameter], float("inf"), [("fsdp", group)])

        torch.testing.assert_close(finite_sum, torch.tensor(25.0))
        torch.testing.assert_close(inf_max, torch.tensor(4.0))
        assert collectives == [
            (25.0, clip_grad_norm_module.dist.ReduceOp.SUM, group),
            (4.0, clip_grad_norm_module.dist.ReduceOp.MAX, group),
        ]

    def test_rank_zero_logging(self, monkeypatch):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate rank-zero-only logging.
        Remarks: This test mocks distributed state and does not require /data/ci resources.
        """
        messages = []
        monkeypatch.setattr(clip_grad_norm_module.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(clip_grad_norm_module.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(clip_grad_norm_module.logger, "info", messages.append)

        clip_grad_norm_module._get_rank0_log("gradient clipping completed")

        assert messages == ["gradient clipping completed"]

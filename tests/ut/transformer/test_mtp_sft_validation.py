# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU-only regression tests for packed SFT MTP argument validation."""

import importlib.util
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


class TestMTPSFTValidation(unittest.TestCase):
    @staticmethod
    def _feature():
        module_name = "mindspeed.features_manager.feature"
        dependency = types.ModuleType(module_name)

        class BaseFeature:
            def __init__(self, **kwargs):
                del kwargs

        dependency.MindSpeedFeature = BaseFeature
        source = Path(__file__).resolve().parents[3] / "mindspeed_llm/features_manager/transformer/mtp.py"
        spec = importlib.util.spec_from_file_location("_mtp_validation_test", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {module_name: dependency}):
            spec.loader.exec_module(module)
        return module.MultiTokenPredictionFeature()

    @staticmethod
    def _args(**overrides):
        values = {
            "stage": "sft",
            "reset_attention_mask": True,
            "mtp_num_layers": 1,
            "micro_batch_size": 1,
            "attention_mask_type": "causal",
            "context_parallel_size": 4,
            "context_parallel_algo": "kvallgather_cp_algo",
        }
        values.update(overrides)
        return Namespace(**values)

    def test_rejects_multiple_samples_before_mtp_is_hidden(self):
        for cp_size in (1, 4):
            with self.subTest(cp_size=cp_size):
                args = self._args(micro_batch_size=2, context_parallel_size=cp_size)
                with self.assertRaisesRegex(ValueError, "--micro-batch-size 1"):
                    self._feature().pre_validate_args(args)
                self.assertEqual(args.mtp_num_layers, 1)

    def test_rejects_causal_ring_padding_before_mtp_is_hidden(self):
        args = self._args(context_parallel_algo="megatron_cp_algo")
        with self.assertRaisesRegex(ValueError, "one-dimensional boundaries"):
            self._feature().pre_validate_args(args)
        self.assertEqual(args.mtp_num_layers, 1)

    def test_permitted_configurations_preserve_mtp_after_validation(self):
        configurations = (
            {},
            {"mtp_num_layers": 2},
            {"context_parallel_size": 1, "context_parallel_algo": "megatron_cp_algo"},
            {"attention_mask_type": "general", "context_parallel_algo": "megatron_cp_algo"},
            {"mtp_num_layers": None, "micro_batch_size": 2, "context_parallel_algo": "megatron_cp_algo"},
            {"reset_attention_mask": False, "micro_batch_size": 2, "context_parallel_algo": "megatron_cp_algo"},
            {"stage": "dpo", "micro_batch_size": 2, "context_parallel_algo": "megatron_cp_algo"},
            {"stage": None, "micro_batch_size": 2, "context_parallel_algo": "megatron_cp_algo"},
        )
        for overrides in configurations:
            with self.subTest(overrides=overrides):
                args = self._args(**overrides)
                original_layers = args.mtp_num_layers
                feature = self._feature()
                feature.pre_validate_args(args)
                if original_layers and args.context_parallel_size > 1:
                    self.assertIsNone(args.mtp_num_layers)
                feature.validate_args(args)
                feature.post_validate_args(args)
                self.assertEqual(args.mtp_num_layers, original_layers)


if __name__ == "__main__":
    unittest.main()

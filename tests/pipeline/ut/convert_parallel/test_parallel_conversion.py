# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for parallel_dispatch_converted in weight_conv_adapter.py.

Only tests the NEW parallel dispatch function compared to the original model_loader.py.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import torch

from mindspeed_llm.fsdp2.checkpoint.weight_conv_adapter import parallel_dispatch_converted


def make_conversion_task(layer_idx, expert_count=4, hidden=64, intermediate=128):
    """
    Build a (converter, full_name, tensors_only, original_keys) task tuple.
    The converter's .convert() simulates gate_up flatten:
      [num_experts,inter_dim,hid_dim] + [num_experts,inter_dim,hid_dim] -> cat on dim=1
    """
    num_experts, inter_dim, hid_dim = expert_count, intermediate, hidden
    gate = torch.randn(num_experts, inter_dim, hid_dim)
    up = torch.randn(num_experts, inter_dim, hid_dim)
    collected = {
        f"model.layers.{layer_idx}.mlp.gate_proj.weight": [gate],
        f"model.layers.{layer_idx}.mlp.up_proj.weight": [up],
    }
    original_keys = {f"model.layers.{layer_idx}.mlp.gate_proj.weight": ["key_0"]}

    converter = MagicMock()
    converter.source_patterns = [
        f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        f"model.layers.{layer_idx}.mlp.up_proj.weight",
    ]
    converter.convert = MagicMock(
        return_value={f"model.layers.{layer_idx}.mlp.gate_up_proj.weight": torch.cat([gate, up], dim=1)}
    )
    converter.collected_tensors = collected
    full_name = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
    return (converter, full_name, collected, original_keys)


class TestParallelDispatchConverted(unittest.TestCase):
    def test_single_task_falls_back_to_sequential(self):
        """When only 1 task, the else branch (no ThreadPoolExecutor) is used."""
        tasks = [make_conversion_task(0)]
        dispatched = {}
        parallel_dispatch_converted(tasks, lambda name, tensor: dispatched.update({name: tensor}))
        self.assertIn("model.layers.0.mlp.gate_up_proj.weight", dispatched)
        # Verify shape: cat on dim=1 of [4,128,64]+[4,128,64] = [4,256,64]
        self.assertEqual(dispatched["model.layers.0.mlp.gate_up_proj.weight"].shape, (4, 256, 64))

    def test_multiple_tasks_all_collected(self):
        """All tasks' results are dispatched when running in parallel."""
        tasks = [make_conversion_task(i) for i in range(6)]
        dispatched = {}
        parallel_dispatch_converted(tasks, lambda name, tensor: dispatched.update({name: tensor}))
        for i in range(6):
            name = f"model.layers.{i}.mlp.gate_up_proj.weight"
            self.assertIn(name, dispatched)

    def test_num_workers_capped_at_4(self):
        """ThreadPoolExecutor is created with max_workers=min(tasks, 4)."""
        tasks = [make_conversion_task(i) for i in range(10)]
        dispatched = {}
        captured_max_workers = []

        original_init = ThreadPoolExecutor.__init__

        def spy_init(self, *args, max_workers=None, **kwargs):
            captured_max_workers.append(max_workers)
            return original_init(self, *args, max_workers=max_workers, **kwargs)

        with patch.object(ThreadPoolExecutor, "__init__", spy_init):
            parallel_dispatch_converted(tasks, lambda name, tensor: dispatched.update({name: tensor}))

        self.assertEqual(captured_max_workers, [4])

    def test_thread_count_restored_after_parallel(self):
        """torch.set_num_threads is restored in finally block."""
        original = torch.get_num_threads()
        parallel_dispatch_converted(
            [make_conversion_task(i) for i in range(3)],
            lambda name, tensor: None,
        )
        self.assertEqual(torch.get_num_threads(), original)

    def test_thread_count_restored_on_exception(self):
        """torch.set_num_threads is restored even if a task raises."""
        original = torch.get_num_threads()

        bad_converter = MagicMock()
        bad_converter.convert = MagicMock(side_effect=RuntimeError("boom"))
        bad_converter.collected_tensors = {}
        tasks = [(bad_converter, "bad.param", {}, {})]

        with self.assertRaises(RuntimeError):
            parallel_dispatch_converted(tasks, lambda name, tensor: None)

        self.assertEqual(torch.get_num_threads(), original)

    def test_parallel_matches_sequential_results(self):
        """Parallel execution produces identical results to sequential execution."""
        tasks = [make_conversion_task(i, expert_count=e, hidden=32, intermediate=64) for i, e in enumerate([2, 4, 8])]

        # Sequential baseline: dispatch one by one
        seq_results = {}
        for task in tasks:
            cv, fn, tn, ok = task
            for name, tensor in cv.convert(fn).items():
                seq_results[name] = tensor

        # Parallel via parallel_dispatch_converted
        par_results = {}
        parallel_dispatch_converted(tasks, lambda name, tensor: par_results.update({name: tensor}))

        self.assertEqual(set(seq_results.keys()), set(par_results.keys()))
        for key in seq_results:
            self.assertTrue(torch.equal(seq_results[key], par_results[key]), f"Mismatch for {key}")

    def test_empty_tasks_no_crash(self):
        """Empty conversion_tasks list does not crash."""
        dispatched = {}
        parallel_dispatch_converted([], lambda name, tensor: dispatched.update({name: tensor}))
        self.assertEqual(dispatched, {})

    def test_converter_receives_correct_full_name(self):
        """Each converter receives its own full_name, not another task's."""
        captured_calls = []

        def make_capture_converter(name_prefix):
            c = MagicMock()
            c.convert = MagicMock(
                side_effect=lambda target_name: (
                    captured_calls.append((name_prefix, target_name)),
                    {name_prefix: torch.tensor(0.0)},
                )[1]
            )
            c.collected_tensors = {}
            return c

        tasks = [(make_capture_converter(f"layer_{i}"), f"layer_{i}", {}, {}) for i in range(3)]
        parallel_dispatch_converted(tasks, lambda name, tensor: None)

        for i in range(3):
            calls_for_i = [target for pfx, target in captured_calls if pfx == f"layer_{i}"]
            self.assertEqual(calls_for_i, [f"layer_{i}"])


if __name__ == "__main__":
    unittest.main()

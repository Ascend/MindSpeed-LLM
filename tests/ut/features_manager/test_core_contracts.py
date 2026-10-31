# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; CPU argument and lifecycle contracts.

Remarks: No model files, accelerator or distributed initialization required.
"""

from argparse import ArgumentParser, Namespace
import ast
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import types

import pytest

from test_json_config import load_cases


ROOT = Path(__file__).resolve().parents[3]


class FeatureEnvironment:
    def __init__(self, monkeypatch, mindspeed_root):
        self.monkeypatch = monkeypatch
        self.roots = {"mindspeed_llm": ROOT / "mindspeed_llm", "mindspeed": mindspeed_root}

    def package(self, name):
        if name in sys.modules:
            return sys.modules[name]
        root, *parts = name.split(".")
        module = types.ModuleType(name)
        module.__path__ = [str(self.roots[root].joinpath(*parts))]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        self.monkeypatch.setitem(sys.modules, name, module)
        if parts:
            parent = self.package(name.rsplit(".", 1)[0])
            self.monkeypatch.setattr(parent, parts[-1], module, raising=False)
        return module

    def module(self, name):
        self.package(name.rsplit(".", 1)[0])
        root, *parts = name.split(".")
        path = self.roots[root].joinpath(*parts).with_suffix(".py")
        if path.is_file():
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.ImportFrom) and node.module == "mindspeed.features_manager":
                    for alias in node.names:
                        self.export(alias.name)
        return importlib.import_module(name)

    def export(self, symbol):
        """Resolve package re-exports from the real dependency, without its adaptor."""
        package = self.package("mindspeed.features_manager")
        if hasattr(package, symbol):
            return getattr(package, symbol)
        path = self.roots["mindspeed"] / "features_manager/__init__.py"
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == symbol:
                        name = node.module
                        # A few exports come from a subpackage's __init__.
                        module_path = self.roots["mindspeed"].parent / name.replace(".", "/")
                        if module_path.is_dir():
                            self.package(name)
                            for subnode in ast.parse((module_path / "__init__.py").read_text()).body:
                                if isinstance(subnode, ast.ImportFrom):
                                    for subalias in subnode.names:
                                        if subalias.name == alias.name:
                                            name = (
                                                importlib.util.resolve_name(
                                                    "." * subnode.level + subnode.module, node.module
                                                )
                                                if subnode.level
                                                else subnode.module
                                            )
                        value = getattr(self.module(name), alias.name)
                        self.monkeypatch.setattr(package, symbol, value, raising=False)
                        return value
        raise LookupError(symbol)

    def registry(self):
        return self.module("mindspeed.patch_utils").MindSpeedPatchesManager

    def callbacks(self, name, symbols):
        callbacks = {}
        for symbol in symbols.split():

            def callback(*args, **kwargs):
                raise AssertionError("An out-of-scope training callback was executed")

            callback.__name__ = symbol
            callback.__module__ = name
            callbacks[symbol] = callback
        return self.boundary(name, **callbacks)

    def catalog(self):
        name = "mindspeed_llm.features_manager"
        path = self.roots["mindspeed_llm"] / "features_manager/__init__.py"
        for source in path.parent.rglob("*.py"):
            for node in ast.parse(source.read_text()).body:
                if isinstance(node, ast.ImportFrom) and node.module == "mindspeed.features_manager":
                    for alias in node.names:
                        self.export(alias.name)
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        self.monkeypatch.setitem(sys.modules, name, module)
        self.monkeypatch.setattr(self.package("mindspeed_llm"), "features_manager", module, raising=False)
        spec.loader.exec_module(module)
        return module

    def feature(self, path, class_name):
        module = self.module("mindspeed_llm.features_manager." + path)
        expected = ROOT / "mindspeed_llm/features_manager" / (path.replace(".", "/") + ".py")
        assert Path(module.__file__).resolve() == expected.resolve()
        return getattr(module, class_name)()

    def boundary(self, name, **symbols):
        """Provide explicit out-of-scope callbacks; never replace feature logic."""
        if name.split(".")[0] in self.roots:
            parent = self.package(name.rsplit(".", 1)[0])
        else:
            parent = None
        module = types.ModuleType(name)
        module.__dict__.update(symbols)
        self.monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            self.monkeypatch.setattr(parent, name.rsplit(".", 1)[1], module, raising=False)
        return module


@pytest.fixture(name="feature_env")
def feature_environment(monkeypatch):
    source = os.environ.get("MINDSPEED_SOURCE")
    if source:
        mindspeed_root = Path(source).resolve() / "mindspeed"
    else:
        spec = importlib.machinery.PathFinder.find_spec("mindspeed")
        if spec is None:
            pytest.fail("Install MindSpeed or set MINDSPEED_SOURCE to its checkout")
        assert spec.submodule_search_locations
        mindspeed_root = Path(spec.submodule_search_locations[0])
    assert (mindspeed_root / "features_manager/feature.py").is_file()
    saved = {
        name: module
        for name, module in sys.modules.copy().items()
        if name == "mindspeed"
        or name.startswith("mindspeed.")
        or name == "mindspeed_llm"
        or name.startswith("mindspeed_llm.")
    }
    for name in saved:
        monkeypatch.delitem(sys.modules, name)
    env = FeatureEnvironment(monkeypatch, mindspeed_root)
    try:
        env.package("mindspeed")
        env.package("mindspeed_llm.features_manager")
        base = env.module("mindspeed.features_manager.feature").MindSpeedFeature
        monkeypatch.setattr(env.package("mindspeed.features_manager"), "MindSpeedFeature", base, raising=False)
        yield env
    finally:
        monkeypatch.undo()
        # importlib adds entries outside monkeypatch; remove them before restoring
        # the original objects and parent attributes (including CI's live classes).
        for name in list(sys.modules):
            if (
                name == "mindspeed"
                or name.startswith("mindspeed.")
                or name == "mindspeed_llm"
                or name.startswith("mindspeed_llm.")
            ):
                sys.modules.pop(name)
        sys.modules.update(saved)


@pytest.mark.parametrize(
    "targets,experts,dispatcher,extend,error",
    load_cases(__file__, 'test_lora_conflicts', 'targets,experts,dispatcher,extend,error'),
)
def test_lora_conflicts(feature_env, targets, experts, dispatcher, extend, error):
    """Author: miku1122; Date: 2026-10-13; Description: LoRA/MoE conflicts."""
    feature = feature_env.feature("finetune.lora", "LoraFeature")
    args = Namespace(
        lora_target_modules=targets, num_experts=experts, moe_token_dispatcher_type=dispatcher, moe_tp_extend_ep=extend
    )
    if error:
        with pytest.raises(AssertionError, match=error):
            feature.validate_args(args)
    else:
        feature.validate_args(args)


@pytest.mark.parametrize(
    "overrides,error",
    load_cases(__file__, 'test_num_layer_list_lifecycle', 'overrides,error'),
)
def test_num_layer_list_lifecycle(feature_env, overrides, error):
    """Author: miku1122; Date: 2026-10-13; Description: dynamic PP validation."""
    feature = feature_env.feature("pipeline_parallel.num_layer_list", "NumLayerListFeature")
    args = Namespace(
        num_layer_list="4,4",
        num_layers=8,
        pipeline_model_parallel_size=2,
        num_layers_per_virtual_pipeline_stage=None,
        schedules_method="default",
        noop_layers=None,
    )
    vars(args).update(overrides)
    if error == "mutually exclusive":
        with pytest.raises(ValueError, match=error):
            feature.pre_validate_args(args)
        assert args.num_layers == 8
        with pytest.raises(ValueError, match=error):
            feature.validate_args(args)
        return
    feature.pre_validate_args(args)
    assert args.num_layers == (len(args.num_layer_list.split(",")) if args.num_layer_list else 8)
    if error:
        with pytest.raises(ValueError, match=error):
            feature.validate_args(args)
    else:
        feature.validate_args(args)
    feature.post_validate_args(args)
    assert args.num_layers == 8
    # Reusing the same feature for a second parse must discard prior state.
    next_args = Namespace(num_layer_list=None, num_layers=12)
    feature.pre_validate_args(next_args)
    feature.post_validate_args(next_args)
    assert next_args.num_layers == 12
    assert not hasattr(next_args, "encoder_num_layers")


@pytest.mark.parametrize(
    "no_pad,variable,throughput",
    load_cases(__file__, 'test_finetune_variable_sequence_restore', 'no_pad,variable,throughput'),
)
def test_finetune_variable_sequence_restore(feature_env, no_pad, variable, throughput):
    """Author: miku1122; Date: 2026-10-13; Description: no-pad state restoration."""
    feature = feature_env.feature("finetune.finetune", "FinetuneFeature")
    args = Namespace(no_pad_to_seq_lengths=no_pad, variable_seq_lengths=variable, log_throughput=throughput)
    if no_pad and throughput:
        with pytest.warns(RuntimeWarning, match="accurate TFLOPS"):
            feature.pre_validate_args(args)
    else:
        feature.pre_validate_args(args)
    assert args.variable_seq_lengths == (False if no_pad else variable)
    feature.post_validate_args(args)
    assert args.variable_seq_lengths == (no_pad or variable)
    assert args.log_throughput == (throughput and not no_pad)


@pytest.mark.parametrize("cp,mtp", load_cases(__file__, 'test_mtp_state_is_restored', 'cp,mtp'))
def test_mtp_state_is_restored(feature_env, cp, mtp):
    """Author: miku1122; Date: 2026-10-13; Description: MTP/CP bypass restoration."""
    feature = feature_env.feature("transformer.mtp", "MultiTokenPredictionFeature")
    args = Namespace(context_parallel_size=cp, mtp_num_layers=mtp)
    feature.pre_validate_args(args)
    assert args.mtp_num_layers == (None if cp > 1 else mtp)
    feature.post_validate_args(args)
    assert args.mtp_num_layers == mtp
    args.mtp_num_layers = None
    feature.pre_validate_args(args)
    feature.post_validate_args(args)
    assert args.mtp_num_layers is None


@pytest.mark.parametrize(
    "world,ldt,cp,expected",
    load_cases(__file__, 'test_virtual_tp_world_size_restore', 'world,ldt,cp,expected'),
)
def test_virtual_tp_world_size_restore(feature_env, world, ldt, cp, expected):
    """Author: miku1122; Date: 2026-10-13; Description: VTP topology bypass."""
    feature = feature_env.feature("layerwise_disaggregated_training.vtp_feature", "VTPFeature")
    args = Namespace(
        world_size=world,
        layerwise_disaggregated_training=ldt,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        context_parallel_size=cp,
    )
    feature.pre_validate_args(args)
    assert args.world_size == expected
    feature.post_validate_args(args)
    assert args.world_size == world
    assert not hasattr(args, "_vtp_orig_world_size")


@pytest.mark.parametrize(
    "path,cls,values,error_type,message",
    load_cases(
        __file__, 'test_invalid_feature_configuration', 'path,cls,values,error_type,message', exceptions=('error_type',)
    ),
)
def test_invalid_feature_configuration(feature_env, path, cls, values, error_type, message):
    """Author: miku1122; Date: 2026-10-13; Description: actionable invalid configs."""
    with pytest.raises(error_type, match=message):
        feature_env.feature(path, cls).validate_args(Namespace(**values))


@pytest.mark.parametrize(
    "path,cls,values",
    load_cases(__file__, 'test_valid_feature_configuration', 'path,cls,values'),
)
def test_valid_feature_configuration(feature_env, path, cls, values):
    """Author: miku1122; Date: 2026-10-13; Description: accepted feature configs."""
    feature_env.feature(path, cls).validate_args(Namespace(**values))


def test_lora_parser_list_and_boolean_contracts(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: CLI list/type/defaults."""
    parser = ArgumentParser()
    feature_env.feature("finetune.lora", "LoraFeature").register_args(parser)
    defaults = parser.parse_args([])
    assert defaults.lora_target_modules == []
    assert (defaults.lora_r, defaults.lora_alpha) == (16, 32)
    assert defaults.lora_register_forward_hook == ["word_embeddings", "input_layernorm"]
    args = parser.parse_args(
        ["--lora-target-modules", "linear_qkv", "linear_proj", "--lora-r", "8", "--lora-fusion", "--lora-ckpt-filter"]
    )
    assert args.lora_target_modules == ["linear_qkv", "linear_proj"]
    assert args.lora_r == 8 and args.lora_fusion and args.lora_ckpt_filter
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--lora-r", "invalid"])
    assert error.value.code == 2

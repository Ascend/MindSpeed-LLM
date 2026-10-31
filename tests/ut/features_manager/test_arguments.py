# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; public CLI contracts across feature groups.
Remarks: ArgumentParser is real; inherited MindSpeed features are real.
"""

from argparse import ArgumentParser

import pytest

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


CASES = load_cases(__file__, 'test_argument_defaults_and_overrides', 'path,cls,argv,defaults,expected')


@pytest.mark.parametrize("path,cls,argv,defaults,expected", CASES, ids=[c[1] for c in CASES])
def test_argument_defaults_and_overrides(feature_env, path, cls, argv, defaults, expected):
    """Author: miku1122; Date: 2026-10-13; Description: typed public CLI contract."""
    parser = ArgumentParser()
    feature_env.feature(path, cls).register_args(parser)
    base = vars(parser.parse_args([]))
    args = vars(parser.parse_args(argv))
    assert {key: base[key] for key in defaults} == defaults
    assert {key: args[key] for key in expected} == expected


@pytest.mark.parametrize(
    "path,cls,argv",
    load_cases(__file__, 'test_argument_rejects_invalid_values', 'path,cls,argv'),
)
def test_argument_rejects_invalid_values(feature_env, path, cls, argv):
    """Author: miku1122; Date: 2026-10-13; Description: argument rejects invalid values.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    parser = ArgumentParser()
    feature_env.feature(path, cls).register_args(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(argv)
    assert error.value.code == 2


def test_extensions_preserve_existing_choices(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: extensions preserve existing choices.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    parser = ArgumentParser()
    original = ["rope", "none"]
    parser.add_argument("--position-embedding-type", choices=original.copy())
    parser.add_argument("--optimizer", choices=("adam", "sgd"))
    parser.add_argument("--moe-router-score-function", choices=["softmax", "sigmoid"])
    parser.add_argument("--moe-router-load-balancing-type", choices=["aux_loss", "none"])
    parser.add_argument("--apply-wd-to-qk-layernorm", action="store_true")
    for path, cls in [
        ("transformer.flash_attention.alibi_feature", "AlibiFeature"),
        ("transformer.multi_latent_attention.csa_feature", "CSAFeature"),
        ("moe.moe_router", "MoERouter"),
        ("optimizer.muon_optimizer_feature", "MuonOptimizerFeature"),
    ]:
        feature_env.feature(path, cls).register_args(parser)
    assert parser.parse_args(["--position-embedding-type", "deepseek4"]).position_embedding_type == "deepseek4"
    assert parser.parse_args(["--position-embedding-type", "alibi"]).position_embedding_type == "alibi"
    assert parser.parse_args(["--optimizer", "dist_muon"]).optimizer == "dist_muon"
    assert (
        parser.parse_args(["--moe-router-score-function", "sqrtsoftplus"]).moe_router_score_function == "sqrtsoftplus"
    )
    assert original == ["rope", "none"]
    assert sum("--apply-wd-to-qk-layernorm" in a.option_strings for a in parser._actions) == 1
    assert parser.parse_args(["--muon-no-split-qkv"]).muon_split_qkv is False

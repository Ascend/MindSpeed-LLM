# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Validate case schemas without importing accelerator or training dependencies."""

import json
from pathlib import Path

import pytest


EXCEPTION_TYPES = {
    'AssertionError': AssertionError,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'FileNotFoundError': FileNotFoundError,
}


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f'Duplicate JSON field: {name}')
        result[name] = value
    return result


def load_cases(module_file, test_name, argnames, *, exceptions=(), tuple_values=()):
    """Return scalar/tuple rows with exact named fields in parameter order."""
    path = Path(module_file).with_suffix('.json')
    with path.open(encoding='utf-8') as stream:
        config = json.load(stream, object_pairs_hook=_unique_fields)
    if not isinstance(config, dict) or test_name not in config:
        raise ValueError(f'Missing JSON testcase: {test_name}')
    names = tuple(name.strip() for name in argnames.split(','))
    if not all(names) or len(set(names)) != len(names):
        raise ValueError('Parameter names must be nonempty and unique')
    if not set(exceptions).union(tuple_values) <= set(names):
        raise ValueError('Converted fields must be declared parameters')
    cases = config[test_name]
    if not isinstance(cases, list) or not cases:
        raise ValueError(f'Testcase must contain a nonempty case array: {test_name}')
    rows = []
    for case in cases:
        if not isinstance(case, dict) or tuple(case) != names:
            raise ValueError(f'Case fields must match parameter order: {test_name}: {names}')
        values = dict(case)
        for name in exceptions:
            symbol = values[name]
            if not isinstance(symbol, str) or symbol not in EXCEPTION_TYPES:
                raise ValueError(f'Unsupported exception type: {symbol}')
            values[name] = EXCEPTION_TYPES[symbol]
        for name in tuple_values:
            mapping = values[name]
            if not isinstance(mapping, dict) or any(not isinstance(v, list) for v in mapping.values()):
                raise ValueError(f'Tuple callback mapping must contain arrays: {name}')
            values[name] = {key: tuple(value) for key, value in mapping.items()}
        row = tuple(values[name] for name in names)
        rows.append(row[0] if len(names) == 1 else row)
    return rows


@pytest.mark.parametrize(
    'payload,argnames,options,message',
    load_cases(__file__, 'test_rejects_invalid_case_configuration', 'payload,argnames,options,message'),
)
def test_rejects_invalid_case_configuration(tmp_path, payload, argnames, options, message):
    """Author: miku1122; Date: 2026-10-13; Description: reject ambiguous JSON cases.

    Invalid inputs must fail collection/data loading rather than silently omit
    cases or change positional parameter meanings. No hardware is involved.
    """
    source = tmp_path / 'case.py'
    source.with_suffix('.json').write_text(payload, encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        load_cases(source, 'test_case', argnames, **options)

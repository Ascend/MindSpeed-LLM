# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
import pytest


# we still want to configure path argument by ourselves
# for different prefix_name of different scripts so we use this method.
# One more thing, as you can see, it has more scalibility.
def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--baseline-json", action="store", default=None,
                    help="Path to the baseline JSON file")
    parser.addoption("--generate-log", action="store", default=None,
                    help="Path to the generate log file")
    parser.addoption("--generate-json", action="store", default=None,
                    help="Path to the generate JSON file")


@pytest.fixture(autouse=True)
def baseline_json(request: pytest.FixtureRequest):
    return request.config.getoption("--baseline-json")


@pytest.fixture(autouse=True)
def generate_log(request: pytest.FixtureRequest):
    return request.config.getoption("--generate-log")


@pytest.fixture(autouse=True)
def generate_json(request: pytest.FixtureRequest):
    return request.config.getoption("--generate-json")

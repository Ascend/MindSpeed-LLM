import os

import pytest

from tests.test_tools.st_runner import discover_test_scripts, run_st_case, setup_st_environment


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(BASE_DIR, "baseline")
TEST_CI_SCRIPT = os.path.join(BASE_DIR, "..", "..", "test_tools", "test_ci_st.py")

test_scripts = discover_test_scripts(BASE_DIR, recursive=True, exclude_dirs={BASELINE_DIR})


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    setup_st_environment(BASE_DIR)


@pytest.mark.parametrize(
    "script_path", test_scripts, ids=lambda script_path: os.path.splitext(os.path.basename(script_path))[0]
)
def test_st_script(script_path):
    run_st_case(script_path, BASELINE_DIR, TEST_CI_SCRIPT)

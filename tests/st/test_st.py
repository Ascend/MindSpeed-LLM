import os
import subprocess
import tempfile
import pytest


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHELL_SCRIPTS_DIR = os.path.join(BASE_DIR, "shell_scripts")
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_results")

# Dynamically discover all test scripts
test_scripts = [f for f in os.listdir(SHELL_SCRIPTS_DIR) if f.endswith(".sh")]


@pytest.fixture(scope="session", autouse=True)
def setup_environment():
    """Set up environment and precompile operators"""
    # Set PYTHONPATH
    os.environ["PYTHONPATH"] = f"{BASE_DIR}:{os.environ.get('PYTHONPATH', '')}"

    # Precompile operators to improve execution stability
    ops_to_load = [
        "GMMOpBuilder",
        "GMMV2OpBuilder",
        "MatmulAddOpBuilder",
        "MoeTokenPermuteOpBuilder",
        "MoeTokenUnpermuteOpBuilder",
        "RotaryPositionEmbeddingOpBuilder",
        "GroupMatmulAddOpBuilder"
    ]

    for op_name in ops_to_load:
        cmd = ["python", "-c", f"'import mindspeed; from mindspeed.op_builder import {op_name}; {op_name}().load()'"]
        try:
            subprocess.run(cmd, shell=False, check=True, capture_output=True)
        except Exception as e:
            print(f"Warning: Failed to load {op_name}: {e}")


def run_test_script(script_path):
    """Run test script and return output"""
    result = subprocess.run(
        ["bash", script_path],
        shell=False,
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.parametrize("script_name", test_scripts)
def test_st_script(script_name):
    """Dynamically generated test case, run each test script and compare results"""
    script_path = os.path.join(SHELL_SCRIPTS_DIR, script_name)
    file_name_prefix = os.path.splitext(script_name)[0]

    print(f"Running test: {file_name_prefix}")

    # Run test script
    exit_code, stdout, stderr = run_test_script(script_path)

    # Check if script execution succeeded
    if exit_code != 0:
        print(f"\n=== Script {script_name} failed ===")
        print(f"Exit code: {exit_code}")
        print(f"=== Stdout ===\n{stdout}")
        print(f"=== Stderr ===\n{stderr}")
        pytest.fail(f"Script {script_name} failed with exit code {exit_code}")

    # Create temporary files to store logs and generated JSON
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(temp_dir, f"{file_name_prefix}.log")
        json_path = os.path.join(temp_dir, f"{file_name_prefix}.json")

        # Save log
        with open(log_path, "w") as f:
            f.write(stdout)

        # Run comparison script
        baseline_json = os.path.join(BASELINE_DIR, f"{file_name_prefix}.json")

        # Ensure baseline file exists
        assert os.path.exists(baseline_json), f"Baseline file not found: {baseline_json}"

        # Run test_ci_st.py for comparison
        test_ci_script = os.path.join(BASE_DIR, "..", "test_tools", "test_ci_st.py")

        compare_result = subprocess.run(
            [
                "python", "-m", "pytest",
                test_ci_script,
                "-x",
                f"--baseline-json={baseline_json}",
                f"--generate-log={log_path}",
                f"--generate-json={json_path}",
                "-v"
            ],
            capture_output=True,
            text=True
        )

        # Check comparison result
        if compare_result.returncode != 0:
            print(f"\n=== Comparison failed for {file_name_prefix} ===")
            print(f"=== Stdout ===\n{compare_result.stdout}")
            print(f"=== Stderr ===\n{compare_result.stderr}")
            pytest.fail(f"Comparison failed for {file_name_prefix}")
        print(f"Test {file_name_prefix} passed successfully!")

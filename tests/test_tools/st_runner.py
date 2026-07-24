import os
import subprocess
import tempfile


def discover_test_scripts(script_dir, recursive=False, exclude_dirs=None):
    """Discover shell test scripts."""
    exclude_dirs = set(exclude_dirs or [])
    if not recursive:
        return sorted(os.path.join(script_dir, file) for file in os.listdir(script_dir) if file.endswith(".sh"))

    test_scripts = []
    for root, _, files in os.walk(script_dir):
        if root in exclude_dirs:
            continue
        for file in files:
            if file.endswith(".sh"):
                test_scripts.append(os.path.join(root, file))
    return sorted(test_scripts)


def setup_st_environment(base_dir):
    """Set up environment and precompile operators."""
    os.environ["PYTHONPATH"] = f"{base_dir}:{os.environ.get('PYTHONPATH', '')}"

    ops_to_load = [
        "GMMOpBuilder",
        "GMMV2OpBuilder",
        "MatmulAddOpBuilder",
        "MoeTokenPermuteOpBuilder",
        "MoeTokenUnpermuteOpBuilder",
        "RotaryPositionEmbeddingOpBuilder",
        "GroupMatmulAddOpBuilder",
    ]

    for op_name in ops_to_load:
        cmd = ["python", "-c", f"'import mindspeed; from mindspeed.op_builder import {op_name}; {op_name}().load()'"]
        try:
            subprocess.run(cmd, shell=False, check=True, capture_output=True)
        except Exception as e:
            print(f"Warning: Failed to load {op_name}: {e}")


def run_test_script(script_path):
    """Run test script and return output."""
    result = subprocess.run(["bash", script_path], shell=False, check=False, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def run_st_case(script_path, baseline_dir, test_ci_script):
    """Run one ST script and compare its result with baseline."""
    import pytest

    script_name = os.path.basename(script_path)
    file_name_prefix = os.path.splitext(script_name)[0]

    print(f"Running test: {file_name_prefix}")

    exit_code, stdout, stderr = run_test_script(script_path)
    if exit_code != 0:
        print(f"\n=== Script {script_name} failed ===")
        print(f"Exit code: {exit_code}")
        print(f"=== Stdout ===\n{stdout}")
        print(f"=== Stderr ===\n{stderr}")
        pytest.fail(f"Script {script_name} failed with exit code {exit_code}")

    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = os.path.join(temp_dir, f"{file_name_prefix}.log")
        json_path = os.path.join(temp_dir, f"{file_name_prefix}.json")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(stdout)

        baseline_json = os.path.join(baseline_dir, f"{file_name_prefix}.json")
        assert os.path.exists(baseline_json), f"Baseline file not found: {baseline_json}"

        compare_result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                test_ci_script,
                "-x",
                f"--baseline-json={baseline_json}",
                f"--generate-log={log_path}",
                f"--generate-json={json_path}",
                "-v",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        if compare_result.returncode != 0:
            print(f"\n=== Comparison failed for {file_name_prefix} ===")
            print(f"=== Stdout ===\n{compare_result.stdout}")
            print(f"=== Stderr ===\n{compare_result.stderr}")
            pytest.fail(f"Comparison failed for {file_name_prefix}")
        print(f"Test {file_name_prefix} passed successfully!")

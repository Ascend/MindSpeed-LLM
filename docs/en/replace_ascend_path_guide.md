# MindSpeed LLM Ascend HDK Path Batch Replacement Guide

## Background

The Docker mount configurations and runtime scripts in the MindSpeed LLM repository contain hardcoded references to the `/usr/local/Ascend/driver/` path.
On some machines, the actual HDK installation path is `/usr/local/npu/driver/`. Therefore, complete the batch replacement before using the repository to ensure that HDK-related mounts and library loading work correctly.

> Note: This replacement applies **only to the HDK path** (`/usr/local/Ascend/driver/`). Other subpaths such as CANN, ascend-toolkit, and nnal/atb remain unchanged.

This guide provides the complete steps for performing a batch path replacement using the `replace_ascend_path.py` script, and describes the adaptation requirements for the return value of the `dcmi_get_device_chip_info` interface on some versions.

---

## Prerequisites

- Python 3.10+
- Read/write permission to the repository directory
- You are advised to commit or back up the current state via `git` before performing the replacement

---

## Affected File Scope

| File Type | Description | Typical Path Examples |
|---------|------|-------------|
| Shell scripts (`.sh`) | Various runtime scripts, including but not limited to: data preprocessing, weight conversion, pretraining, fine-tuning, evaluation, inference, testing | `examples/*/*.sh`, `tests/*/*.sh` |
| Markdown documents (`.md`) | All documentation, including but not limited to: installation guides, quick-start guides, task-specific guides, feature descriptions | `docs/zh/install_guide.md`, `docker/OVERVIEW.md`, `docs/zh/pytorch/*/*.md` |
| RST documents (`.rst`) | reStructuredText-style documentation | `docs/*/*.rst` |
| TXT documents (`.txt`) | Plain text description files or configuration notes | `requirements.txt` |
| Python files (`.py`) | Source code (if it contains path references) | Source files of each module |
| Dockerfile | Docker image build scripts | `docker/Dockerfile` |

> Path variation notes: This replacement only covers driver-related path references, for example:
>
> - `/usr/local/Ascend/driver/lib64/` (Docker mount path, the most common)
> - `/usr/local/Ascend/driver/` (HDK installation root path)
>
> The following paths are **not** within the replacement scope and remain unchanged:
>
> - `/usr/local/Ascend/cann/set_env.sh` (environment variable initialization)
> - `/usr/local/Ascend/ascend-toolkit/set_env.sh` (Ascend Toolkit initialization)
> - `/usr/local/Ascend/nnal/atb/set_env.sh` (ATB library initialization)

---

## Usage Steps

### 1. Entering the Repository Root Directory

```bash
cd /path/to/MindSpeed-LLM
```

### 2. Previewing the Changes to Be Made (Recommended)

Before making actual changes, confirm the scope of changes in `--dry-run` mode:

```bash
python3 tests/tools/replace_ascend_path.py --dry-run
```

Sample output:

```bash
[DRY RUN] Path replacement: /usr/local/Ascend/driver -> /usr/local/npu/driver
Scan directory : /path/to/MindSpeed-LLM
File types     : .md, .py, .rst, .sh, .txt + Dockerfile
------------------------------------------------------------
Found XXX candidate file(s), processing...

  [would replace   1] docker/Dockerfile
  [would replace   2] docker/OVERVIEW.md
  [would replace   2] docker/OVERVIEW.zh.md
  ...

============================================================
[DRY RUN] XXX file(s) would be modified, XXX replacement(s) total.
          Remove --dry-run to apply changes.
```

### 3. Performing the Batch Replacement

After confirming the preview is correct, perform the actual replacement:

```bash
# Default: replace /usr/local/Ascend/driver with /usr/local/npu/driver
python3 tests/tools/replace_ascend_path.py
```

After execution completes, the script outputs the number of modified files and the total number of replacements.

### 4. Verifying the Replacement Results

```bash
# Check for any remaining unreplaced driver paths (the result should be 0)
grep -r "/usr/local/Ascend/driver" . \
--include='.sh' \
--include='.md' \
--include='.rst' \
--include='.py' \
--include='.txt' \
--include='Dockerfile' \
--exclude='replace_ascend_path.py' \
--exclude='replace_ascend_path_guide.md' \
--exclude-dir='.git' \
| wc -l
```

---

## Post-Replacement Verification

### 1. Verifying Driver Path Loading

```bash
# Verify that the driver directory exists under the new path
ls /usr/local/npu/driver/lib64/

# Load environment variables (the ascend-toolkit path is unchanged; the original path is still used)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Verify that the environment variables take effect
echo $ASCEND_HOME_PATH
```

### 2. Verifying Component Installation

```bash
# Verify that MindSpeed LLM is installed successfully
python3 -c "import mindspeed_llm; print('MindSpeed LLM installed successfully')"

# Verify that the NPU is available
python3 -c "import torch_npu; print('NPU available:', torch_npu.npu.is_available())"
```

### 3. Verifying the Chip Information Interface (`dcmi_get_device_chip_info`)

Some versions require the chip model identifier returned by the `dcmi_get_device_chip_info` interface to be `A2G3` or `A2G4`.

> Note: The current MindSpeed LLM code does not call this interface directly. This is provided here only as an adaptation note. If upper-layer service or operation scripts rely on the return value of this interface to determine the chip model, the return value must be `A2G3` or `A2G4`. Otherwise, model-related logic branches may be affected.

For the verification method, please refer to:
[dcmi_get_device_chip_info interface prototype](https://support.huawei.com/enterprise/zh/doc/EDOC1100568435/8739bb5a)

### 4. Core Functionality Smoke Verification

Refer to the README of the corresponding model for configuration, and verify that the training process can start normally.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Run the example script (based on the specific model)
bash examples/<model_name>/pretrain_<model_name>.sh
```

---

## Complete Script Parameter Reference

```bash
usage: replace_ascend_path.py [-h] [--source SOURCE] [--target TARGET]
                               [--dir DIR] [--extensions EXT [EXT ...]]
                               [--dry-run]

Options:
  -h, --help            Show help information
  --source SOURCE       Source path (default: /usr/local/Ascend/driver)
  --target TARGET       Target path (default: /usr/local/npu/driver)
  --dir DIR             Directory to scan (default: current directory .)
  --extensions EXT...   File extension whitelist (default: .sh .md .rst .py .txt + Dockerfile)
  --dry-run             Preview changes only, do not modify files
```

# MindSpeed LLM Docker Image Overview

## Quick Reference

| Item | Description |
| ------ | ------ |
| **Image Name** | mindspeed-llm |
| **Maintainer** | MindSpeed LLM Team |
| **Source Repository** | [https://gitcode.com/Ascend/MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) |
| **Dockerfile Path** | `docker/` |
| **License** | Apache-2.0 |

## Image Tag Key Field Description

Image Tag naming format: `{Version}-{ChipType}-{OS}-py{PythonVersion}-{Architecture}`

| Field | Description | Example Value |
| ------ | ------ | -------- |
| Version | MindSpeed LLM version label, also serves as Git branch name | 26.0.0 |
| ChipType | NPU chip type (lowercase) | a3, 910b |
| OS | Operating system version | openeuler24.03 |
| PythonVersion | Python runtime version | py3.11 |
| Architecture | CPU architecture type | aarch64 |

### Tag Examples

| Tag | NPU | Operating System | Python | Architecture |
| ----- | ----- | --------- | -------- | ------ |
| `26.0.0-a3-openeuler24.03-py3.11-aarch64` | A3 | openEuler 24.03 | 3.11 | aarch64 |
| `26.0.0-910b-openeuler24.03-py3.11-aarch64` | 910B | openEuler 24.03 | 3.11 | aarch64 |

## Dockerfile Archive Path

| NPU | Operating System | Dockerfile Path |
| ----- | --------- | ---------------- |
| A3 | openEuler | `docker/Dockerfile` |
| 910B | openEuler | `docker/Dockerfile` |

Dockerfile naming format: `Dockerfile[.{Chip}.{OS}.{ExtraFields}]`

- Unified Dockerfile supports all NPU models and operating systems via build arguments
- Separator between fields: `.`
- Separator inside single field: `-`
- Chip names use lowercase: a3, 910b
- OS follows standard naming convention: openEuler

## Project Directory Structure Specification

```text
docker/
├── Dockerfile                 # Universal Dockerfile for multi-NPU
├── image_build.sh             # Image build script
├── configure_yum_repo.sh      # YUM repository configuration script
├── OVERVIEW.md                # English overview document
├── OVERVIEW.zh.md             # Chinese overview document
```

## Quick Start

### 1. Image Build Guide

#### Custom Base Image Building

The `image_build.sh` script supports flexible parameter configuration. Default values are for reference only and can be adjusted as needed.

| Parameter | Description | Default Value (Example) |
| ------ | ------ | ------------ |
| `-t, --npu-type` | NPU type: A3 or 910B | Required |
| `-o, --os` | Operating system: openeuler24.03 | openeuler24.03 |
| `-v, --version` | MindSpeed LLM version, also used as Git branch name and script directory identifier | 26.0.0 |
| `--torch-version` | PyTorch version | 2.9.0 |
| `--torch-npu-version` | torch-npu version | 2.9.0 |
| `--base-image-version` | CANN version of base image | 9.0.0-beta.2 |

### Basic Build Examples

```bash
cd docker

# Build 910B image (default)
bash image_build.sh -t 910B

# Build A3 image
bash image_build.sh -t A3

# Build A3 + openEuler image
bash image_build.sh -t A3 -o openEuler24.03

# Build with specified PyTorch version
bash image_build.sh -t A3 --torch-version 2.9.0 --torch-npu-version 2.9.0

# Build with specified CANN base image version
bash image_build.sh -t A3 --base-image-version 9.0.0-beta.2

# Build with specified LLM version
bash image_build.sh -t A3 -v 26.0.0
```

#### Automatic Download Function Description

The build script supports automatic downloading of the following resources. Please ensure the network is unobstructed.

**Base Image**: Automatically pull the specified `--base-image` when it does not exist locally.

### 2. Image Usage Instructions

**Important Note**: Due to different dependency environments of various models, only basic torch and torch_npu dependency packages are pre-installed in the image. After pulling the image and starting the container, users need to manually install dependencies required by the target model in the base environment according to the model README file.

#### Run Image

```bash
# Basic run
docker run -it --rm \
    mindspeed-llm:26.0.0-910b-openeuler24.03-py3.11-aarch64 bash

# Run with NPU device (Example: /dev/davinci1)
# Assume NPU device /dev/davinci1 and NPU driver installed at /usr/local/Ascend
docker run -it --rm \
    --name mindspeed-llm \
    --privileged \
    --network host \
    --ipc=host \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/:/home/ \
    -v /data:/data \
    -v /mnt:/mnt \
    mindspeed-llm:26.0.0-910b-openeuler24.03-py3.11-aarch64 \
    /bin/bash
```

#### Built-in Environment

The image contains the following pre-configured environment:

| Environment | Description | Working Directory |
| ------ | ------ | --------- |
| base | Basic environment including PyTorch, torch_npu, MindSpeed LLM, MindSpeed and Megatron-LM | /workspace/MindSpeed-LLM |

**Environment Notes**:

- Due to varying dependency requirements across different models, only basic torch and torch_npu packages are pre-installed in the image.
- After pulling the image and starting the container, users shall manually install model-specific dependencies in the base environment according to the README of the target model.

## Secondary Development

Create a custom Dockerfile based on this image:

```dockerfile
FROM mindspeed-llm:26.0.0-910b-openeuler24.03-py3.11-aarch64

RUN pip install your-package==1.0.0

COPY . /workspace/your-project

WORKDIR /workspace/your-project
```

Build and run (Example: /dev/davinci1):

```bash
docker build -t my-mindspeed-app:latest .
docker run -it --rm \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    my-mindspeed-app:latest bash
```

### Software Stack

| Component | Version |
| ------ | ------ |
| CANN | 9.0.0-beta.2 |
| Python | 3.11 |
| Miniconda | 26.1.1-1 |
| PyTorch | 2.9.0 |
| torch_npu | 2.9.0 |
| MindSpeed LLM | 26.0.0 |

### Compatibility Change Notes

#### Version 26.0.0

- Initial release
- Based on CANN 9.0.0-beta.2
- PyTorch 2.9.0 + torch_npu 2.9.0
- Python 3.11 (Miniconda 26.1.1-1)
- Supports openEuler 24.03

## License

MindSpeed LLM is released under the Apache License 2.0. See the [LICENSE](../LICENSE) file for details.

Like all Docker images, this image may contain other software subject to separate license agreements, such as Bash from the base system and all direct and indirect dependencies of integrated core software.

Users of pre-built images shall be responsible for ensuring that all usage of the image complies with the license requirements of all included software components.

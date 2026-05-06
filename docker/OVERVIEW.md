# MindSpeed LLM Docker Image Overview

## Quick Reference

| Item | Description |
| ------ | ------ |
| **Image Name** | mindspeed-llm |
| **Maintainer** | MindSpeed LLM Team |
| **Source Repository** | [https://gitcode.com/Ascend/MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) |
| **Dockerfile Path** | `docker/Dockerfile` |
| **License** | Apache-2.0 |

## Image Tag Key Field Description

Image Tag naming format: `{Version}-{ChipType}-{OS}-py{PythonVersion}-{Architecture}`

| Field | Description | Example Value |
| ------ | ------ | -------- |
| Version | MindSpeed LLM version label, also serves as Git branch name | `master`, `26.0.0` |
| ChipType | NPU chip type (lowercase) | `a3`, `910b` |
| OS | Operating system version | `openeuler24.03`, `ubuntu22.04` |
| PythonVersion | Python runtime version | `3.11` |
| Architecture | CPU architecture type | `aarch64`, `x86_64` |

### Tag Examples

| Tag | NPU | Operating System | Python | Architecture |
| ----- | ----- | --------- | -------- | ------ |
| `master-a3-openeuler24.03-py3.11-aarch64` | `a3` | `openeuler24.03` | `3.11` | `aarch64` |
| `26.0.0-910b-ubuntu22.04-py3.11-x86_64` | `910b` | `ubuntu22.04` | `3.11` | `x86_64` |

## Dockerfile Archive Path

`docker/Dockerfile`

## Project Directory Structure Specification

### Directory Structure

```text
docker/
├── Dockerfile                 # Universal Dockerfile for multi-NPU
├── image_build.sh             # Image build script
├── configure_yum_repo.sh      # YUM repository configuration script
├── configure_apt_repo.sh      # Apt repository configuration script
├── OVERVIEW.md                # English overview document
├── OVERVIEW.zh.md             # Chinese overview document
```

## Quick Start

### 1. Image Build Guide

#### Custom Base Image Building

The `image_build.sh` script supports flexible parameter configuration. Default values are for reference only and can be adjusted as needed.

| Parameter | Description                                  | Default (Example) |
| ------ |-------------------------------------| ------------ |
| `-t, --npu-type` | NPU type:`a3` or `910b`                | `910b` |
| `-o, --os` | OS：`openeuler24.03`or`ubuntu22.04` | `openeuler24.03` |
| `--mindspeed-llm-branch` |MindSpeed LLM version tag, also used as Git branch name    | `26.0.0` |
| `--mindspeed-branch` | MindSpeed version tag, also used as Git branch name        | `26.0.0_core_r0.12.1` |
| `--megatron-branch` | Megatron-LM version tag, also used as Git branch name      | `core_v0.12.1` |
| `--python-version` | Python version                           | `3.11` |
| `--torch-version` | PyTorch version                          | `2.7.1` |
| `--torch-npu-version` | torch-npu version                        | `2.7.1` |
| `--base-image-version` | Base image CANN version                        | `8.5.2` |
| `--base-image` | Full base image name, passed as-is to pull the image if not empty           | None |

**Note:** The current NPU types are A2 and A3; A5 is yet to be built.

### Basic Build Examples

```bash
cd docker

# Build 910B image (default)
bash image_build.sh

# Build a3 image
bash image_build.sh -t a3

# Build a3 + openEuler image
bash image_build.sh -t a3 -o openEuler24.03

# Build with specified PyTorch version
bash image_build.sh -t a3 --torch-version 2.7.1 --torch-npu-version 2.7.1

# Build a3 + specified CANN base image version
bash image_build.sh -t a3 --base-image-version 8.5.2

# Build a3 + specified version
bash image_build.sh -t a3 --mindspeed-llm-branch 26.0.0 --mindspeed-branch 26.0.0_core_r0.12.1 --megatron-branch core_v0.12.1
```

#### Automatic Download Function Description

The build script supports automatic downloading of the following resources. Please ensure a stable network connection:

**Base Image:** Automatically fetches the image if `--base-image` is specified and it does not exist locally. The chip information in the image tag and CANN base image name must be lowercase, such as `a3` and `910b`. The complete `--base-image` will be passed as is, therefore the tag must be exactly the same as the published CANN image name.

```bash
# Specify the base image
cd docker
bash image_build.sh \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.2-910b-openeuler24.03-py3.11
```

### 2. Image Usage Instructions

**Important Note**: Due to different dependency environments of various models, only basic torch and torch_npu dependency packages are pre-installed in the image. After pulling the image and starting the container, users need to manually install dependencies required by the target model in the base environment according to the model README file.

#### Run Image

Image names use the `REPOSITORY:TAG` from `docker images`, for example, `mindspeed-llm:master-910b-openeuler24.03-py3.11-aarch64`.

```bash
# Basic run
docker run -it --rm \
    mindspeed-llm:master-910b-openeuler24.03-py3.11-aarch64 bash

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
    mindspeed-llm:master-910b-openeuler24.03-py3.11-aarch64 \
    /bin/bash

# Enter the running container
docker exec -it mindspeed-llm /bin/bash
```

#### Built-in Environment

The image contains the following pre-configured environment:

| Environment | Description | Working Directory |
| ------ | ------ | --------- |
| base | Basic environment including `PyTorch`、`torch_npu`、`MindSpeed LLM`、`MindSpeed`、`Megatron-LM` | `/workspace/MindSpeed-LLM` |

## Secondary Development

Create a custom Dockerfile based on this image:

```dockerfile
FROM mindspeed-llm:master-910b-openeuler24.03-py3.11-aarch64

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
| CANN | 8.5.2 |
| Python | 3.11 |
| Miniconda | 26.1.1-1 |
| PyTorch | 2.7.1 |
| torch_npu | 2.7.1 |
| MindSpeed LLM | 26.0.0 |

### Compatibility Change Notes

- The current version uses a unified Dockerfile + build script structure and supports configurable CANN base image selection.
- The default base image uses `CANN 8.5.2`, `910b`, `openEuler24.03`, and `Python3.11`.
- You can switch to `ubuntu22.04`, `a3`, or other `CANN` base image versions via `docker/image_build.sh`.
- `MindSpeed-LLM` is cloned to `/MindSpeed-LLM`, `MindSpeed` is cloned to `/MindSpeed`, and `Megatron-LM` is cloned to `/Megatron-LM`.
- The image installs `PyTorch`, `torch_npu`, `MindSpeed-LLM`, `MindSpeed`, `Megatron-LM`, and the `Python` dependency from `requirements.txt`.

## License

MindSpeed LLM is released under the Apache License 2.0. See the [LICENSE](../LICENSE) file for details.

Like all Docker images, this image may contain other software subject to separate license agreements, such as Bash from the base system and all direct and indirect dependencies of integrated core software.

Users of pre-built images shall be responsible for ensuring that all usage of the image complies with the license requirements of all included software components.

# MindSpeed LLM Docker Image Overview

## Quick Reference

| Item | Description |
| ------ | ------ |
| **Image Name** | mindspeed-llm |
| **Maintainer** | MindSpeed LLM Team |
| **Source Repository** | [https://gitcode.com/Ascend/MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) |
| **Dockerfile Path** | `docker/Dockerfile` |
| **License** | Apache-2.0 |
| **Where to get help** | [Issue Feedback](https://gitcode.com/Ascend/MindSpeed-LLM/issues) |

## MindSpeed-LLM

MindSpeed-LLM is a distributed training suite for large language models tailored to the Huawei Atlas ecosystem. It delivers end-to-end LLM training solutions for ecosystem partners of Huawei Atlas chips. The suite supports distributed pre-training and distributed instruction fine-tuning, and comes with a full development toolchain encompassing data preprocessing, weight conversion, online inference, baseline evaluation and more core capabilities.

## Supported Tags and Dockerfile Links

All MindSpeed-LLM tags follow this format: `v{MindSpeed LLM Version}-cann{CANN Version}-torch_npu{TorchNPU Version}-{ChipType}-{OS}-py{Python Version}`

| Field | Example Value | Description |
| ------ | ------ | -------- |
| MindSpeed LLM Version | `26.1.0` | MindSpeed LLM version label, also serves as Git branch name |
| CANN Version | `9.1.0` | CANN base image version |
| TorchNPU Version | `2.7.1.post8` | TorchNPU package version |
| Chip Type | `910b`, `a3`, `950` | NPU chip type (lowercase) |
| OS | `openeuler24.03`, `ubuntu22.04` | Operating system version |
| Python Version | `3.12` | Python runtime version |

> The latest tags in the image registry are multi-architecture images combining `x86_64` and `aarch64`, so they do not include an `-x86_64` or `-aarch64` architecture suffix. When the image is built locally from the Dockerfile, the build script still generates a tag with the host architecture suffix by default.

### Latest Version 26.1.0

The following table lists all multi-architecture image tags for the latest MindSpeed LLM 26.1.0 release. Each tag combines `x86_64` and `aarch64` images. For all historical tags, see [Supported Tags](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/supported_tags.md).

| Tag | Dockerfile | Content |
| --- | --- | --- |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |

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
├── supported_tags.md          # Published and historical image tags
├── OVERVIEW.md                # English overview document
├── OVERVIEW.zh.md             # Chinese overview document
```

## Quick Start

### How to Build Locally

#### Custom Base Image Building

The `image_build.sh` script supports flexible parameter configuration. Its defaults are aligned with the latest published image tags and can be overridden as needed.

| Parameter                 | Description                                  | Default Value |
|---------------------------|-------------------------------------| ------------ |
| `-t, --npu-type`          | NPU type: `910b`, `a3`, or `950`                | `910b` |
| `-o, --os`                | OS：`openeuler24.03`or`ubuntu22.04` | `openeuler24.03` |
| `--no-cache`              | Build without using Docker build cache                          | None |
| `--mindspeed-llm-branch`  |MindSpeed LLM version tag, also used as Git branch name    | `26.1.0` |
| `--mindspeed-branch`      | MindSpeed version tag, also used as Git branch name        | `26.1.0_core_r0.12.1` |
| `--megatron-branch`       | Megatron-LM version tag, also used as Git branch name      | `core_v0.12.1` |
| `--python-version`        | Python version                           | `3.12` |
| `--torch-version`         | PyTorch version                          | `2.7.1` |
| `--torch-npu-version`     | TorchNPU package version                   | `2.7.1.post8` |
| `--triton-ascend-version` | Triton-Ascend version                        | `3.2.2` |
|  `--fla-npu-branch`       | flash-linear-attention-npu version tag, also used as Git branch name       | `v26.1.0` |
| `--base-image-version`    | Base image CANN version                        | `9.1.0` |
| `--base-image`            | Full base image name, passed as-is to pull the image if not empty           | None |
| `--cleanup-on-fail`       | Clean up dangling images/containers when build fails           | None |

**Note:** The latest published image tags and `image_build.sh` cover `910b` (Atlas A2 training products), `a3` (Atlas A3 training products), and `950` (Ascend 950 series products).

### Basic Build Examples

Only pass the parameters that need to be changed. Any omitted parameters use the defaults listed above.

```bash
cd docker

# Use all defaults (910b + openEuler24.03)
bash image_build.sh

# Customize the NPU type and operating system
bash image_build.sh -t 950 -o ubuntu22.04

# Customize the CANN, PyTorch, and TorchNPU package versions
bash image_build.sh \
  --base-image-version 9.1.0 \
  --torch-version 2.7.1 \
  --torch-npu-version 2.7.1.post8

# Change the source branches
bash image_build.sh \
  --mindspeed-llm-branch 26.1.0 \
  --mindspeed-branch 26.1.0_core_r0.12.1 \
  --megatron-branch core_v0.12.1

# Change the output image name
bash image_build.sh -i myproject/mindspeed-llm:custom
```

#### Automatic Download Function Description

The build script supports automatic downloading of the following resources. Please ensure a stable network connection:

**Base Image:** Automatically fetches the image if `--base-image` is specified and it does not exist locally. The chip information in the image tag and CANN base image name must be lowercase, such as `910b`, `a3`, and `950`. The complete `--base-image` will be passed as is, therefore the tag must be exactly the same as the published CANN image name. When `--npu-type` is omitted, the script automatically detects these three NPU types from the base image tag.

```bash
# Specify a 910b base image; the script automatically detects the NPU type
cd docker
bash image_build.sh \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.12
```

#### flash-linear-attention-npu Ops Build

During image build, the Dockerfile sources the CANN environment after cloning `flash-linear-attention-npu`, then builds and installs the GDN custom operator run package and the `torch_custom/fla_npu` wheel.

The FLA NPU `--soc` value is mapped from the selected NPU type by default:

| NPU type | FLA NPU `--soc` |
| ------ | ------ |
| `910b` | `ascend910b` |
| `a3` | `ascend910_93` |
| `950` | `ascend950` |

Override the mapping if needed:

```bash
bash image_build.sh --fla-npu-soc ascend910_93
```

The FLA NPU operator list is maintained in the `FLA_NPU_OPS` array in `docker/image_build.sh`. Add new operator names to that array, and the script will convert it to the comma-separated value required by `build.sh --ops`.

### Run a MindSpeed LLM Container

**Important Note**: Due to different dependency environments of various models, only basic PyTorch and TorchNPU dependency packages are pre-installed in the image. After pulling the image and starting the container, users need to manually install dependencies required by the target model in the base environment according to the model README file.

#### Run the Container

Image names use the `REPOSITORY:TAG` from `docker images`, for example, `mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12`.

```bash
# Basic run
docker run -it --rm \
    mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12 bash

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
    mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12 \
    /bin/bash

# Enter the running container
docker exec -it mindspeed-llm /bin/bash
```

#### Built-in Environment

The image contains the following pre-configured environment:

| Environment | Description | Working Directory |
| ------ | ------ | --------- |
| base | Basic environment including `PyTorch`,`TorchNPU`,`MindSpeed LLM`,`MindSpeed`,`Megatron-LM`,`FSDPTurbo`,`Triton-Ascend` | `/workspace/MindSpeed-LLM` |

## Secondary Development

Create a custom Dockerfile based on this image:

```dockerfile
FROM mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12

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
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    my-mindspeed-app:latest bash
```

### Software Stack

| Component | Version |
| ------ | ------ |
| CANN | 9.1.0 |
| Python | 3.12 |
| PyTorch | 2.7.1 |
| TorchNPU | 26.1.0 |
| torch-npu package | 2.7.1.post8 |
| Triton-Ascend | 3.2.2 |
| MindSpeed LLM | 26.1.0 |

### Compatibility Change Notes

- The current version uses a unified Dockerfile + build script structure and supports configurable CANN base image selection.
- The latest published images use `CANN 9.1.0`, `TorchNPU 2.7.1.post8`, and `Python 3.12`.
- Published image variants cover `910b`, `a3`, and `950` on both `openEuler24.03` and `ubuntu22.04`. Registry tags combine `x86_64` and `aarch64` images through a multi-architecture manifest and omit the architecture suffix, while locally built images retain the host architecture suffix by default.
- `MindSpeed-LLM` is cloned to `/workspace/MindSpeed-LLM`, `MindSpeed` is cloned to `/workspace/MindSpeed`, and `Megatron-LM` is cloned to `/workspace/Megatron-LM`.
- The image installs `PyTorch`, `TorchNPU`, `MindSpeed-LLM`, `MindSpeed`, `Megatron-LM`, and the `Python` dependency from `requirements.txt`.

## License

MindSpeed LLM is released under the Apache License 2.0. See the [LICENSE](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/LICENSE) file for details.

Like all Docker images, this image may contain other software subject to separate license agreements, such as Bash from the base system and all direct and indirect dependencies of integrated core software.

Users of pre-built images shall be responsible for ensuring that all usage of the image complies with the license requirements of all included software components.

## Disclaimer

The released Ascend software images are community versions and are not intended for commercial accountability. They are provided solely as references for production practices.

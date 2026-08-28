# MindSpeed LLM Docker 镜像概述

## 快速参考

| 项目 | 说明 |
| ------ | ------ |
| **镜像名称** | mindspeed-llm |
| **维护者** | MindSpeed LLM 团队 |
| **源码仓库** | [https://gitcode.com/Ascend/MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) |
| **Dockerfile 路径** | `docker/Dockerfile` |
| **许可证** | Apache-2.0 |
| **问题反馈** | [Issue Feedback](https://gitcode.com/Ascend/MindSpeed-LLM/issues) |

## MindSpeed-LLM

MindSpeed LLM：基于昇腾生态的大语言模型分布式训练套件，旨在为华为昇腾芯片生态合作伙伴提供端到端的大语言模型训练方案，包含分布式预训练、分布式指令微调以及对应的开发工具链，如：数据预处理、权重转换、在线推理、基线评估等。

## 支持的 Tags 及 Dockerfile 链接

MindSpeed-LLM 所有 Tag 遵循以下格式：`v{MindSpeed LLM版本}-cann{CANN版本}-torch_npu{TorchNPU版本}-{芯片信息}-{操作系统}-py{Python版本}`

| 字段 | 示例值 | 说明 |
| ------ | ------ | -------- |
| MindSpeed LLM版本 | `26.1.0` | MindSpeed LLM 版本标识，同时也是 Git 分支名称 |
| CANN版本 | `9.1.0` | CANN 基础镜像版本 |
| TorchNPU版本 | `2.7.1.post8` | TorchNPU 安装包版本 |
| 芯片信息 | `910b`, `a3`, `950` | NPU 芯片类型（小写） |
| 操作系统 | `openeuler24.03`, `ubuntu22.04` | 操作系统类型 |
| Python版本 | `3.12` | Python 版本 |

> 镜像仓库中的最新 Tag 为 `x86_64` 与 `aarch64` 二合一的多架构镜像，**不包含** `-x86_64` 或 `-aarch64` 架构后缀。通过 Dockerfile 在本地构建镜像时，构建脚本默认仍会生成带宿主机架构后缀的 Tag。

### 最新版本 26.1.0

如下所示是 MindSpeed LLM 最新发布的 26.1.0 版本多架构镜像 Tag，每个 Tag 均为 `x86_64` 与 `aarch64` 二合一镜像。历史版本所有 Tag 请参考 [Supported Tags](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/docker/supported_tags.md)。

| Tag | Dockerfile | Content |
| --- | --- | --- |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-openeuler24.03-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-openeuler24.03-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |
| `v26.1.0-cann9.1.0-torch_npu2.7.1.post8-950-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindSpeed-LLM/blob/26.1.0/docker/Dockerfile) | CANN 9.1.0/PyTorch 2.7.1/Triton-Ascend 3.2.2/MindSpeed 26.1.0_core_r0.12.1/MindSpeed-LLM 26.1.0/Megatron-LM core_v0.12.1/FSDPTurbo main |

## Dockerfile 归档路径

`docker/Dockerfile`

## 项目目录结构规范

Docker 项目目录遵循清晰的分层结构，便于维护和扩展：

### 目录结构

```text
docker/
├── Dockerfile                 # 统一 Dockerfile，支持多 NPU 类型
├── image_build.sh             # 镜像构建脚本
├── configure_yum_repo.sh      # 配置 yum 软件源库脚本
├── configure_apt_repo.sh      # 配置 apt 软件源库脚本
├── supported_tags.md          # 已发布及历史镜像 Tag
├── OVERVIEW.md                # 英文版说明文档
├── OVERVIEW.zh.md             # 中文版说明文档
```

## 快速开始

### 如何本地构建

#### 自定义构建基础镜像

构建脚本 `image_build.sh` 支持多种参数配置，默认值与最新发布的镜像 Tag 保持一致，可根据实际需求覆盖：

| 参数 | 说明                                  | 默认值 |
| ------ |-------------------------------------| ------------ |
| `-t, --npu-type` | NPU 类型：`910b`、`a3` 或 `950`                | `910b` |
| `-o, --os` | 操作系统：`openeuler24.03`或`ubuntu22.04` | `openeuler24.03` |
| `--no-cache` | 构建时不使用 Docker 构建缓存                          | 无 |
| `--mindspeed-llm-branch` | MindSpeed LLM 版本标识，同时作为 Git 分支名称    | `26.1.0` |
| `--mindspeed-branch` | MindSpeed 版本标识，同时作为 Git 分支名称        | `26.1.0_core_r0.12.1` |
| `--megatron-branch` | Megatron-LM 版本标识，同时作为 Git 分支名称      | `core_v0.12.1` |
| `--python-version` | Python 版本                           | `3.12` |
| `--torch-version` | PyTorch 版本                          | `2.7.1` |
| `--torch-npu-version` | TorchNPU 安装包版本                            | `2.7.1.post8` |
| `--triton-ascend-version` | Triton-Ascend 版本                        | `3.2.2` |
| `--fla-npu-branch` | flash-linear-attention-npu 分支       | `v26.1.0` |
| `--base-image-version` | 基础镜像 CANN 版本                        | `9.1.0` |
| `--base-image` | 完整基础镜像名称，当设置不为空时会原样传入拉取镜像           | 无 |
| `--cleanup-on-fail` | 构建失败时清理悬空的镜像和容器           | 无 |

**提示：** 最新发布的镜像 Tag 和 `image_build.sh` 构建脚本均支持 `910b`（Atlas A2 训练系列产品）、`a3`（Atlas A3 训练系列产品）和 `950`（Ascend 950 系列产品）。

#### 基础构建示例

仅需传入想要修改的参数，未指定的参数会自动使用上表中的默认值。

```bash
cd docker

# 使用全部默认值构建（910b + openEuler24.03）
bash image_build.sh

# 自定义 NPU 类型和操作系统
bash image_build.sh -t 950 -o ubuntu22.04

# 自定义 CANN、PyTorch 和 TorchNPU 软件包版本
bash image_build.sh \
  --base-image-version 9.1.0 \
  --torch-version 2.7.1 \
  --torch-npu-version 2.7.1.post8

# 修改源码分支
bash image_build.sh \
  --mindspeed-llm-branch 26.1.0 \
  --mindspeed-branch 26.1.0_core_r0.12.1 \
  --megatron-branch core_v0.12.1

# 修改输出镜像名称
bash image_build.sh -i myproject/mindspeed-llm:custom
```

#### 自动下载功能说明

构建脚本支持自动下载以下资源，请确保网络通畅：

**基础镜像：** 当指定`--base-image`且本地不存在时自动拉取，镜像 tag 和 CANN 基础镜像名中的“芯片信息”必须使用小写，例如`910b`、`a3`和`950`。完整`--base-image`会原样传入，因此其中的 tag 必须与已发布的 CANN 镜像名完全一致。未指定`--npu-type`时，脚本会从基础镜像 Tag 中自动识别这三种 NPU 类型。

```bash
# 指定 910b 基础镜像，脚本会自动识别 NPU 类型
cd docker
bash image_build.sh \
  --base-image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.12
```

#### flash-linear-attention-npu 算子编译

镜像构建过程中会在 clone `flash-linear-attention-npu` 后自动 source CANN 环境，并编译安装 GDN 相关自定义算子 run 包和 `torch_custom/fla_npu` whl 包。

`--soc` 默认按机型自动映射：

| 机型 | FLA NPU `--soc` |
| ------ | ------ |
| `910b` | `ascend910b` |
| `a3` | `ascend910_93` |
| `950` | `ascend950` |

如需覆盖默认映射，可通过构建参数指定：

```bash
bash image_build.sh --fla-npu-soc ascend910_93
```

FLA NPU 算子列表在 `docker/image_build.sh` 的 `FLA_NPU_OPS` 数组中统一维护。后续如需新增算子，只需向该数组追加算子名称即可，脚本会自动拼接为 `build.sh --ops` 所需的逗号分隔参数。

### 运行 MindSpeed LLM 容器

**重要提示：** 由于不同模型的依赖环境存在差异，镜像中仅预安装了`PyTorch`、`TorchNPU`基础依赖包。用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的依赖环境。

#### 运行容器

镜像名使用`docker images`中的`REPOSITORY:TAG`，例如`mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12`。

```bash
# 基本运行
docker run -it --rm \
  mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12 bash

# 使用 NPU 设备运行（示例：设备 /dev/davinci1）
# 假设您的 NPU 设备安装在 /dev/davinci1 上，并且 NPU 驱动程序安装在 /usr/local/Ascend 上：
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

# 进入已启动容器
docker exec -it mindspeed-llm /bin/bash
```

#### 内置环境

镜像包含以下预配置环境：

| 环境 | 说明 | 工作目录 |
| ------ | ------ | --------- |
| base | 基础环境，包含`PyTorch`，`TorchNPU`，`MindSpeed LLM`，`MindSpeed`，`Megatron-LM`，`FSDPTurbo`，`Triton-Ascend` | `/workspace/MindSpeed-LLM` |

## 二次开发

基于此镜像创建自定义Dockerfile：

```dockerfile
FROM mindspeed-llm:v26.1.0-cann9.1.0-torch_npu2.7.1.post8-910b-openeuler24.03-py3.12

RUN pip install your-package==1.0.0

COPY . /workspace/your-project

WORKDIR /workspace/your-project
```

构建并运行（示例：设备 /dev/davinci1）：

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

### 软件栈

| 组件 | 版本       |
| ------ |----------|
| CANN | 9.1.0    |
| Python | 3.12     |
| PyTorch | 2.7.1    |
| TorchNPU | 26.1.0    |
| Triton-Ascend | 3.2.2    |
| MindSpeed LLM | 26.1.0   |

### 兼容性说明

- 当前版本采用统一 Dockerfile + 构建脚本结构，支持可配置的 CANN 基础镜像选择。
- 最新发布镜像使用 `CANN 9.1.0`、`TorchNPU 2.7.1.post8` 和 `Python 3.12`。
- 已发布镜像覆盖 `910b`、`a3`、`950`，并同时提供 `openEuler24.03`、`ubuntu22.04`。镜像仓库中的 Tag 通过多架构 manifest 合并 `x86_64` 与 `aarch64` 镜像且不带架构后缀，本地构建的镜像默认仍保留宿主机架构后缀。
- `MindSpeed-LLM`克隆到 `/workspace/MindSpeed-LLM`，`MindSpeed` 克隆到 `/workspace/MindSpeed`，`Megatron-LM`克隆到 `/workspace/Megatron-LM`。
- 镜像安装`PyTorch`、`TorchNPU`、`MindSpeed-LLM`、`MindSpeed`、`Megatron-LM` 以及 `requirements.txt` 中的 `Python`依赖。

## 许可证

MindSpeed LLM 基于 Apache License 2.0 许可证发布。详见 [LICENSE](https://gitcode.com/Ascend/MindSpeed-LLM/blob/master/LICENSE) 文件。

与所有 Docker 镜像一样，这些镜像可能还包含受其他许可证约束的其他软件（例如基础发行版中的 Bash，以及所包含主要软件的任何直接或间接依赖项）。

对于预构建镜像的任何使用，镜像用户有责任确保对此镜像的任何使用符合其中包含的所有软件的相关许可证。

## 免责声明

发布的昇腾软件镜像均是社区版本，不对商业负责、仅作为生产实践的参考。

# MindSpeed LLM Docker 镜像概述

## 快速参考

| 项目 | 说明 |
| ------ | ------ |
| **镜像名称** | mindspeed-llm |
| **维护者** | MindSpeed LLM 团队 |
| **源码仓库** | [https://gitcode.com/Ascend/MindSpeed-LLM](https://gitcode.com/Ascend/MindSpeed-LLM) |
| **Dockerfile 路径** | `docker/` |
| **许可证** | Apache-2.0 |

## 镜像 Tag 关键字段描述

镜像 Tag 命名遵循模板：`{版本号}-{芯片信息}-{操作系统}-py{Python版本}-{架构类型}`

| 字段 | 说明 | 示例值 |
| ------ | ------ | -------- |
| 版本号 | MindSpeed LLM 版本标识，同时也是 Git 分支名称 | 26.0.0 |
| 芯片信息 | NPU 芯片类型（小写） | a3, 910b |
| 操作系统 | 操作系统 | openeuler24.03 |
| Python版本 | Python 版本 | py3.11 |
| 架构类型 | CPU 架构 | aarch64 |

### 示例 Tag

| Tag | NPU | 操作系统 | Python | 架构 |
| ----- | ----- | --------- | -------- | ------ |
| `26.0.0-a3-openeuler24.03-py3.11-aarch64` | A3 | openEuler 24.03 | 3.11 | aarch64 |
| `26.0.0-910b-openeuler24.03-py3.11-aarch64` | 910B | openEuler 24.03 | 3.11 | aarch64 |

## Dockerfile 归档路径

| NPU | 操作系统 | Dockerfile 路径 |
| ----- | --------- | ---------------- |
| A3 | openEuler | `docker/Dockerfile` |
| 910B | openEuler | `docker/Dockerfile` |

Dockerfile 命名遵循模板：`Dockerfile[.{芯片信息}.{操作系统}.{其他字段}]`

- 统一的 `Dockerfile` 通过构建参数支持所有 NPU 类型和操作系统版本
- 字段间连接符使用 `.`
- 字段内连接符使用 `-`
- 芯片信息使用小写（a3, 910b）
- 操作系统使用 PascalCase（openEuler）

## 项目目录结构规范

Docker 项目目录遵循清晰的分层结构，便于维护和扩展：

### 目录结构

```text
docker/
├── Dockerfile                 # 统一 Dockerfile，支持多 NPU 类型
├── image_build.sh             # 镜像构建脚本
├── configure_yum_repo.sh      # 配置 yum 软件源库脚本
├── OVERVIEW.md                # 英文版说明文档
├── OVERVIEW.zh.md             # 中文版说明文档
```

## 快速开始

### 1、镜像构建指导

#### 自定义构建基础镜像

构建脚本 `image_build.sh` 支持多种参数配置，以下默认值仅为示例，请根据实际需求调整：

| 参数 | 说明 | 默认值（示例） |
| ------ | ------ | ------------ |
| `-t, --npu-type` | NPU 类型：A3 或 910B | 无（必需） |
| `-o, --os` | 操作系统：openeuler24.03 | openeuler24.03 |
| `-v, --version` | MindSpeed LLM 版本标识，同时作为 Git 分支名称和脚本目录选择依据 | 26.0.0 |
| `--torch-version` | PyTorch 版本 | 2.9.0 |
| `--torch-npu-version` | torch-npu 版本 | 2.9.0 |
| `--base-image-version` | 基础镜像 CANN 版本 | 9.0.0-beta.2 |

**提示：** 当前的NPU类型为A2和A3，A5待搭建

#### 基础构建示例

```bash
cd docker

# 构建 910B 镜像（默认）
bash image_build.sh -t 910B

# 构建 A3 镜像
bash image_build.sh -t A3

# 构建 A3 + openEuler 镜像
bash image_build.sh -t A3 -o openEuler24.03

# 指定 PyTorch 版本构建
bash image_build.sh -t A3 --torch-version 2.9.0 --torch-npu-version 2.9.0

# 指定基础镜像版本构建
bash image_build.sh -t A3 --base-image-version 9.0.0-beta.2

# 指定版本构建
bash image_build.sh -t A3 -v 26.0.0
```

#### 自动下载功能说明

构建脚本支持自动下载以下资源，请确保网络通畅：

**基础镜像：** 当指定 `--base-image` 且本地不存在时自动拉取

### 2、镜像使用指导

**重要提示：** 由于不同模型的依赖环境存在差异，镜像中仅预安装了 torch、torch_npu 基础依赖包。用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的依赖环境。

#### 运行镜像

```bash
# 基本运行
docker run -it --rm \
  mindspeed-llm:26.0.0-910b-openeuler24.03-py3.11-aarch64 bash

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
  mindspeed-llm:26.0.0-910b-openeuler24.03-py3.11-aarch64 \
  /bin/bash
```

#### 内置环境

镜像包含以下预配置环境：

| 环境 | 说明 | 工作目录 |
| ------ | ------ | --------- |
| base | 基础环境，包含 PyTorch、torch_npu、MindSpeed LLM、MindSpeed、Megatron-LM | /workspace/MindSpeed-LLM |

**环境说明：**

- 考虑到不同模型的依赖环境存在差异，镜像中仅预安装了 torch、torch_npu基础依赖包。
- 用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的依赖环境。

## 二次开发

基于此镜像创建自定义 Dockerfile：

```dockerfile
FROM mindspeed-llm:26.0.0-910b-openeuler24.03-py3.11-aarch64

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
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  my-mindspeed-app:latest bash
```

### 软件栈

| 组件 | 版本 |
| ------ | ------ |
| CANN | 9.0.0-beta.2 |
| Python | 3.11 |
| Miniconda | 26.1.1-1 |
| PyTorch | 2.9.0 |
| torch_npu | 2.9.0 |
| MindSpeed LLM | 26.0.0 |

### 兼容性变更说明

#### 26.0.0 版本

- 初始发布版本
- 基于 CANN 9.0.0-beta.2
- PyTorch 2.9.0 + torch_npu 2.9.0
- Python 3.11（Miniconda 26.1.1-1）
- 支持 openEuler 24.03

## 许可证

MindSpeed LLM 基于 Apache License 2.0 许可证发布。详见 [LICENSE](../LICENSE) 文件。

与所有 Docker 镜像一样，这些镜像可能还包含受其他许可证约束的其他软件（例如基础发行版中的 Bash，以及所包含主要软件的任何直接或间接依赖项）。

对于预构建镜像的任何使用，镜像用户有责任确保对此镜像的任何使用符合其中包含的所有软件的相关许可证。

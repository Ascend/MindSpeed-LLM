# MindSpeed LLM Complete NPU Dev Container

本目录提供 MindSpeed-LLM 的完整 NPU 开发容器配置。使用 VS Code Dev
Containers 可以在隔离环境中编辑、调试、运行 UT，并执行需要 Ascend NPU
的训练、推理、评估和权重转换任务。

`devcontainer.json` 会将 VS Code 当前打开的 MindSpeed-LLM 仓库挂载到
`/workspace/MindSpeed-LLM`。宿主机和容器使用同一份项目代码，任一侧的
修改或删除都会同步到另一侧。MindSpeed、Megatron-LM 和 FSDPTurbo 则由
镜像构建并放在 `/workspace` 下，不与宿主机目录同步。

## 环境组成

| 组件 | 默认值 |
| --- | --- |
| 基础镜像 | CANN 9.0.0 / 910B / openEuler 24.03 / Python 3.11 |
| PyTorch | 2.7.1 |
| TorchNPU | 2.7.1 |
| Triton-Ascend | 3.2.1 |
| MindSpeed | `26.0.0_core_r0.12.1` |
| Megatron-LM | `core_v0.12.1` |
| FSDPTurbo | 仓库默认分支 |

容器还安装 torchvision、torchaudio、pytest、pytest-xdist、pytest-cov、
ruff、pre-commit、pandas、safetensors、编译工具和常用 Linux 调试工具。
项目 `requirements.txt` 中的通用模型依赖会在容器创建后自动安装。

MindSpeed-LLM 和外部源码依赖在 `/workspace` 下并列：

```text
/workspace/
├── MindSpeed-LLM
├── MindSpeed
├── Megatron-LM
└── FSDPTurbo
```

容器创建后还会在工作区生成 `megatron` 软链接，兼容需要仓库内 Megatron
包布局的脚本。

## 前置条件

- 必须使用安装了 Ascend NPU 的 Linux 开发机；Windows Docker Desktop
  无法直接透传 Ascend NPU。
- 宿主机已经安装 Docker、NPU 驱动和固件。
- 宿主机执行 `npu-smi info` 正常。
- VS Code 已安装 **Dev Containers** 扩展。
- 构建机能够访问华为云 SWR、PyPI、Triton-Ascend PyPI、GitCode 和
  GitHub。
- 宿主机存在以下默认路径：

  ```text
  /usr/local/Ascend/driver
  /usr/local/Ascend/firmware
  /usr/local/dcmi
  /usr/local/bin/npu-smi
  /etc/ascend_install.info
  /data
  ```

如果驱动、固件、DCMI、`npu-smi` 或数据目录位于其他位置，需要修改
`devcontainer.json` 中相应的 `--volume` 参数。

## 启动和构建

以下操作假设 VS Code 已经打开包含 `.devcontainer` 配置的
MindSpeed-LLM 仓库根目录。

### 1. 首次构建并进入容器

在已经打开仓库根目录的 VS Code 窗口中：

1. 按 `F1` 或 `Ctrl+Shift+P` 打开命令面板。
2. 输入并执行 **Dev Containers: Reopen in Container**。
3. 如果 VS Code 弹出“检测到 Dev Container 配置”的通知，也可以直接点击
   **Reopen in Container**。
4. 等待镜像构建、容器创建和 `postCreateCommand` 初始化完成。
5. 构建成功后，VS Code 会自动重新加载窗口；左下角应显示当前已连接到
   Dev Container。

首次构建需要拉取基础镜像、安装 Python 包并克隆三个外部源码仓库，耗时
取决于网络环境。构建期间可以点击右下角进度通知中的 **Show Log** 查看
详细日志。如果没有看到通知，可按 `F1` 执行
**Dev Containers: Show Container Log**。

初始化完成时，日志会显示：

```text
MindSpeed LLM complete NPU development environment is ready.
```

打开 VS Code 集成终端后，执行：

```bash
pwd
echo "${PYTHONPATH}"
npu-smi info
python3 -m pip check
```

工作目录应为 `/workspace/MindSpeed-LLM`，`npu-smi info` 应能看到宿主机
NPU。`pip check` 用于查看依赖冲突；发现冲突时会输出告警，但不会阻止
容器初始化。

### 2. 后续重新进入

容器创建成功后，再次打开仓库时通常会自动连接现有容器。如果当前处于
宿主机环境，按 `F1` 执行：

```text
Dev Containers: Reopen in Container
```

VS Code 会优先复用现有容器和镜像，不会每次都重新安装所有依赖。

### 3. 配置变更后重建

修改以下内容后，需要按 `F1` 执行
**Dev Containers: Rebuild and Reopen in Container**：

- `.devcontainer/Dockerfile`
- `.devcontainer/devcontainer.json`
- CANN、PyTorch、TorchNPU 或 Triton-Ascend 版本
- MindSpeed、Megatron-LM 或模型专属依赖
- NPU 驱动及数据目录挂载参数

如果怀疑旧镜像层或依赖缓存导致问题，执行：

```text
Dev Containers: Rebuild Container Without Cache
```

无缓存重建会重新执行全部 Dockerfile 步骤，耗时明显更长。

### 4. 退出 Dev Container

需要回到宿主机目录时，按 `F1` 执行：

```text
Dev Containers: Reopen Folder Locally
```

### 容器内部构建流程

1. 拉取 CANN 基础镜像。
2. 安装编译、网络和调试工具。
3. 安装 PyTorch、torchvision、torchaudio 和 TorchNPU。
4. 安装 Triton-Ascend 与测试工具。
5. 克隆并以 editable 模式安装 MindSpeed。
6. 克隆并以 editable 模式安装 FSDPTurbo。
7. 克隆指定版本的 Megatron-LM，并加入 `PYTHONPATH`。
8. 创建容器并挂载当前工作区、日志和宿主机 `/data`。
9. 安装当前仓库 `requirements.txt` 和可选的模型专属 requirements。
10. 执行 `pip check`；发现依赖冲突时输出告警并继续初始化。

### 常见启动问题

- **Docker 权限错误**：确认远程用户可以执行 `docker version`，必要时由
  管理员配置 Docker 用户组。
- **宿主机目录挂载失败**：检查“前置条件”列出的 Ascend 路径和 `/data`
  是否存在，并修改 `devcontainer.json` 中不匹配的 `--volume`。
- **基础镜像、pip 或 Git 克隆失败**：检查服务器到 SWR、PyPI、GitCode
  和 GitHub 的网络连接。
- **`postCreateCommand` 失败**：在 Container Log 中查找第一个失败的
  `pip install` 或其他命令，修复后重新执行
  **Rebuild and Reopen in Container**。`pip check` 报告的依赖冲突只会
  输出告警，不会导致初始化失败。
- **修改配置后没有生效**：执行
  **Dev Containers: Rebuild Container Without Cache**。

## NPU 透传

容器默认使用以下参数：

- `--privileged`
- host network、PID 和 IPC
- 512 GB `/dev/shm` 上限
- 只读挂载宿主机 Ascend 驱动、固件、DCMI 和 `npu-smi`
- 将宿主机 `/data` 读写挂载到容器 `/data`

这些权限用于 NPU、HCCL 和分布式开发，只应在可信开发机及可信代码上使用。
`--shm-size=512g` 是共享内存上限，不会在容器启动时立即占用等量物理内存。

进入容器后可以执行：

```bash
npu-smi info
python3 -c "import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available())"
python3 -c "import mindspeed, megatron; print('MindSpeed and Megatron are available')"
python3 -m pip check
```

## 模型专属依赖

根目录 `requirements.txt` 会始终安装，覆盖仓库声明的通用模型依赖。不同
模型仍可能需要额外 Python 包、私有 wheel 或相互冲突的版本，因此不能
安全地把所有模型的专属依赖同时硬编码到基础镜像。

如目标模型提供独立 requirements 文件，在 `devcontainer.json` 中设置：

```json
"MODEL_REQUIREMENTS_FILE": "/workspace/MindSpeed-LLM/path/to/model-requirements.txt"
```

也可以指向挂载到 `/data` 下的文件：

```json
"MODEL_REQUIREMENTS_FILE": "/data/dependencies/qwen3-requirements.txt"
```

重建容器后，该文件会在根目录 `requirements.txt` 之后安装。模型权重和
训练数据不会复制进镜像，应放在宿主机 `/data` 或改用其他运行时挂载。

## 修改版本

核心版本均通过 `devcontainer.json` 的 `build.args` 控制：

```json
"TORCH_VERSION": "2.7.1",
"TORCH_NPU_VERSION": "2.7.1",
"TRITON_ASCEND_VERSION": "3.2.1",
"MINDSPEED_BRANCH": "26.0.0_core_r0.12.1",
"MEGATRON_BRANCH": "core_v0.12.1"
```

修改 CANN、Python、操作系统或芯片类型时，需要替换 `BASE_IMAGE`。切换
Ubuntu 镜像时还要把 `OS_FAMILY` 改为 `ubuntu`。PyTorch、TorchNPU、
Triton-Ascend、CANN 和 MindSpeed/Megatron 分支必须选择兼容组合。

FSDPTurbo 直接使用其仓库默认分支，不固定 tag 或 commit。因此重建环境
时可能获得不同代码；如果默认分支更新了 PyTorch、TorchNPU 或
Transformers 版本约束，需要同步调整本配置中的核心依赖版本。

## 注意事项

- 当前配置没有单独的 Docker ignore 文件，因此 Docker 会发送整个仓库
  作为构建上下文；仓库中存在大型权重或数据时会显著拖慢构建。
- 当前配置不使用 BuildKit cache mount 或持久化 pip 缓存卷，因此不依赖
  Buildx 的缓存语法，但重新构建时会重复下载系统包和 Python 包。
- 当前 `devcontainer.json` 将 `CONFIGURE_REPOSITORY` 设置为 `true`，构建
  时会配置系统软件源。Dockerfile 中的默认值 `false` 仅在直接构建且未
  传入该参数时生效。
- 修改挂载参数只需重建容器，不一定需要清除镜像缓存。

# MindSpeed LLM CANN 安装路径批量替换指南

## 背景

出海版本中，CANN/HDK 的默认安装路径已由 `/usr/local/Ascend` 变更为 `/usr/local/npu`。
MindSpeed LLM 仓库中的训练脚本、安装文档等均使用了硬编码的 `/usr/local/Ascend` 路径。
若出海版本机器上的实际安装路径为 `/usr/local/npu`，需在使用前完成批量替换，确保环境变量可以正常加载。

本指南提供使用 `replace_ascend_path.py` 脚本进行批量路径替换的完整步骤。

---

## 前置条件

- Python 3.10+
- 拥有仓库目录的读写权限
- 建议在执行替换前，先通过 git 将当前状态提交或备份

---

## 受影响的文件范围

| 文件类型 | 说明 | 典型路径示例 |
|---------|------|-------------|
| Shell 脚本（`.sh`）| 各类运行脚本，包括但不限于：数据预处理、权重转换、预训练、微调、评估、推理、测试等流程 | `examples/*/*.sh`、`tests/*/*.sh` |
| Markdown 文档（`.md`）| 全量说明文档，包括但不限于：安装指南、快速上手、各任务指南、特性说明等 | `docs/zh/install_guide.md`、`docker/OVERVIEW.md`、`docs/zh/pytorch/*/*.md`  |
| RST 文档（`.rst`）| reStructuredText 风格说明文档 | `docs/*/*.rst` |
| TXT 文档（`.txt`）| 普通文本说明文件或配置说明 | `requirements.txt` |
| Python 文件（`.py`）| 源码（如有路径引用） | 各模块源文件 |
| Dockerfile | Docker 镜像构建脚本 | `docker/Dockerfile` |

> 路径变体说明：仓库中存在以下几种 Ascend 路径引用，均会被一并替换：
>
> - `/usr/local/Ascend/cann/set_env.sh`（最常见，环境变量初始化）
> - `/usr/local/Ascend/ascend-toolkit/set_env.sh`（Ascend Toolkit 初始化）
> - `/usr/local/Ascend/nnal/atb/set_env.sh`（ATB 库初始化）
> - `/usr/local/Ascend/driver/lib64/`（Docker 挂载路径）

---

## 使用步骤

### 1. 进入仓库根目录

```bash
cd /path/to/MindSpeed-LLM
```

### 2. 预览将要修改的内容（推荐）

在实际修改前，先以 `--dry-run` 模式确认变更范围：

```bash
python3 tests/tools/replace_ascend_path.py --dry-run
```

输出示例：

```bash
[DRY RUN] Path replacement: /usr/local/Ascend -> /usr/local/npu
Scan directory : /path/to/MindSpeed-LLM
File types     : .md, .py, .rst, .sh, .txt + Dockerfile
------------------------------------------------------------
Found XXX candidate file(s), processing...

  [would replace   1] docker/Dockerfile
  [would replace   7] docker/OVERVIEW.md
  [would replace   7] docker/OVERVIEW.zh.md
  ...

============================================================
[DRY RUN] XXX file(s) would be modified, XXX replacement(s) total.
          Remove --dry-run to apply changes.
```

### 3. 执行批量替换

确认预览无误后，执行实际替换：

```bash
# 默认：将 /usr/local/Ascend 替换为 /usr/local/npu
python3 scripts/replace_ascend_path.py
```

执行完毕后，脚本会输出修改的文件数和替换总次数。

### 4. 验证替换结果

```bash
# 检查是否还有未替换的路径（结果应为 0）
grep -r "/usr/local/Ascend" . \
--include='.sh' \
--include='.md' \
--include='.rst' \
--include='.py' \
--include='.txt' \
--exclude='replace_ascend_path.py' \
--exclude='replace_ascend_path_guide.md' \
--exclude-dir='.git' \
| wc -l
```

---

## 执行后验证

### 1. 环境变量加载验证

```bash
# 验证新路径下的 set_env.sh 文件存在
ls /usr/local/npu/ascend-toolkit/set_env.sh

# 加载环境变量
# 根据实际情况修改 ascend-toolkit 路径
source /usr/local/npu/ascend-toolkit/set_env.sh

# 验证环境变量生效
echo $ASCEND_HOME_PATH
```

### 2. 组件安装验证

```bash
# 验证MindSpeed LLM 安装成功
python3 -c "import mindspeed_llm; print('MindSpeed LLM installed successfully')"

# 验证 NPU 可用
python3 -c "import torch_npu; print('NPU available:', torch_npu.npu.is_available())"
```

### 3. 核心功能冒烟验证

参考对应模型的readme进行配置，验证训练流程可正常启动

```bash
source /usr/local/npu/ascend-toolkit/set_env.sh

# 运行示例脚本（以具体模型为准）
bash examples/<model_name>/pretrain_<model_name>.sh
```

---

## 完整脚本参数说明

```bash
usage: replace_ascend_path.py [-h] [--source SOURCE] [--target TARGET]
                               [--dir DIR] [--extensions EXT [EXT ...]]
                               [--dry-run]

选项：
  -h, --help            显示帮助信息
  --source SOURCE       源路径（默认：/usr/local/Ascend）
  --target TARGET       目标路径（默认：/usr/local/npu）
  --dir DIR             扫描目录（默认：当前目录 .）
  --extensions EXT...   文件扩展名白名单（默认：.sh .md .rst .py .txt）
  --dry-run             仅预览变更，不修改文件
```

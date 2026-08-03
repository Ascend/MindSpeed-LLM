# syntax=docker/dockerfile:1.7

# Keep these defaults aligned with docker/Dockerfile and docker/image_build.sh.
ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11
FROM ${BASE_IMAGE}

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Core framework versions and external source locations.
ARG OS_FAMILY=openeuler
ARG CONFIGURE_REPOSITORY=false
ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1
ARG TRITON_ASCEND_VERSION=3.2.1
ARG MINDSPEED_BRANCH=26.0.0_core_r0.12.1
ARG MEGATRON_BRANCH=core_v0.12.1
ARG MINDSPEED_REPOSITORY=https://gitcode.com/ascend/MindSpeed.git
ARG FSDP_TURBO_REPOSITORY=https://gitcode.com/Ascend/FSDPTurbo.git
ARG MEGATRON_REPOSITORY=https://github.com/NVIDIA/Megatron-LM.git

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit \
    MINDSPEED_DEPS_ROOT=/workspace \
    PYTHONPATH=/workspace/MindSpeed-LLM:/workspace/Megatron-LM \
    TEST_MODEL=true

# Repository mirror helpers are only used when CONFIGURE_REPOSITORY=true.
COPY docker/configure_yum_repo.sh docker/configure_apt_repo.sh /tmp/repo-config/

# Install compilers, build tools, and common diagnostics for development.
RUN set -e; \
    if [[ "${OS_FAMILY}" == "openeuler" ]]; then \
        if [[ "${CONFIGURE_REPOSITORY}" == "true" ]]; then \
            bash /tmp/repo-config/configure_yum_repo.sh; \
        fi; \
        yum install -y \
            ca-certificates curl wget git openssh-clients \
            gcc gcc-c++ make cmake ninja-build \
            iproute procps-ng which findutils tar gzip unzip \
            rsync jq lsof dos2unix vim-minimal less numactl; \
    elif [[ "${OS_FAMILY}" == "ubuntu" ]]; then \
        if [[ "${CONFIGURE_REPOSITORY}" == "true" ]]; then \
            bash /tmp/repo-config/configure_apt_repo.sh; \
        else \
            apt-get update; \
        fi; \
        apt-get install -y --no-install-recommends \
            ca-certificates curl wget git openssh-client \
            gcc g++ make cmake ninja-build \
            iproute2 procps findutils tar gzip unzip \
            rsync jq lsof dos2unix vim less numactl \
            python3-dev python3-pip; \
    else \
        echo "Unsupported OS_FAMILY: ${OS_FAMILY}" >&2; \
        exit 1; \
    fi; \
    rm -rf /tmp/repo-config

# Retry pip indexes in order instead of querying multiple indexes at once.
RUN printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -u' \
        'indexes="${PIP_INDEX_URLS:-https://repo.huaweicloud.com/repository/pypi/simple https://mirrors.aliyun.com/pypi/simple/}"' \
        'for index_url in ${indexes}; do' \
        '    echo "Trying pip index: ${index_url}"' \
        '    if env -u PIP_EXTRA_INDEX_URL -u PIP_FIND_LINKS PIP_CONFIG_FILE=/dev/null python3 -m pip install --index-url "${index_url}" "$@"; then' \
        '        exit 0' \
        '    fi' \
        '    echo "pip install failed with ${index_url}; trying the next index." >&2' \
        'done' \
        'echo "pip install failed with all configured indexes." >&2' \
        'exit 1' \
        > /usr/local/bin/pip-install-with-fallback && \
    chmod +x /usr/local/bin/pip-install-with-fallback

# Install the PyTorch/NPU stack and the Python tools used by tests and linting.
# x86_64 uses the official CPU wheel because NPU execution is provided by torch-npu.
RUN set -e; \
    python3 -m pip config set global.index-url \
        https://repo.huaweicloud.com/repository/pypi/simple; \
    arch="$(uname -m)"; \
    if [[ "${arch}" == "x86_64" ]]; then \
        python3 -m pip install \
            "torch==${TORCH_VERSION}" torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu; \
    elif [[ "${arch}" == "aarch64" || "${arch}" == "arm64" ]]; then \
        pip-install-with-fallback \
            "torch==${TORCH_VERSION}" torchvision torchaudio; \
    else \
        echo "Unsupported architecture: ${arch}" >&2; \
        exit 1; \
    fi && \
    pip-install-with-fallback \
        "torch-npu==${TORCH_NPU_VERSION}" && \
    python3 -m pip install \
        "triton-ascend==${TRITON_ASCEND_VERSION}" \
        --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple && \
    pip-install-with-fallback \
        coverage pandas pre-commit pytest pytest-cov pytest-xdist \
        ruff safetensors xxhash

# Keep third-party source checkouts alongside the MindSpeed-LLM workspace.
# MindSpeed is installed in editable mode to simplify framework-level debugging.
RUN mkdir -p "${MINDSPEED_DEPS_ROOT}" && \
    git clone --depth 1 --branch "${MINDSPEED_BRANCH}" \
        "${MINDSPEED_REPOSITORY}" "${MINDSPEED_DEPS_ROOT}/MindSpeed" && \
    pip-install-with-fallback \
        -r "${MINDSPEED_DEPS_ROOT}/MindSpeed/requirements.txt" && \
    pip-install-with-fallback \
        -e "${MINDSPEED_DEPS_ROOT}/MindSpeed"

# FSDPTurbo intentionally follows its repository default branch.
RUN git clone --depth 1 \
        "${FSDP_TURBO_REPOSITORY}" "${MINDSPEED_DEPS_ROOT}/FSDPTurbo" && \
    pip-install-with-fallback \
        -e "${MINDSPEED_DEPS_ROOT}/FSDPTurbo"

# Megatron-LM is imported directly from its source tree through PYTHONPATH.
# Abort stalled transfers and retry transient GitHub connection failures.
RUN set -e; \
    target="${MINDSPEED_DEPS_ROOT}/Megatron-LM"; \
    cloned=false; \
    for attempt in 1 2 3; do \
        rm -rf "${target}"; \
        echo "Cloning Megatron-LM, attempt ${attempt}/3..."; \
        if timeout 180s env \
            GIT_HTTP_LOW_SPEED_LIMIT=1024 \
            GIT_HTTP_LOW_SPEED_TIME=30 \
            git -c http.version=HTTP/1.1 clone \
                --depth 1 --single-branch --no-tags \
                --branch "${MEGATRON_BRANCH}" \
                "${MEGATRON_REPOSITORY}" "${target}"; then \
            cloned=true; \
            break; \
        fi; \
        echo "Megatron-LM clone failed; retrying in 5 seconds..." >&2; \
        sleep 5; \
    done; \
    [[ "${cloned}" == "true" ]] || { \
        echo "Megatron-LM clone failed after 3 attempts." >&2; \
        exit 1; \
    }

# Load CANN/ATB variables for interactive and non-interactive Bash sessions.
RUN printf '%s\n' \
        '#!/usr/bin/env bash' \
        'if [ -f /usr/local/Ascend/cann/set_env.sh ]; then' \
        '    . /usr/local/Ascend/cann/set_env.sh' \
        'elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then' \
        '    . /usr/local/Ascend/ascend-toolkit/set_env.sh' \
        'fi' \
        'if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then' \
        '    . /usr/local/Ascend/nnal/atb/set_env.sh' \
        'fi' \
        'export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}"' \
        'export MINDSPEED_DEPS_ROOT="${MINDSPEED_DEPS_ROOT:-/workspace}"' \
        'export PYTHONPATH="/workspace/MindSpeed-LLM:${MINDSPEED_DEPS_ROOT}/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"' \
        'export TEST_MODEL="${TEST_MODEL:-true}"' \
        > /etc/profile.d/mindspeed-llm.sh && \
    chmod 0644 /etc/profile.d/mindspeed-llm.sh && \
    printf '%s\n' \
        '[ -f /etc/profile.d/mindspeed-llm.sh ] && source /etc/profile.d/mindspeed-llm.sh' \
        >> /root/.bashrc

ENV BASH_ENV=/etc/profile.d/mindspeed-llm.sh

# The repository itself is mounted here by devcontainer.json at runtime.
RUN mkdir -p /workspace/MindSpeed-LLM

WORKDIR /workspace/MindSpeed-LLM

# Keep the development container alive with an interactive shell by default.
CMD ["/bin/bash"]

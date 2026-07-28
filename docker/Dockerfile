# ============================================
# MindSpeed LLM Docker Image
# NPU Type: Configurable (910b, a3, etc.)
# Supports: x86_64 and aarch64 (ARM)
# Supports: openEuler and Ubuntu
# ============================================

# Global arguments for base image selection
ARG OS=openeuler24.03
ARG BASE_IMAGE_VERSION=9.0.0
ARG NPU_TYPE=910b
ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=""

# ------------------------------
# Stage 1: Base Image
# ------------------------------
FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py${PYTHON_VERSION}} AS base

# Switch to root user
USER root

ARG OS_FAMILY=openeuler

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Detect architecture (x86_64 or aarch64)
RUN ARCH=$(uname -m) && \
    echo "Detected CPU architecture: ${ARCH}" && \
    if [ "$ARCH" = "x86_64" ]; then \
        echo "Architecture type: x86"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        echo "Architecture type: ARM"; \
    else \
        echo "ERROR: Unsupported architecture: $ARCH"; \
        exit 1; \
    fi

# Configure DNS for stable network connection
RUN if [ -f /etc/resolv.conf ]; then \
        echo "Existing resolv.conf:" && cat /etc/resolv.conf; \
    else \
        echo "No resolv.conf found, creating one..."; \
    fi && \
    echo "nameserver 114.114.114.114" > /etc/resolv.conf && \
    echo "nameserver 8.8.8.8" >> /etc/resolv.conf && \
    echo "nameserver 223.5.5.5" >> /etc/resolv.conf && \
    echo "Updated resolv.conf:" && cat /etc/resolv.conf

# Configure repository based on OS_FAMILY (openeuler/ubuntu)
COPY configure_repo.sh /tmp/configure_repo.sh
RUN chmod +x /tmp/configure_repo.sh && \
    bash /tmp/configure_repo.sh && \
    rm /tmp/configure_repo.sh

# Install system dependencies
RUN echo "Installing system dependencies..." && \
    if [ "$OS_FAMILY" = "openeuler" ]; then \
        yum install -y \
            git \
            iproute \
            wget \
            curl \
            gcc \
            gcc-c++ \
            make \
            cmake \
        && yum clean all; \
    elif [ "$OS_FAMILY" = "ubuntu" ]; then \
        apt-get install -y \
            git \
            iproute2 \
            wget \
            curl \
            gcc \
            g++ \
            make \
            cmake \
        && apt-get clean; \
    else \
        echo "ERROR: Unsupported OS: $OS_FAMILY"; \
        exit 1; \
    fi

# ------------------------------
# Stage 2: Builder (Python env + PyTorch)
# ------------------------------
FROM base AS builder

ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1
ARG TRITON_ASCEND_VERSION=3.2.1

WORKDIR /tmp

# Configure pip source
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && \
    pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple/ https://pypi.douban.com/simple/ https://pypi.tuna.tsinghua.edu.cn/simple/" && \
    pip config set global.trusted-host "repo.huaweicloud.com mirrors.aliyun.com pypi.douban.com pypi.tuna.tsinghua.edu.cn"

# Install PyTorch and torch_npu (ONLINE INSTALL ONLY)
RUN echo "Installing PyTorch and torch_npu from PyPI..." && \
        ARCH=$(uname -m) && \
        MAX_RETRIES=3 && \
        for retry in $(seq 1 $MAX_RETRIES); do \
            echo ">>> Attempt $retry of $MAX_RETRIES"; \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            if [ "$ARCH" = "x86_64" ]; then \
                echo "Installing PyTorch for x86_64..." && \
                pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && break; \
            elif [ "$ARCH" = "aarch64" ]; then \
                echo "Installing PyTorch for aarch64..." && \
                pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio && break; \
            fi; \
        done && \
        for retry in $(seq 1 $MAX_RETRIES); do \
            echo ">>> Installing torch-npu, attempt $retry of $MAX_RETRIES"; \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            pip install --no-cache-dir torch-npu==${TORCH_NPU_VERSION} && break; \
        done;

# Install Triton-Ascend
RUN echo "Installing Triton-Ascend..." && \
    pip install --no-cache-dir triton-ascend==${TRITON_ASCEND_VERSION} --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

# Clean pip cache
RUN pip cache purge

# ------------------------------
# Stage 3: Final Runtime Image
# ------------------------------
FROM builder AS final

# Arguments for MindSpeed-LLM
ARG MINDSPEED_LLM_BRANCH=26.0.0
ARG MINDSPEED_BRANCH=26.0.0_core_r0.12.1
ARG MEGATRON_BRANCH=core_v0.12.1

WORKDIR /workspace

# Disable Git SSL verification (before cloning)
RUN git config --global http.sslVerify false

# Install MindSpeed
RUN git clone https://gitcode.com/ascend/MindSpeed.git && \
    cd MindSpeed && \
    git checkout ${MINDSPEED_BRANCH} && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -e .

# Install FSDPTurbo
RUN git clone https://gitcode.com/Ascend/FSDPTurbo.git && \
    cd FSDPTurbo && \
    pip3 install --no-cache-dir -e . && \
    cd ..

# Install MindSpeed-LLM and Megatron-LM
RUN git clone https://gitcode.com/ascend/MindSpeed-LLM.git && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    git checkout ${MEGATRON_BRANCH} && \
    cp -r megatron ../MindSpeed-LLM/ && \
    cd ../MindSpeed-LLM && \
    git checkout ${MINDSPEED_LLM_BRANCH} && \
    mkdir logs && \
    pip3 install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace/MindSpeed-LLM

# Auto-run on login
RUN echo 'cd /workspace/MindSpeed-LLM' >> /root/.bashrc && \
    echo '[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> /root/.bashrc && \
    echo '[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh' >> /root/.bashrc

    # Environment variables
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit
